"""
Grid-world geodesic probe: keys-&-doors gridworld as a guarded product graph,
rendered top-down as an image; n factor tokens read their value from the image;
a recall-flow integrates per-factor displacement start->goal into a cost trained
to the BFS geodesic. Budget-free: one fixed test-loop count for all pairs.

Variants:
  --readout xattn      factor query tokens cross-attend to the CNN feature map (spatial read)
  --readout convspatial per-factor conv head + spatial-softmax pooling (convolutional factor encoding)
  --readout convpool   CNN -> global avg pool -> project to n factor tokens (plain conv encoding)
  --layout fixed       one fixed map (model may memorize geometry; attend only to locate agent)
  --layout random      fresh random map per instance; model MUST read walls+doors+agent from pixels;
                       held-out maps at test -> generalization across unseen layouts.

State factors: agent x, agent y, key_1..key_K (binary, monotone 0->1 on pickup).
Guard: door_i passable iff key_i collected.
"""
import argparse, itertools, random, json
import torch, torch.nn as nn, torch.nn.functional as F


# ----------------------------- grid world -----------------------------
class GridWorld:
    def __init__(self, walls, doors, keys, H, W, guard="simple"):
        self.walls, self.doors, self.keys = set(walls), list(doors), list(keys)
        self.H, self.W, self.K = H, W, len(keys)
        self.guard = guard                                     # simple: door_i needs key_i; seq: door_i needs keys 0..i
        self.C = 2 + 2 * self.K
        self.states = self._enumerate()

    def _free(self, x, y):
        return 0 <= x < self.W and 0 <= y < self.H and (x, y) not in self.walls

    def _enumerate(self):
        out = []
        for x in range(self.W):
            for y in range(self.H):
                if (x, y) in self.walls:
                    continue
                for bits in itertools.product((0, 1), repeat=self.K):
                    if all(not ((x, y) == self.keys[i] and bits[i] == 0) for i in range(self.K)):
                        out.append((x, y, bits))
        return out

    def neighbours(self, s):
        x, y, bits = s
        res = []
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if not self._free(nx, ny):
                continue
            blocked = False
            for i in range(self.K):
                if (nx, ny) == self.doors[i]:
                    need = range(i + 1) if self.guard == "seq" else (i,)   # seq: need keys 0..i
                    if any(bits[j] == 0 for j in need):
                        blocked = True
            if blocked:
                continue
            nb = list(bits)
            for i in range(self.K):
                if (nx, ny) == self.keys[i]:
                    nb[i] = 1
            res.append((nx, ny, tuple(nb)))
        return res

    def bfs(self, s):
        from collections import deque
        dist = {s: 0}; q = deque([s])
        while q:
            u = q.popleft()
            for v in self.neighbours(u):
                if v not in dist:
                    dist[v] = dist[u] + 1; q.append(v)
        return dist

    def render(self, s):
        x, y, bits = s
        img = torch.zeros(self.C, self.H, self.W)
        for (wx, wy) in self.walls:
            img[0, wy, wx] = 1.0
        img[1, y, x] = 1.0
        for i in range(self.K):
            if bits[i] == 0:
                kx, ky = self.keys[i]; img[2 + i, ky, kx] = 1.0
                dx, dy = self.doors[i]; img[2 + self.K + i, dy, dx] = 1.0
        return img


def fixed_map(H, W, K, guard='simple'):
    walls, doors, keys = set(), [], []
    rows = [int(H * (i + 1) / (K + 1)) for i in range(K)]
    for i, ry in enumerate(rows):
        dx = (i * 3 + 2) % W
        for x in range(W):
            if x != dx:
                walls.add((x, ry))
        doors.append((dx, ry))
        kx = (i * 5 + 1) % W; ky = max(0, ry - 1 - i)
        while (kx, ky) in walls:
            ky -= 1
        keys.append((kx, ky))
    return GridWorld(walls, doors, keys, H, W, guard)


def random_map(rng, H, W, K, guard='simple'):
    """K stacked wall-rows, each with a random door column and a key in the band above it."""
    walls, doors, keys = set(), [], []
    ys = sorted(rng.sample(range(1, H - 1), K))
    for i, ry in enumerate(ys):
        dx = rng.randrange(W)
        for x in range(W):
            if x != dx:
                walls.add((x, ry))
        doors.append((dx, ry))
        lo = (ys[i - 1] + 1) if i > 0 else 0
        hi = max(lo, ry - 1)
        for _ in range(50):
            kx, ky = rng.randrange(W), rng.randint(lo, hi)
            if (kx, ky) not in walls and (kx, ky) not in keys and (kx, ky) not in doors:
                break
        keys.append((kx, ky))
    # a few random loose walls in the interior (keep door columns clear for solvability)
    reserved = set(doors) | set(keys) | {(dx2, ry2 + s) for (dx2, ry2) in doors for s in (-1, 1)}
    for _ in range(rng.randint(0, (H * W) // 12)):
        cx, cy = rng.randrange(W), rng.randrange(H)
        if (cx, cy) not in reserved and (cx, cy) not in doors:
            walls.add((cx, cy))
    return GridWorld(walls, doors, keys, H, W, guard)


def build_pool(gw, per_state=None, seed=0):
    pairs = []
    for s in gw.states:
        dd = gw.bfs(s)
        items = [(s, g, d) for g, d in dd.items() if d >= 1]
        if per_state and len(items) > per_state:
            random.Random(seed).shuffle(items); items = items[:per_state]
        pairs.extend(items)
    return pairs


# ----------------------------- model -----------------------------
class Reader(nn.Module):
    """Shared perception for EVERY head: CNN over the rendered map + per-factor readout -> (B,n,d) tokens."""
    def __init__(self, C, n, d=96, heads=4, readout="xattn", HW=64):
        super().__init__()
        self.d, self.n, self.readout = d, n, readout
        self.cnn = nn.Sequential(
            nn.Conv2d(C, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, d, 3, padding=1), nn.ReLU(),
        )
        self.fid = nn.Embedding(n, d)
        if readout == "xattn":
            self.pos = nn.Parameter(torch.randn(HW, d) * 0.02)
            self.fquery = nn.Parameter(torch.randn(n, d) * 0.02)
            self.xattn = nn.MultiheadAttention(d, heads, batch_first=True)
        elif readout == "convspatial":
            self.head = nn.Conv2d(d, n, 1)                     # per-factor attention logits
        elif readout == "convpool":
            self.pool_proj = nn.Linear(d, n * d)

    def forward(self, img):                                    # (B,C,H,W) -> (B,n,d)
        B = img.shape[0]
        feat = self.cnn(img)                                   # (B,d,H,W)
        if self.readout == "xattn":
            fmap = feat.flatten(2).transpose(1, 2) + self.pos[None]
            q = (self.fquery + self.fid.weight)[None].expand(B, -1, -1)
            z, _ = self.xattn(q, fmap, fmap)
            return z + self.fid.weight[None]
        if self.readout == "convspatial":
            a = torch.softmax(self.head(feat).flatten(2), dim=-1)   # (B,n,HW)
            z = torch.einsum("bnh,bdh->bnd", a, feat.flatten(2))    # (B,n,d)
            return z + self.fid.weight[None]
        # convpool
        v = feat.mean((2, 3))                                  # (B,d)
        return self.pool_proj(v).view(B, self.n, self.d) + self.fid.weight[None]


class GWFlow(nn.Module):
    """Recall-flow integrator: cost = sum_t sum_i ||dz_i|| (arrive-gated), goal + start re-injected each step."""
    def __init__(self, C, n, d=96, heads=4, recall=1, readout="xattn", HW=64, layers=1):
        super().__init__()
        self.d, self.n, self.recall = d, n, recall
        self.reader = Reader(C, n, d, heads, readout, HW)
        lyr = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, dropout=0.0)
        self.block = nn.TransformerEncoder(lyr, num_layers=layers)   # weight-shared across loop steps
        self.role = nn.Embedding(3, d)
        self.arrive_b = nn.Parameter(torch.tensor(0.5))
        self.arrive_s = nn.Parameter(torch.tensor(3.0))

    def read(self, img):
        return self.reader(img)

    def forward(self, img_s, img_g, T):
        n = self.n
        r0, r1, r2 = (self.role.weight[i][None, None] for i in range(3))
        zs, zg = self.read(img_s), self.read(img_g)
        base = torch.cat([zg + r1, zs + r2], 1) if self.recall else (zg + r1)
        z = zs
        cost = torch.zeros(img_s.shape[0], device=img_s.device)
        for _ in range(T):
            nz = self.block(torch.cat([z + r0, base], 1))[:, :n]
            step = (nz - z).norm(dim=-1)
            gate = torch.sigmoid(self.arrive_s * ((nz - zg).norm(dim=-1) - self.arrive_b))
            cost = cost + (step * gate).sum(-1)
            z = nz
        return cost


class GWHead(nn.Module):
    """Baseline heads on the SAME Reader: tokens -> mix transformer (baselayers) -> mean-pool latent -> head.
    kind: sym (||f(s)-f(g)||_1) | iqe / mrn (torchqmet quasimetrics) | scalar (MLP on concat, no metric bias)."""
    def __init__(self, kind, C, n, d=96, heads=4, readout="xattn", HW=64, baselayers=2, latentnorm=0):
        super().__init__()
        self.kind = kind
        self.reader = Reader(C, n, d, heads, readout, HW)
        lyr = nn.TransformerEncoderLayer(d, heads, 4 * d, batch_first=True, dropout=0.0)
        self.mix = nn.TransformerEncoder(lyr, num_layers=baselayers)
        self.proj = nn.Linear(d, d)
        self.ln = nn.LayerNorm(d) if latentnorm else None
        if kind in ("iqe", "mrn"):
            import torchqmet
            self.head = torchqmet.IQE(d, dim_per_component=16) if kind == "iqe" else torchqmet.MRNFixed(d)
        elif kind == "scalar":
            self.head = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, 1))

    def emb1(self, img):
        e = self.proj(self.mix(self.reader(img)).mean(1))
        return self.ln(e) if self.ln is not None else e

    def forward(self, img_s, img_g, T=None):                   # T ignored (non-recurrent)
        es, eg = self.emb1(img_s), self.emb1(img_g)
        if self.kind == "sym":
            return (es - eg).abs().sum(-1)
        if self.kind == "scalar":
            return self.head(torch.cat([es, eg], -1)).squeeze(-1)
        return self.head(es, eg)


# ----------------------------- train / eval -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", type=int, default=8); ap.add_argument("--W", type=int, default=8)
    ap.add_argument("--K", type=int, default=1)
    ap.add_argument("--guard", choices=["simple", "seq"], default="simple")
    ap.add_argument("--layout", choices=["fixed", "random"], default="fixed")
    ap.add_argument("--readout", choices=["xattn", "convspatial", "convpool"], default="xattn")
    ap.add_argument("--maps", type=int, default=1200, help="random layout: number of maps")
    ap.add_argument("--d", type=int, default=96); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--T", type=int, default=6); ap.add_argument("--Ttest", type=int, default=25)
    ap.add_argument("--Tmin", type=int, default=-1, help="anytime: sample train loop T in [Tmin, T] each step")
    ap.add_argument("--layers", type=int, default=1, help="transformer layers per loop step")
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--recall", type=int, default=1)
    ap.add_argument("--steps", type=int, default=30000); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--valevery", type=int, default=3000); ap.add_argument("--save", type=str, default="")
    ap.add_argument("--arch", choices=["integ", "sym", "iqe", "mrn", "scalar"], default="integ",
                    help="integ = recall-flow integrator; others = baseline heads on the SAME reader")
    ap.add_argument("--baselayers", type=int, default=2, help="baseline mix-transformer depth (tunable, fair)")
    ap.add_argument("--latentnorm", type=int, default=0, help="LayerNorm on baseline latent before the head")
    ap.add_argument("--gradclip", type=float, default=1.0, help="grad-norm clip for ALL models (0 = off)")
    ap.add_argument("--tag", type=str, default="")
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = random.Random(a.seed)

    if a.layout == "fixed":
        gw = fixed_map(a.H, a.W, a.K, a.guard)
        pool = [(gw, s, g, d) for (s, g, d) in build_pool(gw, seed=a.seed)]
        test_pool = pool                                        # same map; test = extrapolation by distance
        train = [p for p in pool if p[3] <= a.D]
        n_maps = 1
    else:
        banks = [random_map(rng, a.H, a.W, a.K, a.guard) for _ in range(a.maps)]
        n_test_maps = max(1, a.maps // 6)
        train, test_pool = [], []
        for m, gw in enumerate(banks):
            ps = build_pool(gw, per_state=6, seed=a.seed + m)
            tagged = [(gw, s, g, d) for (s, g, d) in ps]
            if m < a.maps - n_test_maps:
                train += [p for p in tagged if p[3] <= a.D]
            else:
                test_pool += tagged                             # unseen maps
        n_maps = a.maps
    maxd = max(d for _, _, _, d in (test_pool if a.layout == "random" else train + test_pool))
    C, n, HW = (2 + 2 * a.K), (2 + a.K), (a.H * a.W)
    print(f"layout={a.layout} readout={a.readout} {a.W}x{a.H} K{a.K} maps={n_maps} "
          f"| train_pairs={len(train)} test_pairs={len(test_pool)} maxd={maxd} D={a.D} Ttest={a.Ttest}", flush=True)

    def render_batch(pairs, idxs, which):
        return torch.stack([pairs[i][0].render(pairs[i][which]) for i in idxs]).to(dev)

    def batch(pairs, bs):
        idxs = [random.randrange(len(pairs)) for _ in range(bs)]
        s = render_batch(pairs, idxs, 1); g = render_batch(pairs, idxs, 2)
        y = torch.tensor([pairs[i][3] for i in idxs], dtype=torch.float32, device=dev)
        return s, g, y

    if a.arch == "integ":
        model = GWFlow(C, n, a.d, a.heads, a.recall, a.readout, HW, a.layers).to(dev)
    else:
        model = GWHead(a.arch, C, n, a.d, a.heads, a.readout, HW, a.baselayers, a.latentnorm).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"arch={a.arch} params={nparam}", flush=True)

    @torch.no_grad()
    def evaluate(T, full=False):
        model.eval()
        by_d = {}
        for p in test_pool:
            by_d.setdefault(p[3], []).append(p)
        out = {}; P, Y = [], []
        for dd, lst in by_d.items():
            lst = lst[:400]; idxs = list(range(len(lst)))
            s = render_batch(lst, idxs, 1); g = render_batch(lst, idxs, 2)
            pred = model(s, g, T)
            out[dd] = (pred - torch.tensor([dd] * len(lst), dtype=torch.float32, device=dev)).abs().mean().item()
            P.append(pred.float().cpu()); Y.append(torch.full((len(lst),), float(dd)))
        model.train()
        if not full: return out
        P = torch.cat(P).numpy(); Y = torch.cat(Y).numpy()
        import numpy as np
        def corr(m): return round(float(np.corrcoef(P[m], Y[m])[0, 1]), 3) if m.sum() > 2 and P[m].std() > 0 else None
        wi, be = Y <= a.D, Y > a.D
        ex = dict(test_corr=corr(np.ones_like(wi, bool)), corr_within=corr(wi), corr_beyond=corr(be),
                  mae_within=round(float(abs(P[wi] - Y[wi]).mean()), 3) if wi.any() else None,
                  mae_beyond=round(float(abs(P[be] - Y[be]).mean()), 3) if be.any() else None)
        return out, ex

    nskip = 0
    for step in range(1, a.steps + 1):
        s, g, y = batch(train, a.bs)
        T_use = random.randint(a.Tmin, a.T) if a.Tmin > 0 else a.T   # anytime training
        loss = F.smooth_l1_loss(model(s, g, T_use), y)
        if not torch.isfinite(loss):                           # skip non-finite spikes; don't poison weights
            opt.zero_grad(set_to_none=True); nskip += 1; continue
        opt.zero_grad(); loss.backward()
        if a.gradclip > 0: nn.utils.clip_grad_norm_(model.parameters(), a.gradclip)
        opt.step()
        if step % a.valevery == 0 or step == a.steps:
            bd = evaluate(a.Ttest)
            inr = sum(bd[d] for d in bd if d <= a.D) / max(1, sum(1 for d in bd if d <= a.D))
            ext = {d: round(bd[d], 2) for d in sorted(bd) if d > a.D}
            print(f"step {step} loss {loss.item():.3f} inrange(d<={a.D}) {inr:.3f} extrap {ext}", flush=True)

    bd, ex = evaluate(a.Ttest, full=True)
    flat = 0
    for d in sorted(bd):
        if bd[d] < 1.5: flat = d
        else: break
    near = sum(bd[d] for d in bd if d <= a.D) / max(1, sum(1 for d in bd if d <= a.D))
    res = {"layout": a.layout, "readout": a.readout, "H": a.H, "W": a.W, "K": a.K, "maps": n_maps,
           "D": a.D, "Ttest": a.Ttest, "T": a.T, "Tmin": a.Tmin, "layers": a.layers, "guard": a.guard,
           "recall": a.recall, "seed": a.seed, "maxd": maxd,
           "arch": a.arch, "baselayers": a.baselayers, "latentnorm": a.latentnorm, "d": a.d, "heads": a.heads,
           "lr": a.lr, "steps": a.steps, "gradclip": a.gradclip, "params": nparam, "nskip": nskip, "tag": a.tag,
           "near": round(near, 3), "flat": flat, **ex, "by_d": {int(k): round(v, 3) for k, v in bd.items()}}
    print("RESULT " + json.dumps(res), flush=True)
    if a.save:
        torch.save({"state_dict": model.state_dict(), "args": vars(a), "res": res}, a.save)
        print("SAVED " + a.save)


if __name__ == "__main__":
    main()

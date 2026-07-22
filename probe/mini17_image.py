"""Image-input version: agents live on a SHARED GxG canvas (distinct markers) + a key token. The
encoder must bind each agent out of one image, then the proven recipe (multi-scale isometry + repel,
no-displacement gate, gate-guided VGT) reads the DOF ladder. Self-contained.

Env: N agents on a GxG torus (2-D). Agent 0 always free; agent j free iff key>=j. Key changes only
when the mover (agent 0) is on the control cell. DOF at a generic state = 2*(1 + #free among 1..N-1),
so key=0/1/2 -> DOF 2/4/6.

Image encoder: render state as a length-(G*G+1) token grid (cell value = agent-id+1 at each agent's
position, 0 empty; last token = key value, offset to its own range). Embed value+position, then N+1
learned component queries attend over the grid -> one vector per component (agents + key). The
gated L1 distance (no-displacement gate) runs on those component vectors, exactly as in the factored
model. So binding is the only new burden; the distance/recipe are unchanged.
"""
import argparse, itertools, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

N, G, Kk = 3, 8, 3
NP = G * G; CTRL = 0; VAL_VOCAB = (N + 1) + Kk        # 0..N grid markers, then key range offset by N+1
MARGIN = 8.0
def dof_of(key): return 2 * (1 + min(int(key), N - 1))
def set_G(g):
    global G, NP, VAL_VOCAB; G = g; NP = g * g; VAL_VOCAB = (N + 1) + Kk
def set_N(n):
    global N, Kk, VAL_VOCAB; N = n; Kk = n; VAL_VOCAB = (n + 1) + n

GRID = dict(G=[8, 12, 16], d=[64, 96], heads=[4, 8], margin=[6, 10])
def config_from_index(idx):
    keys = list(GRID); combos = list(itertools.product(*[GRID[k] for k in keys]))
    return dict(zip(keys, combos[idx % len(combos)])), len(combos)

def rand_states(n, rng):
    return np.concatenate([rng.integers(0, NP, (n, N)), rng.integers(0, Kk, (n, 1))], 1)

def step2d(p, dr, dc):
    r, c = p // G, p % G
    return ((r + dr) % G) * G + ((c + dc) % G)

DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

def move_multiscale(s, rng, K):
    out = s.copy(); dist = np.zeros(len(s), np.float32)
    for b in range(len(s)):
        key = s[b, N]
        for j in range(N):
            if j == 0 or key >= j:
                dr = int(rng.integers(-K, K + 1)); dc = int(rng.integers(-K, K + 1))
                out[b, j] = step2d(s[b, j], dr, dc); dist[b] += abs(dr) + abs(dc)
    return out, dist

def illegal_neighbour(s, rng):
    out = s.copy(); ok = np.zeros(len(s), bool)
    for b in range(len(s)):
        key = s[b, N]; opts = []
        for j in range(1, N):
            if key < j:
                dr, dc = DIRS[rng.integers(0, 4)]; opts.append(("a", j, dr, dc))
        if s[b, 0] != CTRL:
            if key > 0: opts.append(("k", -1))
            if key < Kk - 1: opts.append(("k", 1))
        if opts:
            m = opts[rng.integers(0, len(opts))]; ok[b] = True
            if m[0] == "a": out[b, m[1]] = step2d(s[b, m[1]], m[2], m[3])
            else: out[b, N] = key + m[1]
    return out, ok

def render(s, device):
    B = s.shape[0]; img = torch.zeros(B, NP + 1, dtype=torch.long, device=device); ar = torch.arange(B, device=device)
    for i in range(N): img[ar, s[:, i]] = i + 1
    img[:, NP] = (N + 1) + s[:, N]                    # key token, own value range
    return img

class ImageDist(nn.Module):
    """encoder in {mha, marker, gather}: how N+1 component vectors are read from the shared canvas.
      mha    - learned queries + MultiheadAttention (entangled; does not bind cleanly).
      marker - queries carry a per-component marker-id embedding, so query i seeks marker i.
      gather - soft-gather: component i = softmax over cells of (marker-key_i . value) applied to tokens;
               binds by construction (each component attends to the unique cell with its marker)."""
    def __init__(self, d=64, heads=4, encoder="gather", keygate=False, **kw):
        super().__init__()
        self.encoder = encoder; self.keygate = keygate
        self.val = nn.Embedding(VAL_VOCAB, d); self.pe = nn.Embedding(NP + 1, d)
        self.query = nn.Parameter(torch.randn(N + 1, d) * 0.02)
        self.mk = nn.Embedding(N + 1, d)
        if encoder in ("mha", "marker"):
            self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        if encoder in ("gather", "gather2"):
            self.vproj = nn.Linear(d, d)
        self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.wk = nn.Parameter(torch.zeros(()))
        self.d = d
    def components(self, s):
        img = render(s, s.device); B = img.shape[0]
        vale = self.val(img); tok = vale + self.pe(torch.arange(NP + 1, device=s.device))[None]   # (B, NP+1, d)
        ids = torch.arange(N + 1, device=s.device)
        if self.encoder == "mha":
            comp, _ = self.attn(self.query[None].expand(B, -1, -1), tok, tok)
        elif self.encoder == "marker":
            q = (self.query + self.mk(ids))[None].expand(B, -1, -1)
            comp, _ = self.attn(q, tok, tok)
        elif self.encoder == "hgather":  # deterministic bind: component i = token at the cell with marker i+1
            comps = []
            for i in range(N):
                m = (img == (i + 1)).float(); w = m / m.sum(1, keepdim=True).clamp(min=1e-6)   # marker-i location
                comps.append(torch.einsum('bn,bnd->bd', w, tok))
            comps.append(tok[:, NP])                                                            # key token
            comp = torch.stack(comps, 1)
        elif self.encoder == "gather2":  # agents gather over GRID cells only; key is its own token
            gv, gt = vale[:, :NP], tok[:, :NP]
            score = torch.einsum('cd,bnd->bcn', self.mk(ids[:N]), self.vproj(gv)) / self.d ** 0.5   # (B, N, NP)
            comp_ag = torch.einsum('bcn,bnd->bcd', torch.softmax(score, -1), gt)                    # (B, N, d)
            comp = torch.cat([comp_ag, tok[:, NP:NP + 1]], 1)                                       # append key token
        else:  # gather: match each component's marker-key against cell VALUES, gather tokens
            score = torch.einsum('cd,bnd->bcn', self.mk(ids), self.vproj(vale)) / self.d ** 0.5   # (B, N+1, NP+1)
            comp = torch.einsum('bcn,bnd->bcd', torch.softmax(score, -1), tok)
        return comp
    def forward(self, x, y):
        cx, cy = self.components(x), self.components(y)
        ax = [cx[:, i] for i in range(N)]; kx = cx[:, N]
        ay = [cy[:, i] for i in range(N)]; ky = cy[:, N]
        d = F.softplus(self.wk) * torch.norm(kx - ky, dim=-1)
        if self.keygate:                                   # gate sees only agent-id + key -> cannot cancel isometry
            kctx = kx + ky; B = x.shape[0]
            for i in range(N):
                aid = self.mk(torch.tensor(i, device=x.device))[None].expand(B, -1)
                w = F.softplus(self.gate(torch.cat([aid, kctx], -1))).squeeze(-1)
                d = d + w * torch.norm(ax[i] - ay[i], dim=-1)
        else:
            ctx = torch.stack([a + b for a, b in zip(ax + [kx], ay + [ky])], 0).mean(0)
            for i in range(N):
                w = F.softplus(self.gate(torch.cat([ax[i] + ay[i], ctx], -1))).squeeze(-1)
                d = d + w * torch.norm(ax[i] - ay[i], dim=-1)
        return d

def mover_move(s, rng, K):
    """move ONLY the mover (agent 0) by (dr,dc) in [-K,K]; target = |dr|+|dc|. Dedicated per-agent
    isometry so the always-free mover's own component must be multi-scale (fixes the DOF-2 collapse)."""
    out = s.copy(); dist = np.zeros(len(s), np.float32)
    for b in range(len(s)):
        dr = int(rng.integers(-K, K + 1)); dc = int(rng.integers(-K, K + 1))
        out[b, 0] = step2d(s[b, 0], dr, dc); dist[b] = abs(dr) + abs(dc)
    return out, dist

def train(head, steps, rng, device, K=3, lam_iso=3.0, margin=MARGIN, lr=3e-3, bs=256, mover_boost=False):
    head.to(device); opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(steps):
        s = rand_states(bs, rng); mv, md = move_multiscale(s, rng, K); il, ok = illegal_neighbour(s, rng); rd = rand_states(bs, rng)
        st, mt, it, rt = (torch.as_tensor(a, device=device) for a in (s, mv, il, rd))
        mdist = torch.as_tensor(md, device=device)
        loss = lam_iso * ((head(st, mt) - mdist) ** 2).mean() + F.softplus(margin - head(st, rt)).mean()
        okm = torch.as_tensor(ok, device=device)
        if okm.any(): loss = loss + F.softplus(margin - head(st, it)[okm]).mean()
        if mover_boost:                                    # extra per-agent isometry for the mover
            mm, mmd = mover_move(s, rng, K)
            loss = loss + lam_iso * ((head(st, torch.as_tensor(mm, device=device)) - torch.as_tensor(mmd, device=device)) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return head

def vgt(dist, lo=0.05, hi=0.6, mlo=6):
    d = np.sort(dist[dist > 1e-9]); Nn = d.size
    if Nn < 12: return np.nan
    a, b = max(mlo, int(lo * Nn)), int(hi * Nn)
    if b - a < 5 or d[b - 1] - d[a] < 1e-6: return np.nan
    return float(np.polyfit(np.log(d[a:b]), np.log(np.arange(1, Nn + 1, dtype=float)[a:b]), 1)[0])

@torch.no_grad()
def dist_all(head, probe, pool, device, bs=12000):
    pt = torch.as_tensor(probe[None], device=device); out = []
    for k in range(0, len(pool), bs):
        b = torch.as_tensor(pool[k:k + bs], device=device); out.append(head(pt.expand(len(b), -1), b).cpu().numpy())
    return np.concatenate(out)

@torch.no_grad()
def free_agents(head, probe, device, ratio=5.0):
    """free = agents whose 1-step distance is within `ratio` of the smallest (the always-free mover);
    robust to non-uniform free scales and inconsistent frozen far-ness (frozen is >> the mover)."""
    pt = torch.as_tensor(probe[None], device=device); steps = []
    for j in range(N):
        q = probe.copy(); q[j] = step2d(q[j], 1, 0)
        steps.append(float(head(pt, torch.as_tensor(q[None], device=device))[0]))
    thr = ratio * min(steps)
    return [j for j in range(N) if steps[j] <= thr]

@torch.no_grad()
def measure(head, probe, rng, device, W=3, M=50000, L=1500):
    free = free_agents(head, probe, device)
    if not free: return np.nan, 0
    pool = np.tile(probe, (M, 1))
    for j in free:
        r = (pool[:, j] // G + rng.integers(-W, W + 1, M)) % G; c = (pool[:, j] % G + rng.integers(-W, W + 1, M)) % G
        pool[:, j] = r * G + c
    d = dist_all(head, probe, pool, device); order = np.argsort(d)
    return vgt(np.sort(d[order[1:L + 1]])), len(free)

@torch.no_grad()
def strat_ratio(head, device, rng, npr=8):
    """min-frozen-step / max-free-step at states that have frozen agents; >1 = stratified."""
    rs = []
    for _ in range(npr):
        for m in (0, 1):
            p = np.concatenate([rng.integers(1, NP, N), [m]]); pt = torch.as_tensor(p[None], device=device)
            fr, fz = [], []
            for j in range(N):
                q = p.copy(); q[j] = step2d(q[j], 1, 0)
                dd = float(head(pt, torch.as_tensor(q[None], device=device))[0])
                (fr if (j == 0 or m >= j) else fz).append(dd)
            if fz and fr: rs.append(min(fz) / max(max(fr), 1e-6))
    return float(np.median(rs)) if rs else np.nan

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=10000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--d", type=int, default=64); ap.add_argument("--lam_iso", type=float, default=3.0)
    ap.add_argument("--margin", type=float, default=8.0); ap.add_argument("--K", type=int, default=3)
    ap.add_argument("--G", type=int, default=8); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--encoder", default="gather"); ap.add_argument("--grid_index", type=int, default=-1)
    ap.add_argument("--save", action="store_true"); ap.add_argument("--save_path", default="mini17_ckpt.pt")
    ap.add_argument("--keygate", action="store_true"); ap.add_argument("--mover_boost", action="store_true")
    ap.add_argument("--Nag", type=int, default=3); args = ap.parse_args()
    if args.grid_index >= 0:
        cfg, _ = config_from_index(args.grid_index)
        for k, v in cfg.items(): setattr(args, k, v)
    set_N(args.Nag); set_G(args.G)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
    head = train(ImageDist(d=args.d, heads=args.heads, encoder=args.encoder, keygate=args.keygate), args.steps, rng, device, K=args.K, lam_iso=args.lam_iso, margin=args.margin, mover_boost=args.mover_boost)
    if args.save: torch.save(head.state_dict(), args.save_path)

    rj = np.random.default_rng(args.seed + 1)
    strat = strat_ratio(head, device, rj)
    ladder, ns = [], []
    for m in range(N):
        probes = [np.concatenate([rj.integers(1, NP, N), [m]]) for _ in range(10)]
        vg = [measure(head, p, rj, device)[0] for p in probes]; vg = [x for x in vg if np.isfinite(x) and 0 < x < 30]
        ladder.append(float(np.mean(vg)) if vg else np.nan); ns.append(len(vg))
    lad = np.array(ladder); true = np.array([dof_of(m) for m in range(N)]); v = np.isfinite(lad)
    mae = float(np.nanmean(np.abs(lad - true)))
    corr = float(np.corrcoef(lad[v], true[v])[0, 1]) if v.sum() > 1 else np.nan
    dl = np.diff(lad[v]); mono = float((dl > 0).mean()) if dl.size else np.nan
    out = dict(encoder=args.encoder, G=args.G, d=args.d, heads=args.heads, margin=args.margin, lam_iso=args.lam_iso, steps=args.steps, seed=args.seed,
               ladder=[round(x, 2) if np.isfinite(x) else None for x in ladder], ns=ns,
               strat=round(strat, 2) if np.isfinite(strat) else None,
               mae=round(mae, 3), corr=round(corr, 3) if np.isfinite(corr) else None, mono=round(mono, 2) if np.isfinite(mono) else None)
    print("RESULT " + json.dumps(out), flush=True)

if __name__ == "__main__":
    main()

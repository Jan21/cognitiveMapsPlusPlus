"""Minimal harness to debug HOW the embedding distance should be computed.

Hypothesis (from the session): the failure is not the embedding but the DISTANCE. Keep a FAITHFUL
factored embedding (stores every agent's full position, so it is NOT stratified), and let a
knob-gated distance function do the gating instead of collapsing the encoder.

ONE agent, GxG torus, knob in {all=2D, horiz=1D(col-moves), vert=1D(row-moves)}. No flipping, so
the transition graph is a disjoint union of components with a KNOWN geodesic:
  all-block   -> one connected GxG 4-neighbour torus, Manhattan geodesic, local dim 2
  horiz-block -> G disjoint rings (one per row), local dim 1, other rows UNREACHABLE (geo = inf)
  vert-block  -> G disjoint rings (one per col), local dim 1
Only 3*G*G states, so we compute the EXACT all-pairs geodesic by BFS and train each distance head to
regress it directly. Then we read local dimension under each head's distance.

Why Euclidean must fail: the geodesic is L1 (Manhattan), and the SAME row_emb table must make a row
move cost 1 in the all-block yet be far/unreachable in the horiz-block. One Euclidean metric on one
embedding cannot satisfy both; a knob-conditioned distance can.

Heads compared (each co-trains its own faithful embedding):
  euclid    ||cat(fk,fr,fc)_x - cat(...)_y||                      (no gate, L2)     -> baseline
  gL2_learn sqrt( sum_a w_a(k) ||df_a||^2 ),  w = softplus(MLP(fk))  (gate, L2 combine)
  gL1_learn sum_a w_a(k) ||df_a||,            w = softplus(MLP(fk))  (gate, L1 combine) <- expected best
  gL1_hard  sum_a g_a(k) ||df_a||,            g hardwired 0/1        (oracle upper bound)
  attn      attention over factor-diff tokens, query from the knob   (learned combine)

Primary diagnostics per head:
  geo_RMSE      fit to the true geodesic on reachable pairs (Euclidean should be high)
  d_legal       distance horiz-probe -> its legal col-neighbour   (want ~1)
  d_illegal     distance horiz-probe -> its illegal row-neighbour (want LARGE = faithful to the
                disconnection; note: NOT 0 like the encoder-collapse quotient)
  dim_all/horiz local correlation dimension under the head's own distance (want 2 / 1)
"""
import argparse, collections, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

G = 8; NK = 3; NPOS = G * G; N = NPOS * NK
MARGIN = 12.0                                   # target for unreachable (disconnected) pairs
def axes(k):                                     # row_ok, col_ok
    return k in (0, 2), k in (0, 1)
def decode(i): return (i % NPOS) // G, i % G, i // NPOS      # r, c, k
def encode(r, c, k): return k * NPOS + (r % G) * G + (c % G)

def build_graph():
    adj = [[] for _ in range(N)]
    for i in range(N):
        r, c, k = decode(i); ro, co = axes(k)
        if ro: adj[i] += [encode(r + 1, c, k), encode(r - 1, c, k)]
        if co: adj[i] += [encode(r, c + 1, k), encode(r, c - 1, k)]
    INF = 10 ** 9
    geo = np.full((N, N), INF, dtype=np.int64)
    for s in range(N):
        dq = collections.deque([s]); geo[s, s] = 0
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if geo[s, v] == INF:
                    geo[s, v] = geo[s, u] + 1; dq.append(v)
    return adj, geo, INF

# ---------- faithful factored embedding ----------
class Emb(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.knob = nn.Embedding(NK, d); self.row = nn.Embedding(G, d); self.col = nn.Embedding(G, d)
    def factors(self, idx):
        r = (idx % NPOS) // G; c = idx % G; k = idx // NPOS
        return self.knob(k), self.row(r), self.col(c)          # each (B, d)

# ---------- distance heads ----------
class Euclid(nn.Module):
    def forward(self, fx, fy, kx):
        e = torch.cat([fx[a] - fy[a] for a in range(3)], -1)
        return torch.norm(e, dim=-1)

class GatedL(nn.Module):
    """gate weights per factor; combine='l2' -> sqrt(sum w d^2), 'l1' -> sum w d."""
    def __init__(self, d=32, combine="l1", learned=True):
        super().__init__(); self.combine = combine; self.learned = learned
        self.net = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))
    def weights(self, fx, kx):
        if self.learned:
            return F.softplus(self.net(fx[0]))                 # (B,3), gate from the knob factor
        w = torch.zeros(kx.shape[0], 3, device=kx.device)
        w[:, 0] = 1.0                                          # knob factor always counts (=0 within comp)
        for b in range(kx.shape[0]):
            ro, co = axes(int(kx[b])); w[b, 1] = float(ro); w[b, 2] = float(co)
        return w
    def forward(self, fx, fy, kx):
        w = self.weights(fx, kx)
        dnorm = torch.stack([torch.norm(fx[a] - fy[a], dim=-1) for a in range(3)], -1)  # (B,3)
        if self.combine == "l2":
            return torch.sqrt((w * dnorm ** 2).sum(-1) + 1e-9)
        return (w * dnorm).sum(-1)

class AttnDist(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.axis = nn.Embedding(3, d)
        self.q = nn.Linear(d, d); self.k = nn.Linear(2 * d, d); self.v = nn.Linear(2 * d, d)
        self.out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, d))
        self.d = d
    def forward(self, fx, fy, kx):
        B = kx.shape[0]
        diffs = torch.stack([fx[a] - fy[a] for a in range(3)], 1)               # (B,3,d)
        ax = self.axis(torch.arange(3, device=kx.device))[None].expand(B, -1, -1)
        tok = torch.cat([diffs, ax], -1)                                        # (B,3,2d)
        q = self.q(fx[0])[:, None]                                              # query from knob (B,1,d)
        a = torch.softmax((q * self.k(tok)).sum(-1) / self.d ** 0.5, -1)        # (B,3)
        z = (a[..., None] * self.v(tok)).sum(1)                                 # (B,d)
        return torch.norm(self.out(z), dim=-1)

# ---------- training ----------
def sampler(adj, geo, INF, B, rng):
    i = rng.integers(0, N, B)
    j = i.copy()
    walk = rng.random(B) < 0.6
    for b in range(B):
        if walk[b]:
            cur = int(i[b]); L = int(rng.integers(1, 5))
            for _ in range(L):
                if adj[cur]: cur = adj[cur][rng.integers(0, len(adj[cur]))]
            j[b] = cur
        else:
            j[b] = rng.integers(0, N)
    g = geo[i, j].astype(np.float64)
    reach = g < INF
    tgt = np.where(reach, g, MARGIN)
    return i, j, tgt, reach

def train(head, emb, adj, geo, INF, steps, rng, lr=3e-3):
    params = list(head.parameters()) + list(emb.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    for _ in range(steps):
        i, j, tgt, reach = sampler(adj, geo, INF, 256, rng)
        ii = torch.tensor(i); jj = torch.tensor(j)
        kx = ii // NPOS
        d = head(emb.factors(ii), emb.factors(jj), kx)
        t = torch.tensor(tgt, dtype=torch.float32); rm = torch.tensor(reach)
        loss_fit = ((d - t)[rm] ** 2).mean() if rm.any() else 0.0
        loss_far = (F.relu(MARGIN - d)[~rm] ** 2).mean() if (~rm).any() else 0.0
        loss = loss_fit + loss_far
        opt.zero_grad(); loss.backward(); opt.step()
    return head, emb

def vgt(dist, lo=0.03, hi=0.6, mlo=5):
    d = np.sort(dist[dist > 1e-9]); Nn = d.size
    if Nn < 12: return np.nan
    a, b = max(mlo, int(lo * Nn)), int(hi * Nn)
    if b - a < 5 or d[b - 1] - d[a] < 1e-6: return np.nan
    return float(np.polyfit(np.log(d[a:b]), np.log(np.arange(1, Nn + 1, dtype=float)[a:b]), 1)[0])

@torch.no_grad()
def dist_to(head, emb, probe, targets):
    B = len(targets)
    ii = torch.full((B,), probe); jj = torch.tensor(targets)
    kx = ii // NPOS
    return head(emb.factors(ii), emb.factors(jj), kx).numpy()

@torch.no_grad()
def geo_rmse(head, emb, geo, INF, rng):
    i, j, tgt, reach = sampler(build_graph()[0] if False else _ADJ, geo, INF, 4000, rng)
    d = dist_to_pairs(head, emb, i, j)
    m = reach
    return float(np.sqrt(((d[m] - tgt[m]) ** 2).mean()))

@torch.no_grad()
def dist_to_pairs(head, emb, i, j):
    ii = torch.tensor(i); jj = torch.tensor(j); kx = ii // NPOS
    return head(emb.factors(ii), emb.factors(jj), kx).numpy()

def kdim(head, emb, probe, k=40):
    d = dist_to(head, emb, probe, list(range(N)))
    kk = min(k, N)
    nn = np.partition(d, kk - 1)[:kk]
    return vgt(nn)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--G", type=int, default=8)
    args = ap.parse_args()
    global G, NPOS, N, _ADJ
    G = args.G; NPOS = G * G; N = NPOS * NK
    adj, geo, INF = build_graph(); _ADJ = adj
    p_all = encode(3, 3, 0); p_hor = encode(3, 3, 1); p_ver = encode(3, 3, 2)
    leg = encode(3, 4, 1)      # horiz legal: col move
    ill = encode(4, 3, 1)      # horiz illegal: row move (different ring -> geo = inf)
    print(f"states={N}  horiz-probe legal col-nbr geo={geo[p_hor,leg]}  illegal row-nbr geo="
          f"{'inf' if geo[p_hor,ill]>=INF else geo[p_hor,ill]}")
    print(f"{'head':<11}{'geo_RMSE':>9}{'d_legal':>9}{'d_illeg':>9}{'dim_all':>9}{'dim_hor':>9}{'dim_ver':>9}")
    heads = {
        "euclid":    lambda: Euclid(),
        "gL2_learn": lambda: GatedL(combine="l2", learned=True),
        "gL1_learn": lambda: GatedL(combine="l1", learned=True),
        "gL1_hard":  lambda: GatedL(combine="l1", learned=False),
        "attn":      lambda: AttnDist(),
    }
    rng = np.random.default_rng(args.seed)
    for name, mk in heads.items():
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        head, emb = train(mk(), Emb(), adj, geo, INF, args.steps, rng)
        rm = geo_rmse(head, emb, geo, INF, np.random.default_rng(args.seed + 1))
        dl = dist_to(head, emb, p_hor, [leg])[0]; di = dist_to(head, emb, p_hor, [ill])[0]
        da = kdim(head, emb, p_all); dh = kdim(head, emb, p_hor); dv = kdim(head, emb, p_ver)
        print(f"{name:<11}{rm:>9.2f}{dl:>9.2f}{di:>9.2f}{da:>9.2f}{dh:>9.2f}{dv:>9.2f}")

if __name__ == "__main__":
    main()

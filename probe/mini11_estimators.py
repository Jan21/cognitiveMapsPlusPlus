"""(a) 1-D agents for a measurable, pronounced ladder + investigate the DIMENSION ESTIMATOR.

Env: N agents on cycle(G) (1-D each) + a gated key. Agent 0 always free (the mover); agent j>=1 free
iff key>=j. The key changes only when the mover is on the control cell. So DOF at a generic state =
1 + key, a clean 1..N ladder (steps of 1, easier to resolve than the 2-D steps of 2 in mini10).
Gate has NO wired knowledge of which factor is the key (pooled context over all components).

Estimator investigation: correlation dimension (our VGT rank-slope) is known to UNDERESTIMATE high
intrinsic dimension with finite samples. We compare it against two sample-efficient, distance-based
estimators that work directly on the LEARNED distance (no coordinates needed):
  - MLE (Levina-Bickel, MacKay-Ghahramani averaging) from each point's k nearest distances
  - TwoNN (Facco et al.) from the ratio of 2nd/1st nearest-neighbour distances
computed on a local on-stratum sample (pairwise distances via the head).
"""
import argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from mini7_local_vgt import vgt
from mini8_bank import dist_all

N, G = 5, 24; Kk = N; CTRL = 0; MARGIN = 25.0
def dof_of(key): return 1 + min(int(key), N - 1)

def rand_states(n, rng):
    return np.concatenate([rng.integers(0, G, (n, N)), rng.integers(0, Kk, (n, 1))], 1)

def legal_neighbour(s, rng):
    out = s.copy()
    for b in range(s.shape[0]):
        key = s[b, N]; moves = []
        for j in range(N):
            if j == 0 or key >= j: moves += [("a", j, +1), ("a", j, -1)]
        if s[b, 0] == CTRL:
            if key > 0: moves.append(("k", -1))
            if key < Kk - 1: moves.append(("k", +1))
        m = moves[rng.integers(0, len(moves))]
        if m[0] == "a": out[b, m[1]] = (s[b, m[1]] + m[2]) % G
        else: out[b, N] = key + m[1]
    return out

def illegal_neighbour(s, rng):
    out = s.copy(); ok = np.zeros(s.shape[0], bool)
    for b in range(s.shape[0]):
        key = s[b, N]; opts = [("a", j, 1 if rng.random() < .5 else -1) for j in range(1, N) if key < j]
        if s[b, 0] != CTRL:
            if key > 0: opts.append(("k", -1))
            if key < Kk - 1: opts.append(("k", +1))
        if opts:
            m = opts[rng.integers(0, len(opts))]; ok[b] = True
            if m[0] == "a": out[b, m[1]] = (s[b, m[1]] + m[2]) % G
            else: out[b, N] = key + m[1]
    return out, ok

class AttnDist1D(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.pos = nn.ModuleList([nn.Embedding(G, d) for _ in range(N)])
        self.aid = nn.Embedding(N, d); self.key = nn.Embedding(Kk, d)
        self.gate = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.wk = nn.Parameter(torch.zeros(()))
    def _embs(self, s):
        ea = [self.pos[i](s[:, i]) + self.aid(torch.full_like(s[:, i], i)) for i in range(N)]
        return ea, self.key(s[:, N])
    def forward(self, x, y):
        ax, kx = self._embs(x); ay, ky = self._embs(y)
        cx, cy = ax + [kx], ay + [ky]
        g = torch.cat([torch.stack([a + b for a, b in zip(cx, cy)], 0).mean(0),
                       torch.stack([(a - b).abs() for a, b in zip(cx, cy)], 0).mean(0)], -1)
        d = F.softplus(self.wk) * torch.norm(kx - ky, dim=-1)
        for i in range(N):
            ti = torch.cat([ax[i] + ay[i], (ax[i] - ay[i]).abs()], -1)
            d = d + F.softplus(self.gate(torch.cat([ti, g], -1))).squeeze(-1) * torch.norm(ax[i] - ay[i], dim=-1)
        return d

def train(head, steps, rng, lr=3e-3, bs=192):
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(steps):
        s = rand_states(bs, rng); nb = legal_neighbour(s, rng); il, ok = illegal_neighbour(s, rng); rd = rand_states(bs, rng)
        st, nt, it, rt = (torch.tensor(a) for a in (s, nb, il, rd))
        loss = ((head(st, nt) - 1.0) ** 2).mean() + F.softplus(MARGIN - head(st, rt)).mean()
        okm = torch.tensor(ok)
        if okm.any(): loss = loss + F.softplus(MARGIN - head(st, it)[okm]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return head

# ---------- estimators (all distance-based) ----------
def mle_dim(D, k=20):
    """Levina-Bickel MLE with MacKay-Ghahramani averaging, from a pairwise distance matrix D."""
    S = []
    for i in range(D.shape[0]):
        d = np.sort(D[i]); d = d[d > 1e-9]
        if d.size <= k: continue
        dk = d[k - 1]; S.append(np.log(dk / d[:k - 1]).mean())
    return float(1.0 / np.mean(S)) if S and np.mean(S) > 0 else np.nan

def twonn_dim(D, discard=0.1):
    """TwoNN (Facco et al.) from the 2nd/1st NN ratio, using pairwise distance matrix D."""
    mus = []
    for i in range(D.shape[0]):
        d = np.sort(D[i]); d = d[d > 1e-9]
        if d.size >= 2 and d[0] > 1e-9: mus.append(d[1] / d[0])
    mus = np.sort(mus); Nn = len(mus)
    if Nn < 10: return np.nan
    keep = int(Nn * (1 - discard)); mus = mus[:keep]
    Fc = np.arange(1, keep + 1) / Nn
    x, y = np.log(mus), -np.log(1 - Fc + 1e-12)
    return float(np.sum(x * y) / np.sum(x * x))

@torch.no_grad()
def pairwise(head, pts, bs=8192):
    L = len(pts); D = np.zeros((L, L), np.float32); P = torch.tensor(pts)
    for i in range(L):
        row = []
        for k in range(0, L, bs):
            b = P[k:k + bs]; row.append(head(P[i:i + 1].repeat(len(b), 1), b).numpy())
        D[i] = np.concatenate(row)
    return D

@torch.no_grad()
def measure(head, probe, rng, W=6, M=60000, L=800):
    pool = np.tile(probe, (M, 1))
    for i in range(N): pool[:, i] = (pool[:, i] + rng.integers(-W, W + 1, M)) % G   # jitter agents, key fixed
    d = dist_all(head, probe, pool); order = np.argsort(d); ds = d[order]
    # GAP-CUT: isolate the free cluster (frozen jitters are repelled far -> a big multiplicative jump)
    dpos = ds[ds > 1e-9]; ld = np.log(dpos); gap = ld[1:] - ld[:-1]
    lo, hi = 20, int(0.9 * dpos.size)
    cand = np.where(gap[lo:hi] > 0.8)[0]
    ncut = (lo + int(cand[0]) + 1) if cand.size else dpos.size                      # free-cluster size
    free = order[1:ncut]                                                            # drop the probe itself (dist 0)
    if free.size < 12: return np.nan, np.nan, np.nan, free.size
    vg = vgt(np.sort(d[free]))                                                       # VGT on the whole free cluster
    sample = pool[free[:L]]                                                          # cap for O(L^2) pairwise
    Dm = pairwise(head, sample) + rng.normal(0, 1e-3, (len(sample), len(sample)))    # dequantize tied distances
    return vg, mle_dim(Dm), twonn_dim(Dm), free.size

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--L", type=int, default=500); args = ap.parse_args()
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
    head = train(AttnDist1D(), args.steps, rng)
    rj = np.random.default_rng(args.seed + 1)
    with torch.no_grad():                                  # did the model stratify? free ~1, frozen far?
        print("[diag] 1-step distance per agent (F=free, x=frozen), generic state, per key")
        for m in range(N):
            p = np.concatenate([np.full(N, 5, int), [m]]); pt = torch.tensor(p[None]); ds = []
            for j in range(N):
                q = p.copy(); q[j] = (q[j] + 1) % G; fr = (j == 0 or m >= j)
                ds.append(f"a{j}:{float(head(pt, torch.tensor(q[None]))[0]):.1f}{'F' if fr else 'x'}")
            print(f"  key={m} (DOF {dof_of(m)}): " + " ".join(ds), flush=True)
    print(f"\nN={N} G={G}; DOF = 1+key; local sample L={args.L}\n")
    print(f"{'':<8}{'DOF':>5}{'VGT(corr)':>11}{'MLE':>8}{'TwoNN':>8}{'free-N':>9}")
    for m in range(N):                                     # key=0..4 -> DOF 1..5
        probes = [np.concatenate([rj.integers(1, G, N), [m]]) for _ in range(6)]    # generic (agent0 not at CTRL=0)
        vg, ml, tn, nn = [], [], [], []
        for p in probes:
            a, b, c, ncf = measure(head, p, rj, L=args.L)
            vg.append(a); ml.append(b); tn.append(c); nn.append(ncf)
        f = lambda a: np.nanmean([x for x in a if np.isfinite(x) and 0 < x < 30])
        print(f"key={m}  {dof_of(m):>4}{f(vg):>11.2f}{f(ml):>8.2f}{f(tn):>8.2f}{np.mean(nn):>9.0f}")

if __name__ == "__main__":
    main()

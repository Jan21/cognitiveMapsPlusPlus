"""Gate-guided sampling on GPU + configurable lattice, to separate under-sampling from discreteness.

mini11 conflated two failures. This isolates them:
  - UNDER-SAMPLING (fixed here): use the model's own 1-step response to find which agents are free
    (free ~1, frozen ~30+; no graph, no key), and jitter ONLY the free agents -> the pool is entirely
    on the stratum. The `freeN` column confirms it reads the true number of free agents.
  - DISCRETENESS (what remains): a coarse integer lattice gives quantized distances that break the
    continuous-manifold estimators. Sweep the lattice size G (and jitter range W) to test whether a
    finer lattice recovers the DOF ladder, and compare VGT (correlation) / MLE / TwoNN.

Env / model / training come from mini11; N, G, Kk are overridden per-run. Runs on CUDA if available.
"""
import argparse, numpy as np, torch, torch.nn.functional as F
import mini11_estimators as m11
from mini11_estimators import vgt, mle_dim, twonn_dim

def move_multiscale(s, rng, K):
    """Multi-scale isometry target: displace each FREE agent by a random amount in [-K,K] (cyclic);
    target = L1 sum of |displacements| (the within-stratum geodesic). Generalises 'neighbour at 1
    step -> 1' to 'point at k steps -> k', building the intermediate scales VGT needs."""
    out = s.copy(); dist = np.zeros(len(s), dtype=np.float32)
    for b in range(len(s)):
        key = s[b, m11.N]
        for j in range(m11.N):
            if j == 0 or key >= j:
                st = int(rng.integers(-K, K + 1)); out[b, j] = (s[b, j] + st) % m11.G; dist[b] += abs(st)
    return out, dist

def train(head, steps, rng, device, K=6, lam_iso=1.0, margin=None, lr=3e-3, bs=256):
    mar = m11.MARGIN if margin is None else margin
    head.to(device); opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(steps):
        s = m11.rand_states(bs, rng); mv, mdist = move_multiscale(s, rng, K)
        il, ok = m11.illegal_neighbour(s, rng); rd = m11.rand_states(bs, rng)
        st, mt, it, rt = (torch.as_tensor(a, device=device) for a in (s, mv, il, rd))
        md = torch.as_tensor(mdist, device=device)
        loss = lam_iso * ((head(st, mt) - md) ** 2).mean() + F.softplus(mar - head(st, rt)).mean()   # isometry + repel
        okm = torch.as_tensor(ok, device=device)
        if okm.any(): loss = loss + F.softplus(mar - head(st, it)[okm]).mean()                        # frozen moves far
        opt.zero_grad(); loss.backward(); opt.step()
    return head

@torch.no_grad()
def dist_all(head, probe, pool, device, bs=100000):
    pt = torch.as_tensor(probe[None], device=device); out = []
    for k in range(0, len(pool), bs):
        b = torch.as_tensor(pool[k:k + bs], device=device)
        out.append(head(pt.expand(len(b), -1), b).cpu().numpy())
    return np.concatenate(out)

@torch.no_grad()
def pairwise(head, pts, device):
    L = len(pts); P = torch.as_tensor(pts, device=device); D = np.zeros((L, L), np.float32)
    for i in range(L):
        D[i] = head(P[i:i + 1].expand(L, -1), P).cpu().numpy()
    return D

@torch.no_grad()
def free_agents(head, probe, device, thresh=5.0):
    pt = torch.as_tensor(probe[None], device=device); steps = []
    for j in range(m11.N):
        q = probe.copy(); q[j] = (q[j] + 1) % m11.G
        steps.append(float(head(pt, torch.as_tensor(q[None], device=device))[0]))
    return [j for j in range(m11.N) if steps[j] < thresh]

@torch.no_grad()
def measure(head, probe, rng, device, W, M, L):
    free = free_agents(head, probe, device)
    if not free: return np.nan, np.nan, np.nan, 0
    pool = np.tile(probe, (M, 1))
    for j in free: pool[:, j] = (pool[:, j] + rng.integers(-W, W + 1, M)) % m11.G     # jitter ONLY free agents
    d = dist_all(head, probe, pool, device); order = np.argsort(d)
    dpos = np.sort(d[d > 1e-9])
    vg = vgt(dpos)                                                                     # VGT over the WHOLE on-stratum pool (mini7 style: full range, not a tiny near-ball)
    idx = order[1:L + 1]                                                               # nearest L only for the O(L^2) pairwise estimators
    Dm = pairwise(head, pool[idx], device) + rng.normal(0, 1e-4, (len(idx), len(idx)))
    return vg, mle_dim(Dm), twonn_dim(Dm), len(free)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=9000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--N", type=int, default=5); ap.add_argument("--G", type=int, default=48)
    ap.add_argument("--W", type=int, default=16); ap.add_argument("--M", type=int, default=150000)
    ap.add_argument("--L", type=int, default=2000); args = ap.parse_args()
    m11.N, m11.G, m11.Kk = args.N, args.G, args.N                                       # override lattice
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  N={args.N} G={args.G} W={args.W} M={args.M} L={args.L}  DOF=1+key")
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
    head = train(m11.AttnDist1D(), args.steps, rng, device)
    rj = np.random.default_rng(args.seed + 1)
    print(f"\n{'':<8}{'DOF':>5}{'freeN':>7}{'VGT':>8}{'MLE':>8}{'TwoNN':>8}")
    for m in range(args.N):
        probes = [np.concatenate([rj.integers(1, args.G, args.N), [m]]) for _ in range(6)]
        vg, ml, tn, fn = [], [], [], []
        for p in probes:
            a, b, c, f = measure(head, p, rj, device, args.W, args.M, args.L)
            vg.append(a); ml.append(b); tn.append(c); fn.append(f)
        g = lambda a: np.nanmean([x for x in a if np.isfinite(x) and 0 < x < 40]) if any(np.isfinite(x) and 0 < x < 40 for x in a) else np.nan
        print(f"key={m}  {1 + min(m, args.N - 1):>4}{np.mean(fn):>7.1f}{g(vg):>8.2f}{g(ml):>8.2f}{g(tn):>8.2f}")

if __name__ == "__main__":
    main()

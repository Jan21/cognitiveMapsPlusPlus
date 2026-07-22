"""One config of the large-scale exploration: train, then score BOTH objectives, print one JSON line.
Self-contained (numpy + torch only; imports mini7/8/11/12 which are numpy/torch) so it runs on the
cluster base python. Supports --grid_index for Slurm-array sweeps.

Objectives:
  LADDER (VGT reads DOF): ladder[dof]=mean VGT over generic probes; mae vs true, corr, monotonicity, spread.
  RINGS (isometry): mean 1-NN-adjacent fraction across agents, mean isometry ratio ed(6)/ed(1).
"""
import argparse, itertools, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import mini11_estimators as m11

# ---- grid for --grid_index (Slurm array) ----
GRID = dict(margin=[3, 5, 7, 10, 14], lam_iso=[2, 5, 10, 20], d=[48, 96], G=[48, 64])
def config_from_index(idx):
    keys = list(GRID); combos = list(itertools.product(*[GRID[k] for k in keys]))
    return dict(zip(keys, combos[idx % len(combos)])), len(combos)

# ---- no-displacement isometry gate (inlined from mini14 so no matplotlib/sklearn needed) ----
class AttnDist1D_iso(nn.Module):
    def __init__(self, d=48):
        super().__init__()
        self.pos = nn.ModuleList([nn.Embedding(m11.G, d) for _ in range(m11.N)])
        self.aid = nn.Embedding(m11.N, d); self.key = nn.Embedding(m11.Kk, d)
        self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.wk = nn.Parameter(torch.zeros(()))
    def _embs(self, s):
        ea = [self.pos[i](s[:, i]) + self.aid(torch.full_like(s[:, i], i)) for i in range(m11.N)]
        return ea, self.key(s[:, m11.N])
    def forward(self, x, y):
        ax, kx = self._embs(x); ay, ky = self._embs(y); cx, cy = ax + [kx], ay + [ky]
        ctx = torch.stack([a + b for a, b in zip(cx, cy)], 0).mean(0)
        d = F.softplus(self.wk) * torch.norm(kx - ky, dim=-1)
        for i in range(m11.N):
            w = F.softplus(self.gate(torch.cat([ax[i] + ay[i], ctx], -1))).squeeze(-1)
            d = d + w * torch.norm(ax[i] - ay[i], dim=-1)
        return d

def nn_adjacent_frac(W, G, k=1):
    D = np.linalg.norm(W[:, None] - W[None], axis=2); np.fill_diagonal(D, np.inf)
    knn = np.argsort(D, 1)[:, :k]; ok = 0
    for p in range(G):
        offs = {((q - p + G // 2) % G) - G // 2 for q in knn[p]}
        if offs <= set(range(-k, k + 1)) - {0}: ok += 1
    return ok / G

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", default="iso"); ap.add_argument("--margin", type=float, default=10.0)
    ap.add_argument("--lam_iso", type=float, default=3.0); ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--d", type=int, default=48); ap.add_argument("--G", type=int, default=48)
    ap.add_argument("--steps", type=int, default=8000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--N", type=int, default=5); ap.add_argument("--grid_index", type=int, default=-1)
    args = ap.parse_args()
    if args.grid_index >= 0:
        cfg, ncombo = config_from_index(args.grid_index)
        for k, v in cfg.items(): setattr(args, k, v)
    m11.N, m11.G, m11.Kk = args.N, args.G, args.N
    from mini12_gate_guided import train, measure
    from mini11_estimators import AttnDist1D
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
    head = (AttnDist1D_iso if args.gate == "iso" else AttnDist1D)(d=args.d)
    head = train(head, args.steps, rng, device, K=args.K, lam_iso=args.lam_iso, margin=args.margin)

    rj = np.random.default_rng(args.seed + 1); ladder = []
    for m in range(args.N):
        probes = [np.concatenate([rj.integers(1, args.G, args.N), [m]]) for _ in range(6)]
        vg = [measure(head, p, rj, device, W=min(16, args.G // 3), M=80000, L=1200)[0] for p in probes]
        vg = [x for x in vg if np.isfinite(x) and 0 < x < 40]
        ladder.append(float(np.mean(vg)) if vg else np.nan)
    ladder = np.array(ladder); true = np.arange(1, args.N + 1); v = np.isfinite(ladder)
    mae = float(np.nanmean(np.abs(ladder - true)))
    corr = float(np.corrcoef(ladder[v], true[v])[0, 1]) if v.sum() > 1 else np.nan
    dl = np.diff(ladder[v]); mono = float((dl > 0).mean()) if dl.size else np.nan
    spread = float(ladder[v][-1] - ladder[v][0]) if v.sum() > 1 else np.nan

    hc = head.cpu(); gd = np.array([min(k, args.G - k) for k in range(args.G)])
    nnadj, ratios = [], []
    for i in range(args.N):
        W = hc.pos[i].weight.detach().numpy(); nnadj.append(nn_adjacent_frac(W, args.G, 1))
        ed = np.linalg.norm(W - W[0], axis=1); ratios.append(ed[gd == 6].mean() / max(ed[gd == 1].mean(), 1e-6))
    out = dict(gate=args.gate, margin=args.margin, lam_iso=args.lam_iso, K=args.K, d=args.d, G=args.G,
               steps=args.steps, seed=args.seed,
               ladder=[round(x, 2) if np.isfinite(x) else None for x in ladder],
               mae=round(mae, 3), corr=round(corr, 3) if np.isfinite(corr) else None,
               mono=round(mono, 3) if np.isfinite(mono) else None,
               spread=round(spread, 2) if np.isfinite(spread) else None,
               nn_adj=round(float(np.mean(nnadj)), 3), iso_ratio=round(float(np.mean(ratios)), 2))
    print("RESULT " + json.dumps(out), flush=True)

if __name__ == "__main__":
    main()

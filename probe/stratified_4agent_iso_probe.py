"""
Improved 4-agent dimension capture. Three upgrades over stratified_4agent_probe.py:

  1. MULTI-SCALE ISOMETRY TRAINING. Instead of only anchoring 1-step neighbors to distance 1,
     anchor move-pairs to their EXACT geodesic distance (toroidal L1 of the displacement, which
     is the true product-space geodesic within a stratum). This flattens local patches ->
     the embedding becomes locally isometric -> VGT's r^d scaling holds -> fixes the low/mid-D
     over-read (the embedding-vs-graph gap).
  2. Bigger latent D=64 (room for the 8D+curvature manifolds).
  3. TwoNN estimator (Facco et al. 2017) alongside the count-slope VGT: uses only the ratio of
     the 2nd/1st nearest-neighbor distances -> very local, robust at small sample sizes, helps
     the undersampled high-D end.

Env: 4 agents on a GxG torus + movement knobs; local DOF 0..8. State space never enumerated.
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from sklearn.neighbors import NearestNeighbors

NAGENTS = 4
G = 40
NPOS = G * G
NKNOB = 4
KDOF = np.array([2, 1, 1, 0])
OUT = os.path.join(os.path.dirname(__file__), "..", "factored_vis")


def rand_states(n, device):
    return (torch.randint(0, NPOS, (n, NAGENTS), device=device),
            torch.randint(0, NKNOB, (n, NAGENTS), device=device))


def sample_move_pair(pos, knob, K):
    """target reached by per-agent offsets on allowed axes; dist = flat-torus L2 geodesic
    (sqrt of sum of squared per-axis arc offsets) -- matches the L2 embedding distance."""
    n, dev = pos.shape[0], pos.device
    dsq = torch.zeros(n, device=dev); tgt = pos.clone()
    for i in range(NAGENTS):
        ki = knob[:, i]
        r, c = pos[:, i] // G, pos[:, i] % G
        row_ok = ((ki == 0) | (ki == 2)).long(); col_ok = ((ki == 0) | (ki == 1)).long()
        drow = torch.randint(-K, K + 1, (n,), device=dev) * row_ok
        dcol = torch.randint(-K, K + 1, (n,), device=dev) * col_ok
        tgt[:, i] = ((r + drow) % G) * G + ((c + dcol) % G)
        dsq = dsq + (drow.float() ** 2) + (dcol.float() ** 2)      # |off|<=K<G/2 -> toroidal
    return tgt, dsq.sqrt()


def sample_knob_change(pos, knob):
    n, dev = knob.shape[0], knob.device
    ar = torch.arange(n, device=dev); a = torch.randint(0, NAGENTS, (n,), device=dev)
    knob = knob.clone()
    knob[ar, a] = (knob[ar, a] + torch.randint(1, NKNOB, (n,), device=dev)) % NKNOB
    return pos, knob


class Factored(nn.Module):
    def __init__(self, d_model=64, D=64, n_layers=2, n_heads=4):
        super().__init__()
        self.E_pos = nn.Embedding(NPOS, d_model)
        self.E_knob = nn.Embedding(NKNOB, d_model)
        self.type_emb = nn.Embedding(2 * NAGENTS, d_model)
        self.pool = nn.Parameter(torch.randn(d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=0.0,
                                         batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.proj = nn.Linear(d_model, D)

    def embed(self, pos, knob):
        toks = [self.E_pos(pos[:, i]) for i in range(NAGENTS)] + [self.E_knob(knob[:, i]) for i in range(NAGENTS)]
        tok = torch.stack(toks, dim=1)
        B = tok.shape[0]
        typ = self.type_emb(torch.arange(2 * NAGENTS, device=tok.device))
        seq = torch.cat([self.pool.expand(B, 1, -1), tok + typ], dim=1)
        return self.proj(self.encoder(seq)[:, 0])


def train(model, steps, batch, lr, device, K=2, p=2.0, rep_offset=12.0, eval_every=1000):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for s in range(steps):
        pos, knob = rand_states(batch, device)
        eu = model.embed(pos, knob)
        tgt, dist = sample_move_pair(pos, knob, K)                 # multi-scale isometry
        loss_iso = (torch.norm(eu - model.embed(tgt, knob), p=p, dim=-1) - dist).square().mean()
        _, nknob = sample_knob_change(pos, knob)                   # knob edges at distance 1
        loss_knob = (torch.norm(eu - model.embed(pos, nknob), p=p, dim=-1) - 1.0).square().mean()
        rpos, rknob = rand_states(batch, device)                   # repulsion
        loss_rep = F.softplus(rep_offset - torch.norm(eu - model.embed(rpos, rknob), p=p, dim=-1)).mean()
        loss = loss_iso + loss_knob + loss_rep
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % eval_every == 0 or s == 0:
            print(f"step {s+1}/{steps} loss {loss.item():.3f} (iso {loss_iso.item():.3f} "
                  f"knob {loss_knob.item():.3f} rep {loss_rep.item():.3f})")
    return model


# ---------- estimators ----------
def vgt_slope(dist, window=3, rmax_frac=0.55, min_count=6, num_radii=30):
    d = np.sort(dist[dist > 1e-9])
    if d.size < 12:
        return np.nan
    rmax = d[int(len(d) * rmax_frac)]
    if rmax <= d[0]:
        return np.nan
    radii = np.logspace(np.log10(d[0]), np.log10(rmax), num_radii)
    counts = np.searchsorted(d, radii, side="left").astype(float)
    valid = counts >= min_count
    if valid.sum() < window + 1:
        return np.nan
    lr, lc = np.log(radii[valid]), np.log(counts[valid])
    slopes = [np.polyfit(lr[i:i + window], lc[i:i + window], 1)[0] for i in range(len(lr) - window)]
    return float(np.median(slopes)) if slopes else np.nan


def twonn(E, discard_frac=0.1):
    """TwoNN (Facco 2017): d from the ratio mu=r2/r1 of the 2 nearest neighbors per point."""
    if len(E) < 40:
        return np.nan
    nn = NearestNeighbors(n_neighbors=3).fit(E)
    d, _ = nn.kneighbors(E)
    mu = d[:, 2] / np.maximum(d[:, 1], 1e-12)
    mu = np.sort(mu[np.isfinite(mu) & (mu > 1 + 1e-9)])
    n = len(mu)
    if n < 40:
        return np.nan
    F_emp = np.arange(1, n + 1) / (n + 1)
    keep = int(n * (1 - discard_frac))
    x = np.log(mu[:keep]); y = -np.log(1 - F_emp[:keep])
    return float(np.sum(x * y) / np.sum(x * x))                     # slope through origin


def sample_local_ball(pos, knob, R, N, rng):
    cols = []
    for i in range(NAGENTS):
        k = knob[i]
        drow = rng.integers(-R, R + 1, N) if k in (0, 2) else np.zeros(N, int)
        dcol = rng.integers(-R, R + 1, N) if k in (0, 1) else np.zeros(N, int)
        cols += [drow, dcol]
    off = np.unique(np.stack(cols, axis=1), axis=0)
    new_pos = np.zeros((off.shape[0], NAGENTS), dtype=np.int64); l1 = np.zeros(off.shape[0], dtype=np.int64)
    for i in range(NAGENTS):
        r, c = pos[i] // G, pos[i] % G
        dr, dc = off[:, 2 * i], off[:, 2 * i + 1]
        new_pos[:, i] = ((r + dr) % G) * G + ((c + dc) % G)
        l1 += np.abs(dr) + np.abs(dc)
    return new_pos, l1


def config_for_dof(dof_t, rng, tries=3000):
    for _ in range(tries):
        cand = rng.integers(0, NKNOB, NAGENTS)
        if int(KDOF[cand].sum()) == dof_t:
            return cand
    return None


@torch.no_grad()
def measure(model, device, R, N, per_dof, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for dof in range(0, 2 * NAGENTS + 1):
        for _ in range(per_dof):
            knob = config_for_dof(dof, rng)
            if knob is None:
                continue
            pos = rng.integers(0, NPOS, NAGENTS)
            ball, l1 = sample_local_ball(pos, knob, R, N, rng)
            M = ball.shape[0]
            E = model.embed(torch.tensor(ball, device=device),
                            torch.tensor(np.tile(knob, (M, 1)), device=device)).cpu().numpy()
            q = model.embed(torch.tensor(pos[None], device=device),
                            torch.tensor(knob[None], device=device)).cpu().numpy()[0]
            de = np.linalg.norm(E - q, axis=1)
            sub = E[rng.choice(M, min(4000, M), replace=False)]     # random subset for TwoNN (kNN cost)
            rows.append((dof, vgt_slope(de), twonn(sub)))
    return np.array(rows, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--R", type=int, default=14)
    ap.add_argument("--N", type=int, default=30000)
    ap.add_argument("--per_dof", type=int, default=30)
    ap.add_argument("--eval_every", type=int, default=4000)
    ap.add_argument("--json_out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(OUT, exist_ok=True)
    print(f"device={device} 4 AGENTS iso-train D=64 K={args.K} R={args.R} N={args.N} (DOF 0..8)")

    model = Factored().to(device)
    train(model, args.steps, args.batch, args.lr, device, K=args.K, eval_every=args.eval_every)

    res = measure(model, device, args.R, args.N, args.per_dof, args.seed)
    dof, vgt, tnn = res[:, 0], res[:, 1], res[:, 2]

    def ladder(x):
        m = ~np.isnan(x)
        return {int(d): round(float(x[m & (dof == d)].mean()), 2) for d in sorted(set(dof.astype(int)))
                if (m & (dof == d)).sum() >= 2}

    def corr(x):
        m = ~np.isnan(x)
        return float(spearmanr(x[m], dof[m]).statistic) if m.sum() > 3 else float("nan")

    print("\n================ 4-AGENT (iso-train) DOF 0..8 ================")
    print(f"[count-VGT ]  corr={corr(vgt):+.3f}   ladder={ladder(vgt)}")
    print(f"[TwoNN     ]  corr={corr(tnn):+.3f}   ladder={ladder(tnn)}")
    print("==============================================================")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(dict(count_vgt=dict(corr=corr(vgt), ladder=ladder(vgt)),
                           twonn=dict(corr=corr(tnn), ladder=ladder(tnn))), f, indent=2)
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()

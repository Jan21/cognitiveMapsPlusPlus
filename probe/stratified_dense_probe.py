"""
Dense-grid stratified probe: 2 agents on a 20x20 torus + movement knobs, but we NEVER
enumerate the 2.56M-state space. VGT is local, so:
  - train the (factored) embedding on sampled edges on-the-fly, and
  - to estimate dimension at a query state, sample its LOCAL movement-ball densely
    (random offsets on the knob-allowed axes), embed those, and run VGT there.

This gives Euclidean-VGT a real `count ~ r^d` scaling regime (unlike the coarse 5x5 lattice
where it failed), while keeping 2 agents and a dense grid.

State = (posA, posB, kA, kB); knob in {all=2D, horiz=1D, vert=1D, none=0D}; true local
DOF = DOF(kA)+DOF(kB) in {0..4}. Trained on adjacency only (contrastive, no actions).
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

G = 40
NPOS = G * G                      # 1600 (states = 1600^2*16 ~ 4e7, never enumerated)
NKNOB = 4
DOF = np.array([2, 1, 1, 0])      # all, horiz, vert, none
OUT = os.path.join(os.path.dirname(__file__), "..", "factored_vis")


# ---------- on-the-fly edge sampling (no enumeration) ----------
def rand_states(n, device):
    return (torch.randint(0, NPOS, (n,), device=device), torch.randint(0, NPOS, (n,), device=device),
            torch.randint(0, NKNOB, (n,), device=device), torch.randint(0, NKNOB, (n,), device=device))


def sample_neighbor(posA, posB, kA, kB):
    """one valid adjacent state per input (a legal agent move OR a knob change)."""
    n = posA.shape[0]; dev = posA.device
    gen = torch.randint(0, 10, (n,), device=dev)   # 0-3 A moves, 4-7 B moves, 8 knobA, 9 knobB
    rowA_ok = (kA == 0) | (kA == 2); colA_ok = (kA == 0) | (kA == 1)
    rowB_ok = (kB == 0) | (kB == 2); colB_ok = (kB == 0) | (kB == 1)
    valid = (gen >= 8)
    valid |= ((gen == 0) | (gen == 1)) & rowA_ok
    valid |= ((gen == 2) | (gen == 3)) & colA_ok
    valid |= ((gen == 4) | (gen == 5)) & rowB_ok
    valid |= ((gen == 6) | (gen == 7)) & colB_ok
    gen = torch.where(valid, gen, torch.full_like(gen, 8))       # fallback: knobA change (always valid)

    rA, cA = posA // G, posA % G
    rB, cB = posB // G, posB % G
    rA = torch.where(gen == 0, (rA + 1) % G, rA); rA = torch.where(gen == 1, (rA - 1) % G, rA)
    cA = torch.where(gen == 2, (cA + 1) % G, cA); cA = torch.where(gen == 3, (cA - 1) % G, cA)
    rB = torch.where(gen == 4, (rB + 1) % G, rB); rB = torch.where(gen == 5, (rB - 1) % G, rB)
    cB = torch.where(gen == 6, (cB + 1) % G, cB); cB = torch.where(gen == 7, (cB - 1) % G, cB)
    nkA = torch.where(gen == 8, (kA + torch.randint(1, NKNOB, (n,), device=dev)) % NKNOB, kA)
    nkB = torch.where(gen == 9, (kB + torch.randint(1, NKNOB, (n,), device=dev)) % NKNOB, kB)
    return rA * G + cA, rB * G + cB, nkA, nkB


# ---------- factored model (scales; no per-state table) ----------
class Factored(nn.Module):
    def __init__(self, d_model=32, D=24, n_layers=2, n_heads=4):
        super().__init__()
        self.E_pos = nn.Embedding(NPOS, d_model)
        self.E_knob = nn.Embedding(NKNOB, d_model)
        self.type_emb = nn.Embedding(4, d_model)
        self.pool = nn.Parameter(torch.randn(d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=0.0,
                                         batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.proj = nn.Linear(d_model, D)

    def embed(self, posA, posB, kA, kB):
        tok = torch.stack([self.E_pos(posA), self.E_pos(posB), self.E_knob(kA), self.E_knob(kB)], dim=1)
        B = tok.shape[0]
        typ = self.type_emb(torch.arange(4, device=tok.device))
        seq = torch.cat([self.pool.expand(B, 1, -1), tok + typ], dim=1)
        return self.proj(self.encoder(seq)[:, 0])


def train(model, steps, batch, lr, device, p=1.5, rep_offset=8.0, eval_every=1000):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for s in range(steps):
        pa, pb, ka, kb = rand_states(batch, device)
        na, nb, nka, nkb = sample_neighbor(pa, pb, ka, kb)
        eu = model.embed(pa, pb, ka, kb); ev = model.embed(na, nb, nka, nkb)
        d_nbr = torch.norm(eu - ev, p=p, dim=-1)
        loss_anchor = (d_nbr - 1.0).square().mean()
        ra, rb, rka, rkb = rand_states(batch, device)
        er = model.embed(ra, rb, rka, rkb)
        loss_rep = F.softplus(rep_offset - torch.norm(eu - er, p=p, dim=-1)).mean()
        loss = loss_anchor + loss_rep
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % eval_every == 0 or s == 0:
            print(f"step {s+1}/{steps} loss {loss.item():.3f} (anc {loss_anchor.item():.3f} rep {loss_rep.item():.3f})")
    return model


# ---------- local-ball dimension estimation ----------
def sample_local_ball(pa, pb, ka, kb, R, N, rng):
    """UNIQUE states from the query's local MOVE-ball (offsets on allowed axes). Dedup is
    essential: a 1D stratum has only ~2R+1 distinct points, so raw sampling duplicates."""
    rA, cA, rB, cB = pa // G, pa % G, pb // G, pb % G
    def offs(knob):
        drow = rng.integers(-R, R + 1, N) if knob in (0, 2) else np.zeros(N, int)   # all/vert -> row moves
        dcol = rng.integers(-R, R + 1, N) if knob in (0, 1) else np.zeros(N, int)   # all/horiz -> col moves
        return drow, dcol
    dRA, dCA = offs(ka); dRB, dCB = offs(kb)
    off = np.unique(np.stack([dRA, dCA, dRB, dCB], axis=1), axis=0)      # dedup
    dRA, dCA, dRB, dCB = off[:, 0], off[:, 1], off[:, 2], off[:, 3]
    posA = ((rA + dRA) % G) * G + ((cA + dCA) % G)
    posB = ((rB + dRB) % G) * G + ((cB + dCB) % G)
    l1 = np.abs(dRA) + np.abs(dCA) + np.abs(dRB) + np.abs(dCB)
    return posA.astype(np.int64), posB.astype(np.int64), l1


def vgt_slope(dist, window=3, rmax_frac=0.55, min_count=5, num_radii=30):
    """robust local slope of log(count-in-ball) vs log(radius); small window so it works on
    the few points of a low-dimensional stratum."""
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


@torch.no_grad()
def measure(model, device, R, N, per_config, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for ka in range(NKNOB):
        for kb in range(NKNOB):
            dof = DOF[ka] + DOF[kb]
            for _ in range(per_config):
                pa = int(rng.integers(0, NPOS)); pb = int(rng.integers(0, NPOS))
                posA, posB, l1 = sample_local_ball(pa, pb, ka, kb, R, N, rng)
                nb = len(posA)
                E = model.embed(torch.tensor(posA, device=device), torch.tensor(posB, device=device),
                                torch.full((nb,), ka, device=device), torch.full((nb,), kb, device=device)).cpu().numpy()
                q = model.embed(torch.tensor([pa], device=device), torch.tensor([pb], device=device),
                                torch.tensor([ka], device=device), torch.tensor([kb], device=device)).cpu().numpy()[0]
                d_emb = np.linalg.norm(E - q, axis=1)
                vgt_emb = vgt_slope(d_emb)
                # graph-VGT reference: count within L1 move-radius r (model-free)
                ml1 = l1[l1 > 0]
                vgt_graph = np.nan
                if ml1.size > 8:
                    rr = np.arange(1, R + 1)
                    C = np.array([(l1 <= r).sum() for r in rr], float)
                    m = C > 5
                    vgt_graph = float(np.polyfit(np.log(rr[m]), np.log(C[m]), 1)[0]) if m.sum() >= 2 else np.nan
                rows.append((dof, vgt_emb, vgt_graph))
    return np.array(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--R", type=int, default=6)
    ap.add_argument("--N", type=int, default=6000)
    ap.add_argument("--per_config", type=int, default=12)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--json_out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(OUT, exist_ok=True)
    print(f"device={device} DENSE grid G={G} (2 agents), states={NPOS*NPOS*16} (not enumerated), "
          f"R={args.R} N_local={args.N}")

    model = Factored().to(device)
    train(model, args.steps, args.batch, args.lr, device, eval_every=args.eval_every)

    res = measure(model, device, args.R, args.N, args.per_config, args.seed)
    dof = res[:, 0]; vgt_emb = res[:, 1]; vgt_graph = res[:, 2]

    def ladder(x):
        m = ~np.isnan(x)
        return {int(d): round(float(x[m & (dof == d)].mean()), 2) for d in sorted(set(dof.astype(int)))
                if (m & (dof == d)).any()}

    def corr(x):
        m = ~np.isnan(x)
        return float(spearmanr(x[m], dof[m]).statistic) if m.sum() > 3 else float("nan")

    print("\n================ DENSE STRATIFICATION RESULTS ================")
    print(f"(true DOF target = 0/1/2/3/4)")
    print(f"[VGT on embedding local-ball]  corr={corr(vgt_emb):+.3f}   ladder={ladder(vgt_emb)}")
    print(f"[graph-VGT (L1 move-ball ref)]  corr={corr(vgt_graph):+.3f}   ladder={ladder(vgt_graph)}")
    print("==============================================================")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(dict(vgt_emb=dict(corr=corr(vgt_emb), ladder=ladder(vgt_emb)),
                           vgt_graph=dict(corr=corr(vgt_graph), ladder=ladder(vgt_graph))), f, indent=2)
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()

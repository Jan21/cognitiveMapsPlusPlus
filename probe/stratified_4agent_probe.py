"""
4-agent stratified probe: local dimension 0..8, estimated from embeddings via local-ball VGT.

NAGENTS agents on a GxG torus, each with a movement knob {all=2D, horiz=1D, vert=1D,
none=0D}. State = (positions[NAGENTS], knobs[NAGENTS]); true local DOF = sum of the agents'
knob-DOF, in {0..2*NAGENTS}. With 4 agents that's 0..8.

State space (~G^8 * 4^4) is astronomically large and NEVER enumerated: train the factored
embedding on sampled edges on-the-fly, and estimate dimension per query from its locally
sampled (deduped) move-ball. Grid/N chosen to capture dimension across 0..8 as well as
feasible (VGT curse-of-dimensionality: crisp for low-mid D, compresses for high D).
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

NAGENTS = 4
G = 40
NPOS = G * G
NKNOB = 4
KDOF = np.array([2, 1, 1, 0])          # all, horiz, vert, none
OUT = os.path.join(os.path.dirname(__file__), "..", "factored_vis")


# ---------- on-the-fly edges (no enumeration) ----------
def rand_states(n, device):
    return (torch.randint(0, NPOS, (n, NAGENTS), device=device),
            torch.randint(0, NKNOB, (n, NAGENTS), device=device))


def sample_neighbor(pos, knob):
    """one valid adjacent state: move a random agent (per its knob) or change a random knob."""
    n, dev = pos.shape[0], pos.device
    ar = torch.arange(n, device=dev)
    a = torch.randint(0, NAGENTS, (n,), device=dev)                 # which agent
    act = torch.randint(0, 3, (n,), device=dev)                     # 0 row-move, 1 col-move, 2 knob
    ka = knob[ar, a]
    row_ok = (ka == 0) | (ka == 2); col_ok = (ka == 0) | (ka == 1)
    valid = (act == 2) | ((act == 0) & row_ok) | ((act == 1) & col_ok)
    act = torch.where(valid, act, torch.full_like(act, 2))          # fallback: knob change
    pa = pos[ar, a]; r, c = pa // G, pa % G
    sign = torch.randint(0, 2, (n,), device=dev) * 2 - 1
    row_new = ((r + sign) % G) * G + c
    col_new = r * G + (c + sign) % G
    new_pos_a = torch.where(act == 0, row_new, torch.where(act == 1, col_new, pa))
    new_knob_a = torch.where(act == 2, (ka + torch.randint(1, NKNOB, (n,), device=dev)) % NKNOB, ka)
    pos = pos.clone(); knob = knob.clone()
    pos[ar, a] = new_pos_a; knob[ar, a] = new_knob_a
    return pos, knob


class Factored(nn.Module):
    def __init__(self, d_model=48, D=32, n_layers=2, n_heads=4):
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
        tok = torch.stack(toks, dim=1)                              # (B, 2*NAGENTS, d)
        B = tok.shape[0]
        typ = self.type_emb(torch.arange(2 * NAGENTS, device=tok.device))
        seq = torch.cat([self.pool.expand(B, 1, -1), tok + typ], dim=1)
        return self.proj(self.encoder(seq)[:, 0])


def train(model, steps, batch, lr, device, p=1.5, rep_offset=8.0, eval_every=1000):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for s in range(steps):
        pos, knob = rand_states(batch, device)
        npos, nknob = sample_neighbor(pos, knob)
        eu = model.embed(pos, knob); ev = model.embed(npos, nknob)
        loss_anc = (torch.norm(eu - ev, p=p, dim=-1) - 1.0).square().mean()
        rpos, rknob = rand_states(batch, device)
        er = model.embed(rpos, rknob)
        loss_rep = F.softplus(rep_offset - torch.norm(eu - er, p=p, dim=-1)).mean()
        loss = loss_anc + loss_rep
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % eval_every == 0 or s == 0:
            print(f"step {s+1}/{steps} loss {loss.item():.3f} (anc {loss_anc.item():.3f} rep {loss_rep.item():.3f})")
    return model


# ---------- local-ball dimension ----------
def sample_local_ball(pos, knob, R, N, rng):
    """UNIQUE states from the query's local move-ball. pos,knob: length-NAGENTS arrays."""
    cols = []; l1 = np.zeros(N, dtype=np.int64)
    for i in range(NAGENTS):
        k = knob[i]
        drow = rng.integers(-R, R + 1, N) if k in (0, 2) else np.zeros(N, int)
        dcol = rng.integers(-R, R + 1, N) if k in (0, 1) else np.zeros(N, int)
        cols.append(drow); cols.append(dcol)
    off = np.unique(np.stack(cols, axis=1), axis=0)                 # dedup (M, 2*NAGENTS)
    new_pos = np.zeros((off.shape[0], NAGENTS), dtype=np.int64)
    l1 = np.zeros(off.shape[0], dtype=np.int64)
    for i in range(NAGENTS):
        r, c = pos[i] // G, pos[i] % G
        dr, dc = off[:, 2 * i], off[:, 2 * i + 1]
        new_pos[:, i] = ((r + dr) % G) * G + ((c + dc) % G)
        l1 += np.abs(dr) + np.abs(dc)
    return new_pos, l1


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


def config_for_dof(dof_t, rng, tries=3000):
    for _ in range(tries):
        cand = rng.integers(0, NKNOB, NAGENTS)
        if int(KDOF[cand].sum()) == dof_t:
            return cand
    return None


@torch.no_grad()
def measure(model, device, R, N, per_dof, seed):
    """stratified by DOF: guarantee coverage of every level 0..2*NAGENTS (incl. rare high ones)."""
    rng = np.random.default_rng(seed)
    rows = []
    queries = [(config_for_dof(d, rng), d) for d in range(0, 2 * NAGENTS + 1) for _ in range(per_dof)]
    for knob, dof in queries:
        if knob is None:
            continue
        pos = rng.integers(0, NPOS, NAGENTS)
        ball_pos, l1 = sample_local_ball(pos, knob, R, N, rng)
        M = ball_pos.shape[0]
        kt = torch.tensor(np.tile(knob, (M, 1)), device=device)
        E = model.embed(torch.tensor(ball_pos, device=device), kt).cpu().numpy()
        q = model.embed(torch.tensor(pos[None], device=device),
                        torch.tensor(knob[None], device=device)).cpu().numpy()[0]
        vgt_emb = vgt_slope(np.linalg.norm(E - q, axis=1))
        ml1 = l1[l1 > 0]
        vgt_graph = np.nan
        if ml1.size > 8:
            rr = np.arange(1, R + 1)
            C = np.array([(l1 <= r).sum() for r in rr], float)
            m = C > 5
            vgt_graph = float(np.polyfit(np.log(rr[m]), np.log(C[m]), 1)[0]) if m.sum() >= 2 else np.nan
        rows.append((dof, vgt_emb, vgt_graph))
    return np.array(rows, dtype=float)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--R", type=int, default=10)
    ap.add_argument("--N", type=int, default=25000)
    ap.add_argument("--per_dof", type=int, default=30)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--json_out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(OUT, exist_ok=True)
    print(f"device={device} {NAGENTS} AGENTS G={G} (DOF 0..{2*NAGENTS}), R={args.R} N={args.N} "
          f"per_dof={args.per_dof}  (states ~ {NPOS}^{NAGENTS} * {NKNOB}^{NAGENTS}, not enumerated)")

    model = Factored().to(device)
    train(model, args.steps, args.batch, args.lr, device, eval_every=args.eval_every)

    res = measure(model, device, args.R, args.N, args.per_dof, args.seed)
    dof, vgt_emb, vgt_graph = res[:, 0], res[:, 1], res[:, 2]

    def ladder(x):
        m = ~np.isnan(x)
        return {int(d): round(float(x[m & (dof == d)].mean()), 2) for d in sorted(set(dof.astype(int)))
                if (m & (dof == d)).sum() >= 2}

    def corr(x):
        m = ~np.isnan(x)
        return float(spearmanr(x[m], dof[m]).statistic) if m.sum() > 3 else float("nan")

    print("\n================ 4-AGENT STRATIFICATION (DOF 0..8) ================")
    print(f"[VGT on embedding local-ball]  corr={corr(vgt_emb):+.3f}   ladder={ladder(vgt_emb)}")
    print(f"[graph-VGT (L1 move-ball ref)]  corr={corr(vgt_graph):+.3f}   ladder={ladder(vgt_graph)}")
    print("===================================================================")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(dict(vgt_emb=dict(corr=corr(vgt_emb), ladder=ladder(vgt_emb)),
                           vgt_graph=dict(corr=corr(vgt_graph), ladder=ladder(vgt_graph))), f, indent=2)
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()

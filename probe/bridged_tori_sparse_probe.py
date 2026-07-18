"""
Sparse-coding disentanglement probe (brain-style population code).

Instead of forcing K=2/3 query tokens to split (which kept collapsing), use MANY latent
units (D=64) + an L1 sparsity penalty so few units are active per state. MDL argument:
with a cost per active unit, the cheapest code for independent factors is the factored
one, so units self-organize into factor-selective groups (place-cell-like for position,
a few units for id / color).

Environment: the 3-factor image env from bridged_tori_3factor_probe (position/id/color,
900 states, 7 actions).

Disentanglement is measured per-unit: eta-squared of each unit's activation explained by
position vs id vs color. A clean sparse code has each active unit dominated by ONE factor
and the three factors occupying disjoint unit sets.
"""

import argparse
import json
import os
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

from bridged_tori_3factor_probe import (
    SIDE, N_POS, N_STATES, N_ACTIONS, ID_MARKERS, COLOR_MARKERS,
    gid3, build_graph, build_transitions, all_pairs_geodesic,
)
from bridged_tori_image_probe import OUT


def render(ids, device):
    color = ids // (N_POS * 2); rem = ids % (N_POS * 2); idx = rem // N_POS; pos = rem % N_POS
    B = ids.shape[0]
    grid = torch.zeros(B, N_POS, dtype=torch.long, device=device)
    for m in ID_MARKERS:
        grid[idx == 1, m] = 2
    for m in COLOR_MARKERS:
        grid[color == 1, m] = 3
    grid[torch.arange(B, device=device), pos] = 1
    return grid


class SparseModel(nn.Module):
    def __init__(self, D=64, d_model=32, n_slots=4, n_heads=4, p=1.5):
        super().__init__()
        self.d_model = d_model
        self.D = D
        self.p = p
        self.pixel_val_emb = nn.Embedding(4, d_model)
        self.pixel_pos_emb = nn.Embedding(N_POS, d_model)
        self.slots = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.to_z = nn.Sequential(nn.Linear(n_slots * d_model, 128), nn.GELU(), nn.Linear(128, D))
        self.act_emb = nn.Embedding(N_ACTIONS, d_model)
        self.dyn = nn.Sequential(nn.Linear(D + d_model, 128), nn.GELU(), nn.Linear(128, D))

    def encode(self, ids):
        grid = render(ids, ids.device)
        B = grid.shape[0]
        pix = self.pixel_val_emb(grid) + self.pixel_pos_emb(torch.arange(N_POS, device=grid.device)).unsqueeze(0)
        Q = self.slots.unsqueeze(0).expand(B, -1, -1)
        z, _ = self.attn(Q, pix, pix)
        return F.relu(self.to_z(z.reshape(B, -1)))            # (B,D) nonneg sparse code

    def distance(self, za, zb):
        return torch.norm(za - zb, p=self.p, dim=-1)

    def dynamics(self, za, action):
        return F.relu(za + self.dyn(torch.cat([za, self.act_emb(action)], dim=-1)))


FACTOR_NAMES = ["position", "id", "color"]


@torch.no_grad()
def analyze(model, device, save_png=None):
    """Per-neuron modulation depth for each factor (comparable scale: range of
    group-means / unit std). Assign each active neuron to its dominant factor ->
    the position/id/color neuron GROUPS. Also flag mixed neurons (low purity)."""
    ids = torch.arange(N_STATES, device=device)
    Z = model.encode(ids).cpu().numpy()                       # (900,D)
    color = np.arange(N_STATES) // (N_POS * 2)
    rem = np.arange(N_STATES) % (N_POS * 2)
    idx = rem // N_POS; pos = rem % N_POS
    std = Z.std(0) + 1e-6
    m_id = np.abs(Z[idx == 0].mean(0) - Z[idx == 1].mean(0)) / std
    m_col = np.abs(Z[color == 0].mean(0) - Z[color == 1].mean(0)) / std
    onehot = np.zeros((N_STATES, N_POS)); onehot[np.arange(N_STATES), pos] = 1
    posmeans = (onehot.T @ Z) / onehot.sum(0)[:, None]        # (225,D)
    m_pos = (posmeans.max(0) - posmeans.min(0)) / std
    S = np.stack([m_pos, m_id, m_col], axis=1)                # (D,3) modulation depth
    activity = (Z > 1e-3).mean(0)
    active = activity > 0.02
    mean_L0 = float((Z > 1e-3).sum(1).mean())
    dom = S.argmax(1)
    purity = S.max(1) / (S.sum(1) + 1e-9)
    groups = {nm: [int(u) for u in np.where(active & (dom == i))[0]] for i, nm in enumerate(FACTOR_NAMES)}
    mixed = int((active & (purity < 0.6)).sum())

    if save_png is not None:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        au = np.where(active)[0]
        order = au[np.argsort(dom[au] * 1e6 - S[au].max(1))]  # group by factor, strongest first
        Sa = S[order]
        fig, ax = plt.subplots(figsize=(3.2, max(3, len(order) * 0.28)))
        im = ax.imshow(Sa, aspect="auto", cmap="magma")
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(FACTOR_NAMES, rotation=30)
        ax.set_yticks(range(len(order))); ax.set_yticklabels([f"n{u}" for u in order], fontsize=7)
        ax.set_title("neuron factor-selectivity\n(modulation depth)", fontsize=9)
        plt.colorbar(im, fraction=0.06)
        fig.tight_layout(); fig.savefig(save_png, dpi=150, bbox_inches="tight"); plt.close(fig)

    return dict(mean_L0=mean_L0, n_active=int(active.sum()), groups=groups,
                mean_purity=float(purity[active].mean()) if active.any() else 0.0, mixed=mixed)


def train(model, trans, geo, steps, batch, lr, device, l1=0.0, rep_offset=10.0, eval_every=1500):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    A, ACT, B = trans[:, 0].to(device), trans[:, 1].to(device), trans[:, 2].to(device)
    is_move = (A != B)
    n = trans.shape[0]
    rng = np.random.default_rng(123)
    ea = rng.integers(0, N_STATES, 3000); eb = rng.integers(0, N_STATES, 3000)
    model.train()
    for s in range(steps):
        idx = torch.randint(0, n, (batch,), device=device)
        a_id, act, b_id = A[idx], ACT[idx], B[idx]
        za = model.encode(a_id); zb = model.encode(b_id)
        r_id = torch.randint(0, N_STATES, (batch,), device=device)
        zr = model.encode(r_id)
        loss_dyn = model.distance(model.dynamics(za, act), zb).square().mean()
        mv = is_move[idx]
        loss_anc = ((model.distance(za[mv], zb[mv]) - 1.0).square().mean()
                    if mv.any() else torch.zeros((), device=device))
        loss_rep = F.softplus(rep_offset - model.distance(za, zr)).mean()
        loss = loss_dyn + loss_anc + loss_rep + l1 * za.abs().mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % eval_every == 0 or s == 0:
            with torch.no_grad():
                d = model.distance(model.encode(torch.tensor(ea, device=device)),
                                   model.encode(torch.tensor(eb, device=device))).cpu().numpy()
            sp = spearmanr(d, geo[ea, eb]).statistic
            l0 = (za > 1e-3).float().sum(1).mean().item()
            print(f"[l1={l1}] step {s+1}/{steps} loss {loss.item():.3f} | spearman {sp:.3f} L0 {l0:.1f}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--D", type=int, default=64)
    ap.add_argument("--l1", type=float, default=0.05)
    ap.add_argument("--eval_every", type=int, default=1500)
    ap.add_argument("--json_out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    print(f"device={device} SPARSE code D={args.D} l1={args.l1} (3-factor env), seed={args.seed}")

    G = build_graph()
    trans = build_transitions()
    geo = all_pairs_geodesic(G)

    model = SparseModel(D=args.D).to(device)
    train(model, trans, geo, args.steps, args.batch, args.lr, device, l1=args.l1,
          eval_every=args.eval_every)

    rng = np.random.default_rng(7)
    a = rng.integers(0, N_STATES, 8000); b = rng.integers(0, N_STATES, 8000)
    m = a != b; a, b = a[m], b[m]
    with torch.no_grad():
        d = model.distance(model.encode(torch.tensor(a, device=device)),
                           model.encode(torch.tensor(b, device=device))).cpu().numpy()
    sp = spearmanr(d, geo[a, b]).statistic
    png = os.path.join(OUT, f"sparse_neuron_groups_l1{args.l1}.png")
    st = analyze(model, device, save_png=png)
    g = st['groups']

    print("\n================ SPARSE RESULTS ================")
    print(f"Spearman(D, 900-state geodesic) = {sp:.3f}")
    print(f"avg active units / state (L0) = {st['mean_L0']:.1f} of {args.D}")
    print(f"active units = {st['n_active']}   (mixed/low-purity = {st['mixed']})")
    print(f"mean factor-purity of active units = {st['mean_purity']:.3f}")
    for nm in FACTOR_NAMES:
        print(f"  {nm:9s} neurons ({len(g[nm])}): {g[nm]}")
    print(f"wrote {os.path.relpath(png)}")
    print("================================================")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(dict(sp=float(sp), mean_L0=st['mean_L0'], n_active=st['n_active'],
                           mixed=st['mixed'], mean_purity=st['mean_purity'],
                           group_sizes={k: len(v) for k, v in g.items()}), f, indent=2)
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()

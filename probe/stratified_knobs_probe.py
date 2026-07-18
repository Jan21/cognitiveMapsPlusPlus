"""
Stratified-space probe: does the embedding organize into strata of DIFFERENT local
(ambient) dimension, matching each state's degrees of freedom?

Environment (tractable version of the 20x20 / 4-agent idea):
  2 agents on a GxG wraparound torus, each with a knob in {all=2D, horiz=1D, vert=1D,
  none=0D} that restricts its movement. State = (posA, posB, knobA, knobB).
  G=5 -> 25*25*4*4 = 10000 states. Local dimension of a state = DOF(knobA)+DOF(knobB),
  ranging 0..4 -> a genuinely STRATIFIED space.

We IGNORE actions: train only on the adjacency structure (contrastive: neighbors ~1 apart,
random pairs pushed apart). Edges = legal one-step agent moves (per each agent's knob) +
knob changes (made always-available to keep the graph connected).

Then we measure, per state, the PARTICIPATION-RATIO dimension of its MOVE-neighbors in the
embedding, and check whether it tracks the true DOF -> stratification.

Variants (--variant): plain (free embedding table) | factored (per-agent/knob tokens) |
image (pixel grid + attention). Run all and compare.
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

G = 5
NPOS = G * G
NKNOB = 4
NSTATES = NPOS * NPOS * NKNOB * NKNOB          # 10000
DOF = np.array([2, 1, 1, 0])                    # all, horiz, vert, none
# allowed move directions per knob: dirs as (dr,dc)
DIRS = {0: [(-1, 0), (1, 0), (0, -1), (0, 1)], 1: [(0, -1), (0, 1)], 2: [(-1, 0), (1, 0)], 3: []}
KNOB_A_CELL, KNOB_B_CELL = NPOS, NPOS + 1       # extra pixels for the image variant
OUT = os.path.join(os.path.dirname(__file__), "..", "factored_vis")


def sid(pa, pb, ka, kb):
    return ((pa * NPOS + pb) * NKNOB + ka) * NKNOB + kb


def decode(ids):
    kb = ids % NKNOB; t = ids // NKNOB
    ka = t % NKNOB; t = t // NKNOB
    pb = t % NPOS; pa = t // NPOS
    return pa, pb, ka, kb


def move(pos, dr, dc):
    r, c = divmod(pos, G)
    return ((r + dr) % G) * G + ((c + dc) % G)


def build_env():
    """returns move_edges, knob_edges (each (E,2) int64), move_nbrs (list per state), dof."""
    move_edges, knob_edges = [], []
    move_nbrs = [[] for _ in range(NSTATES)]
    dof = np.zeros(NSTATES, dtype=np.float32)
    for s in range(NSTATES):
        pa, pb, ka, kb = decode(s)
        dof[s] = DOF[ka] + DOF[kb]
        for dr, dc in DIRS[ka]:
            n = sid(move(pa, dr, dc), pb, ka, kb)
            move_edges.append((s, n)); move_nbrs[s].append(n)
        for dr, dc in DIRS[kb]:
            n = sid(pa, move(pb, dr, dc), ka, kb)
            move_edges.append((s, n)); move_nbrs[s].append(n)
        for ka2 in range(NKNOB):
            if ka2 != ka:
                knob_edges.append((s, sid(pa, pb, ka2, kb)))
        for kb2 in range(NKNOB):
            if kb2 != kb:
                knob_edges.append((s, sid(pa, pb, ka, kb2)))
    return (torch.tensor(move_edges), torch.tensor(knob_edges),
            [np.array(x, dtype=np.int64) for x in move_nbrs], dof)


# ---------- models (each exposes embed_ids(ids) -> (B,D)) ----------
class PlainEmbed(nn.Module):
    def __init__(self, D=24):
        super().__init__()
        self.emb = nn.Embedding(NSTATES, D)

    def embed_ids(self, ids):
        return self.emb(ids)


class SelfNormHead(nn.Module):
    """transformer over K tokens + POOL -> embedding e."""
    def __init__(self, n_tokens, d_model=32, D=24, n_layers=2, n_heads=4):
        super().__init__()
        self.type_emb = nn.Embedding(n_tokens, d_model)
        self.pool = nn.Parameter(torch.randn(d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=0.0,
                                         batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.proj = nn.Linear(d_model, D)

    def forward(self, tok):
        B = tok.shape[0]
        typ = self.type_emb(torch.arange(tok.shape[1], device=tok.device))
        seq = torch.cat([self.pool.expand(B, 1, -1), tok + typ], dim=1)
        return self.proj(self.encoder(seq)[:, 0])


class Factored(nn.Module):
    def __init__(self, d_model=32, D=24):
        super().__init__()
        self.E_pos = nn.Embedding(NPOS, d_model)       # shared position table (both agents)
        self.E_knob = nn.Embedding(NKNOB, d_model)
        self.head = SelfNormHead(4, d_model, D)

    def tokens(self, ids):
        pa, pb, ka, kb = decode(ids)
        return torch.stack([self.E_pos(pa), self.E_pos(pb), self.E_knob(ka), self.E_knob(kb)], dim=1)

    def embed_ids(self, ids):
        return self.head(self.tokens(ids))


class ImageEnc(nn.Module):
    def __init__(self, d_model=32, D=24, n_slots=4, n_heads=4):
        super().__init__()
        self.P = NPOS + 2
        self.val_emb = nn.Embedding(5, d_model)        # 0 bg,1 agentA,2 agentB, knob vals reuse 0-3
        self.pos_emb = nn.Embedding(self.P, d_model)
        self.slots = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(n_slots * d_model, 128), nn.GELU(), nn.Linear(128, D))

    def render(self, ids):
        pa, pb, ka, kb = decode(ids)
        B = ids.shape[0]
        grid = torch.zeros(B, self.P, dtype=torch.long, device=ids.device)
        ar = torch.arange(B, device=ids.device)
        grid[ar, pa] = 1
        grid[ar, pb] = 2                                # agentB overwrites on collision
        grid[:, KNOB_A_CELL] = ka
        grid[:, KNOB_B_CELL] = kb
        return grid

    def embed_ids(self, ids):
        grid = self.render(ids)
        B = grid.shape[0]
        pix = self.val_emb(grid) + self.pos_emb(torch.arange(self.P, device=ids.device)).unsqueeze(0)
        z, _ = self.attn(self.slots.unsqueeze(0).expand(B, -1, -1), pix, pix)
        return self.proj(z.reshape(B, -1))


def make_model(variant):
    return {"plain": PlainEmbed, "factored": Factored, "image": ImageEnc}[variant]()


# ---------- training ----------
def embed_all(model, device, chunk=4096):
    outs = []
    for i in range(0, NSTATES, chunk):
        ids = torch.arange(i, min(i + chunk, NSTATES), device=device)
        outs.append(model.embed_ids(ids))
    return torch.cat(outs)


def train(model, edges, steps, batch, lr, device, p=1.5, rep_offset=8.0, eval_every=1000, dof=None, move_nbrs=None):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    E = edges.to(device)
    ne = E.shape[0]
    model.train()
    for s in range(steps):
        idx = torch.randint(0, ne, (batch,), device=device)
        u, v = E[idx, 0], E[idx, 1]
        eu = model.embed_ids(u); ev = model.embed_ids(v)
        d_nbr = torch.norm(eu - ev, p=p, dim=-1)
        loss_anchor = (d_nbr - 1.0).square().mean()
        r = torch.randint(0, NSTATES, (batch,), device=device)
        er = model.embed_ids(r)
        d_rand = torch.norm(eu - er, p=p, dim=-1)
        loss_rep = F.softplus(rep_offset - d_rand).mean()
        loss = loss_anchor + loss_rep
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % eval_every == 0 or s == 0:
            sp = strat_corr(model, device, dof, move_nbrs, n=1500)
            print(f"step {s+1}/{steps} loss {loss.item():.3f} (anc {loss_anchor.item():.3f} "
                  f"rep {loss_rep.item():.3f}) | strat corr(localdim, DOF) = {sp:.3f}")
    return model


# ---------- stratification measurement ----------
@torch.no_grad()
def participation_dim(diffs):
    """PR dimension of a set of difference vectors (rows)."""
    if diffs.shape[0] == 0:
        return 0.0
    C = diffs.T @ diffs
    ev = np.linalg.eigvalsh(C)
    ev = np.clip(ev, 0, None)
    s1 = ev.sum()
    if s1 < 1e-9:
        return 0.0
    return float((s1 ** 2) / (ev ** 2).sum())


@torch.no_grad()
def local_dims(model, device, ids, move_nbrs):
    """per-state participation-ratio dimension from MOVE-neighbors' embeddings."""
    E = embed_all(model, device).cpu().numpy()
    out = np.zeros(len(ids))
    for i, s in enumerate(ids):
        nb = move_nbrs[s]
        out[i] = participation_dim(E[nb] - E[s]) if len(nb) else 0.0
    return out, E


@torch.no_grad()
def strat_corr(model, device, dof, move_nbrs, n=1500):
    if dof is None:
        return float("nan")
    rng = np.random.default_rng(0)
    ids = rng.integers(0, NSTATES, n)
    ld, _ = local_dims(model, device, ids, move_nbrs)
    return spearmanr(ld, dof[ids]).statistic


def visualize(E, dof, ld, ids, tag):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import umap
    Es = E[ids]
    p2 = PCA(2).fit_transform(Es)
    u2 = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=1).fit_transform(Es)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    s0 = ax[0].scatter(p2[:, 0], p2[:, 1], c=dof[ids], cmap="viridis", s=8); ax[0].set_title("PCA | true DOF")
    plt.colorbar(s0, ax=ax[0], fraction=0.046)
    s1 = ax[1].scatter(p2[:, 0], p2[:, 1], c=ld, cmap="plasma", s=8); ax[1].set_title("PCA | measured local dim")
    plt.colorbar(s1, ax=ax[1], fraction=0.046)
    s2 = ax[2].scatter(u2[:, 0], u2[:, 1], c=dof[ids], cmap="viridis", s=8); ax[2].set_title("UMAP | true DOF")
    plt.colorbar(s2, ax=ax[2], fraction=0.046)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"Stratified knobs ({tag}): does local dimension vary with DOF?", fontsize=11)
    fig.tight_layout()
    path = os.path.join(OUT, f"stratified_{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.relpath(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", default="plain", choices=["plain", "factored", "image"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--json_out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(OUT, exist_ok=True)
    print(f"device={device} STRATIFIED knobs variant={args.variant} states={NSTATES} seed={args.seed}")

    move_e, knob_e, move_nbrs, dof = build_env()
    edges = torch.cat([move_e, knob_e], dim=0)
    print(f"states={NSTATES} move_edges={move_e.shape[0]} knob_edges={knob_e.shape[0]} "
          f"DOF hist={np.bincount(dof.astype(int))}")

    model = make_model(args.variant).to(device)
    train(model, edges, args.steps, args.batch, args.lr, device,
          eval_every=args.eval_every, dof=dof, move_nbrs=move_nbrs)

    # full stratification eval on a large sample
    rng = np.random.default_rng(7)
    ids = rng.integers(0, NSTATES, 4000)
    ld, E = local_dims(model, device, ids, move_nbrs)
    corr = spearmanr(ld, dof[ids]).statistic
    # mean measured local dim per true DOF level
    per = {int(d): float(ld[dof[ids] == d].mean()) for d in sorted(set(dof[ids].astype(int)))}
    print("\n================ STRATIFICATION RESULTS ================")
    print(f"variant={args.variant}")
    print(f"Spearman(local_dim, true DOF) = {corr:.3f}")
    print(f"mean measured local dim per true DOF: {per}")
    print("========================================================")
    visualize(E, dof, ld, ids, args.variant)

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(dict(variant=args.variant, corr=float(corr), per_dof=per), f, indent=2)
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()

"""
Visualize the learned embeddings of the self_norm factored-attention model.

Two embedding sets:
  1. state embedding e(state) in R^emb_dim  (450 states) -- the actual metric space
     whose L-p norm is the learned distance. Colored by torus and by geodesic-to-bridge.
  2. z_pos = E_pos.weight (225 shared position vectors). Colored by grid row / col.

Projections: PCA (2D + 3D) and UMAP (2D). PNGs saved to factored_vis/.

Trains a fresh model (fast) unless a checkpoint exists (--load). Reuses the probe.
"""

import argparse
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

from bridged_tori_probe import (
    SIDE, N_POS, N_STATES, BRIDGE_POS, gid,
    build_graph, build_transitions, all_pairs_geodesic, torus_geo_to_bridge,
    FactoredAttentionModel, train_factored,
)

OUT = os.path.join(os.path.dirname(__file__), "..", "factored_vis")


def scatter(ax, xy, c, title, cmap, cbar_label=None, discrete=False, bridge_idx=None):
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap=cmap, s=14, alpha=0.85,
                    linewidths=0)
    if bridge_idx is not None:
        for bi in np.atleast_1d(bridge_idx):
            ax.scatter(xy[bi, 0], xy[bi, 1], c="red", s=140, marker="*",
                       edgecolors="black", linewidths=0.6, zorder=10)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])
    if cbar_label and not discrete:
        cb = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(cbar_label, fontsize=8)
    return sc


def scatter3d(fig, pos, xyz, c, title, cmap, bridge_idx=None):
    ax = fig.add_subplot(pos, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=c, cmap=cmap, s=12, alpha=0.8, linewidths=0)
    if bridge_idx is not None:
        for bi in np.atleast_1d(bridge_idx):
            ax.scatter(xyz[bi, 0], xyz[bi, 1], xyz[bi, 2], c="red", s=140, marker="*",
                       edgecolors="black", linewidths=0.6)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    return ax


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path))


@torch.no_grad()
def extract(model, device):
    model.eval()
    pos = torch.arange(N_POS, device=device)
    # state embeddings for both tori
    es = []
    for torus in (0, 1):
        t = torch.full((N_POS,), torus, dtype=torch.long, device=device)
        es.append(model.embed(model.state_tokens(pos, t)).cpu().numpy())
    e_state = np.concatenate(es, axis=0)                 # (450, emb_dim), order: torus0 then torus1
    z_pos = model.E_pos.weight.detach().cpu().numpy()    # (225, d_model)
    return e_state, z_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--load", action="store_true", help="load checkpoint if present")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)

    G = build_graph()
    trans = build_transitions()
    geo = all_pairs_geodesic(G)
    tg_bridge = torus_geo_to_bridge(G)

    model = FactoredAttentionModel(head="self_norm").to(device)
    ckpt = os.path.join(OUT, "model.pt")
    if args.load and os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print("loaded", ckpt)
    else:
        print(f"training self_norm {args.steps} steps ...")
        train_factored(model, trans, geo, tg_bridge, args.steps, 512, 2e-3, device,
                       eval_every=args.steps)
        torch.save(model.state_dict(), ckpt)
        print("saved", ckpt)

    e_state, z_pos = extract(model, device)

    # ---- colorings ----
    torus_c = np.array([0] * N_POS + [1] * N_POS)                      # 450
    dist_bridge = geo[gid(BRIDGE_POS, 0)]                              # geodesic to bridge, per state
    bridge_ids = np.array([gid(BRIDGE_POS, 0), gid(BRIDGE_POS, 1)])
    row_c = np.array([p // SIDE for p in range(N_POS)])                # 225
    col_c = np.array([p % SIDE for p in range(N_POS)])

    # ---- projections ----
    pca_s2 = PCA(n_components=2).fit(e_state)
    e_pca2 = pca_s2.transform(e_state)
    e_pca3 = PCA(n_components=3).fit_transform(e_state)
    e_umap2 = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=args.seed).fit_transform(e_state)
    zp_pca2 = PCA(n_components=2).fit_transform(z_pos)
    zp_umap2 = umap.UMAP(n_neighbors=15, min_dist=0.15, random_state=args.seed).fit_transform(z_pos)
    print(f"state PCA2 explained var: {pca_s2.explained_variance_ratio_[:2].round(3)}")

    # ---- individual figures ----
    f, a = plt.subplots(figsize=(5, 4)); scatter(a, e_pca2, torus_c, "state e | PCA | torus", "tab10", discrete=True, bridge_idx=bridge_ids); save(f, "state_pca2d_torus.png")
    f, a = plt.subplots(figsize=(5, 4)); scatter(a, e_pca2, dist_bridge, "state e | PCA | geodesic to bridge", "viridis", "dist to node150", bridge_idx=bridge_ids); save(f, "state_pca2d_distbridge.png")
    f, a = plt.subplots(figsize=(5, 4)); scatter(a, e_umap2, torus_c, "state e | UMAP | torus", "tab10", discrete=True, bridge_idx=bridge_ids); save(f, "state_umap2d_torus.png")
    f, a = plt.subplots(figsize=(5, 4)); scatter(a, e_umap2, dist_bridge, "state e | UMAP | geodesic to bridge", "viridis", "dist to node150", bridge_idx=bridge_ids); save(f, "state_umap2d_distbridge.png")

    f = plt.figure(figsize=(6, 5)); scatter3d(f, 111, e_pca3, torus_c, "state e | PCA 3D | torus", "tab10", bridge_idx=bridge_ids); save(f, "state_pca3d_torus.png")

    f, a = plt.subplots(figsize=(5, 4)); scatter(a, zp_pca2, row_c, "z_pos | PCA | grid row", "twilight", "row"); save(f, "zpos_pca2d_row.png")
    f, a = plt.subplots(figsize=(5, 4)); scatter(a, zp_pca2, col_c, "z_pos | PCA | grid col", "twilight", "col"); save(f, "zpos_pca2d_col.png")
    f, a = plt.subplots(figsize=(5, 4)); scatter(a, zp_umap2, row_c, "z_pos | UMAP | grid row", "twilight", "row"); save(f, "zpos_umap2d_row.png")
    f, a = plt.subplots(figsize=(5, 4)); scatter(a, zp_umap2, col_c, "z_pos | UMAP | grid col", "twilight", "col"); save(f, "zpos_umap2d_col.png")

    # ---- within-torus state embedding (torus 0 only), colored by grid row/col ----
    # this is the learned METRIC space restricted to one torus (unlike raw z_pos,
    # which the attention encoder transforms). Shows whether one torus is toroidal.
    e_t0 = e_state[:N_POS]
    e_t0_pca2 = PCA(n_components=2).fit_transform(e_t0)
    e_t0_umap2 = umap.UMAP(n_neighbors=15, min_dist=0.15, random_state=args.seed).fit_transform(e_t0)
    f, a = plt.subplots(figsize=(5, 4)); scatter(a, e_t0_pca2, row_c, "state e (torus x) | PCA | row", "twilight", "row"); save(f, "state_t0_pca2d_row.png")
    f, a = plt.subplots(figsize=(5, 4)); scatter(a, e_t0_pca2, col_c, "state e (torus x) | PCA | col", "twilight", "col"); save(f, "state_t0_pca2d_col.png")
    f, a = plt.subplots(figsize=(5, 4)); scatter(a, e_t0_umap2, row_c, "state e (torus x) | UMAP | row", "twilight", "row"); save(f, "state_t0_umap2d_row.png")
    f, a = plt.subplots(figsize=(5, 4)); scatter(a, e_t0_umap2, col_c, "state e (torus x) | UMAP | col", "twilight", "col"); save(f, "state_t0_umap2d_col.png")

    # ---- position overview (within-torus metric space) ----
    figp = plt.figure(figsize=(12, 4))
    scatter(figp.add_subplot(1, 3, 1), e_t0_pca2, row_c, "torus x e | PCA | row", "twilight", "row")
    scatter(figp.add_subplot(1, 3, 2), e_t0_pca2, col_c, "torus x e | PCA | col", "twilight", "col")
    scatter(figp.add_subplot(1, 3, 3), e_t0_umap2, col_c, "torus x e | UMAP | col", "twilight", "col")
    figp.suptitle("Within-torus (torus x) learned metric space, colored by grid coordinate", fontsize=11)
    figp.tight_layout()
    save(figp, "overview_position.png")

    # ---- overview panel ----
    fig = plt.figure(figsize=(15, 8))
    scatter(fig.add_subplot(2, 3, 1), e_pca2, torus_c, "state e | PCA | torus", "tab10", discrete=True, bridge_idx=bridge_ids)
    scatter(fig.add_subplot(2, 3, 2), e_pca2, dist_bridge, "state e | PCA | dist to bridge", "viridis", "dist", bridge_idx=bridge_ids)
    scatter(fig.add_subplot(2, 3, 3), e_umap2, torus_c, "state e | UMAP | torus", "tab10", discrete=True, bridge_idx=bridge_ids)
    scatter(fig.add_subplot(2, 3, 4), zp_pca2, row_c, "z_pos | PCA | row", "twilight", "row")
    scatter(fig.add_subplot(2, 3, 5), zp_pca2, col_c, "z_pos | PCA | col", "twilight", "col")
    scatter(fig.add_subplot(2, 3, 6), zp_umap2, col_c, "z_pos | UMAP | col", "twilight", "col")
    fig.suptitle("Factored-attention (self_norm) embeddings  |  red star = bridge node 150", fontsize=12)
    fig.tight_layout()
    save(fig, "overview.png")
    print("done ->", os.path.relpath(OUT))


if __name__ == "__main__":
    main()

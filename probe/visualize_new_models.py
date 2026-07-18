"""
Visualize the learned state-embedding geometry (PCA + UMAP) of three representative
15x15-grid image models from this session:
  1. image self_norm (2-factor: position + id)        -> two tori + bridge detour
  2. equivariant dynamics (2-factor)                  -> best geometry
  3. 3-factor self_norm (position + id + color, 900)  -> scaled-up map

Each model is retrained briefly, its state embedding e(state) extracted, and projected
with PCA (2D/3D) + UMAP (2D), colored by the relevant factors + geodesic-to-bridge.
Outputs -> factored_vis/newmodels_<tag>.png
"""

import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap

from bridged_tori_probe import (
    N_POS, N_STATES, BRIDGE_POS, gid,
    build_graph, build_transitions, all_pairs_geodesic, torus_geo_to_bridge,
)
from bridged_tori_image_probe import ImageFactoredAttentionModel, train_image_factored
from bridged_tori_equivariant_probe import EquivariantModel
import bridged_tori_3factor_probe as tf3

OUT = os.path.join(os.path.dirname(__file__), "..", "factored_vis")
STEPS = 3500
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sc(ax, xy, c, title, cmap, cbar=None, discrete=False, star=None):
    s = ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap=cmap, s=10, alpha=0.85, linewidths=0)
    if star is not None:
        ax.scatter(xy[star, 0], xy[star, 1], c="red", s=120, marker="*",
                   edgecolors="black", linewidths=0.5, zorder=10)
    ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    if cbar and not discrete:
        plt.colorbar(s, ax=ax, fraction=0.046, pad=0.02)


def sc3(fig, gsp, xyz, c, title, cmap, star=None):
    ax = fig.add_subplot(gsp, projection="3d")
    ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=c, cmap=cmap, s=8, alpha=0.8, linewidths=0)
    if star is not None:
        ax.scatter(xyz[star, 0], xyz[star, 1], xyz[star, 2], c="red", s=120, marker="*",
                   edgecolors="black", linewidths=0.5)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])


@torch.no_grad()
def embed_2factor(model):
    model.eval()
    ids = torch.arange(N_STATES, device=DEVICE)
    pos, torus = ids % N_POS, ids // N_POS
    out = []
    for i in range(0, N_STATES, 4096):
        out.append(model.embed(model.state_tokens(pos[i:i+4096], torus[i:i+4096])).cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def embed_3factor(model):
    model.eval()
    ids = torch.arange(tf3.N_STATES, device=DEVICE)
    out = []
    for i in range(0, tf3.N_STATES, 4096):
        out.append(model.embed(model.state_tokens(ids[i:i+4096])).cpu().numpy())
    return np.concatenate(out)


def viz_2factor(model, tag, title):
    e = embed_2factor(model)
    G = build_graph(); geo = all_pairs_geodesic(G)
    torus = np.array([s // N_POS for s in range(N_STATES)])
    dist = geo[gid(BRIDGE_POS, 0)]
    stars = [gid(BRIDGE_POS, 0), gid(BRIDGE_POS, 1)]
    p2 = PCA(2).fit(e); e2 = p2.transform(e); e3 = PCA(3).fit_transform(e)
    u2 = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=1).fit_transform(e)
    fig = plt.figure(figsize=(15, 4))
    sc(fig.add_subplot(1, 4, 1), e2, torus, f"PCA | torus  (var {p2.explained_variance_ratio_[:2].round(2)})", "tab10", discrete=True, star=stars)
    sc(fig.add_subplot(1, 4, 2), e2, dist, "PCA | dist to bridge", "viridis", cbar=1, star=stars)
    sc3(fig, 143, e3, torus, "PCA 3D | torus", "tab10", star=stars)
    sc(fig.add_subplot(1, 4, 4), u2, torus, "UMAP | torus", "tab10", discrete=True, star=stars)
    fig.suptitle(f"{title}  (red star = bridge node 150)", fontsize=11)
    fig.tight_layout(); _save(fig, tag)


def viz_3factor(model, tag, title):
    e = embed_3factor(model)
    G = tf3.build_graph(); geo = tf3.all_pairs_geodesic(G)
    ids = np.arange(tf3.N_STATES)
    color = ids // (tf3.N_POS * 2); rem = ids % (tf3.N_POS * 2); torus = rem // tf3.N_POS
    dist = geo[tf3.gid3(BRIDGE_POS, 0, 0)]
    p2 = PCA(2).fit(e); e2 = p2.transform(e); e3 = PCA(3).fit_transform(e)
    u2 = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=1).fit_transform(e)
    fig = plt.figure(figsize=(15, 8))
    sc(fig.add_subplot(2, 3, 1), e2, torus, "PCA | torus (id)", "tab10", discrete=True)
    sc(fig.add_subplot(2, 3, 2), e2, color, "PCA | color", "coolwarm", discrete=True)
    sc(fig.add_subplot(2, 3, 3), e2, dist, "PCA | dist to bridge", "viridis", cbar=1)
    sc3(fig, 234, e3, torus, "PCA 3D | torus", "tab10")
    sc(fig.add_subplot(2, 3, 5), u2, torus, "UMAP | torus", "tab10", discrete=True)
    sc(fig.add_subplot(2, 3, 6), u2, color, "UMAP | color", "coolwarm", discrete=True)
    fig.suptitle(f"{title}", fontsize=11)
    fig.tight_layout(); _save(fig, tag)


def _save(fig, tag):
    path = os.path.join(OUT, f"newmodels_{tag}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.relpath(path))


def main():
    os.makedirs(OUT, exist_ok=True)
    torch.manual_seed(1); np.random.seed(1)

    G = build_graph(); trans = build_transitions()
    geo = all_pairs_geodesic(G); tgb = torus_geo_to_bridge(G)

    print("training 1/3: image self_norm ...")
    m1 = ImageFactoredAttentionModel().to(DEVICE)
    train_image_factored(m1, trans, geo, tgb, STEPS, 512, 2e-3, DEVICE, eval_every=STEPS)
    viz_2factor(m1, "image_selfnorm", "Image self_norm (2-factor: position + id)")

    print("training 2/3: equivariant dynamics ...")
    m2 = EquivariantModel().to(DEVICE)
    train_image_factored(m2, trans, geo, tgb, STEPS, 512, 2e-3, DEVICE, eval_every=STEPS)
    viz_2factor(m2, "equivariant", "Equivariant dynamics (2-factor)")

    print("training 3/3: 3-factor self_norm ...")
    G3 = tf3.build_graph(); trans3 = tf3.build_transitions(); geo3 = tf3.all_pairs_geodesic(G3)
    m3 = tf3.ThreeFactorModel(image=True).to(DEVICE)
    tf3.train(m3, trans3, geo3, STEPS + 1500, 512, 2e-3, DEVICE, aux_mode="none", eval_every=STEPS + 1500)
    viz_3factor(m3, "threefactor", "3-factor self_norm (position + id + color)")

    print("done ->", os.path.relpath(OUT))


if __name__ == "__main__":
    main()

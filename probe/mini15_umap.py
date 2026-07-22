"""UMAP the per-component embeddings (PCA was linear and only caught ~22% of variance). Also report a
PROJECTION-FREE ring metric: the fraction of positions whose true nearest neighbour in the FULL
embedding is an adjacent position (+-1 on the ring). If that fraction is high the ring order is really
preserved, whatever a 2-D projection shows; UMAP should then unroll it into a circle.

Trains the strong-isometry config (no-displacement gate, multi-scale isometry, low repel margin).
"""
import argparse, os, numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import umap
from sklearn.decomposition import PCA
import mini11_estimators as m11
from mini12_gate_guided import train
from mini14_strong_iso import AttnDist1D_iso

CKPT = "/home/jan/projects/CIIRC/colabs/Alma/cognitiveMapsPlusPlus/factored_vis/mini15_model.pt"

def nn_adjacent_frac(W, k=1):
    """fraction of positions whose k nearest neighbours are ALL adjacent within +-k on the ring."""
    G = len(W); D = np.linalg.norm(W[:, None] - W[None], axis=2); np.fill_diagonal(D, np.inf)
    knn = np.argsort(D, 1)[:, :k]; ok = 0
    for p in range(G):
        offs = {((q - p + G // 2) % G) - G // 2 for q in knn[p]}
        if offs <= set(range(-k, k + 1)) - {0}: ok += 1
    return ok / G

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000); ap.add_argument("--lam_iso", type=float, default=3.0)
    ap.add_argument("--margin", type=float, default=10.0); ap.add_argument("--d", type=int, default=48)
    ap.add_argument("--nn", type=int, default=3); ap.add_argument("--load", action="store_true"); args = ap.parse_args()
    N, G = 5, 48; m11.N, m11.G, m11.Kk = N, G, N
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  margin={args.margin} lam_iso={args.lam_iso} d={args.d}  umap n_neighbors={args.nn}", flush=True)
    rng = np.random.default_rng(0); torch.manual_seed(0); np.random.seed(0)
    if args.load and os.path.exists(CKPT):
        head = AttnDist1D_iso(d=args.d); head.load_state_dict(torch.load(CKPT, map_location="cpu")); print("loaded", CKPT, flush=True)
    else:
        head = train(AttnDist1D_iso(d=args.d), args.steps, rng, device, K=6, lam_iso=args.lam_iso, margin=args.margin).cpu()
        torch.save(head.state_dict(), CKPT); print("saved", CKPT, flush=True)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for i in range(N):
        W = head.pos[i].weight.detach().numpy()
        Y = umap.UMAP(n_components=2, n_neighbors=args.nn, min_dist=0.15, random_state=0).fit_transform(W)
        f1, f2 = nn_adjacent_frac(W, 1), nn_adjacent_frac(W, 2)
        print(f"agent {i}: 1NN-adjacent={f1:.2f}  2NN-both-adjacent={f2:.2f}", flush=True)
        frac = f1
        ax = axes.flat[i]; loop = np.vstack([Y, Y[0]])
        ax.plot(loop[:, 0], loop[:, 1], '-', color="#0e8f8a", alpha=.4, lw=1)
        ax.scatter(Y[:, 0], Y[:, 1], c=np.arange(G), cmap="twilight", s=40, zorder=3)
        ax.set_title(f"agent {i}   1NN-adj={f1:.2f} 2NN-adj={f2:.2f}", fontsize=11)
    # key: only 5 points -> PCA, not UMAP
    Wk = head.key.weight.detach().numpy(); Yk = PCA(2).fit_transform(Wk)
    ax = axes.flat[5]; ax.plot(Yk[:, 0], Yk[:, 1], '-o', color="#d9622f")
    for k in range(len(Yk)): ax.annotate(str(k), Yk[k], fontsize=9)
    ax.set_title("key (PCA, 5 pts)", fontsize=11)
    kf = nn_adjacent_frac(Wk); print(f"key: NN-adjacent = {kf:.3f}", flush=True)
    plt.suptitle(f"Per-component embeddings via UMAP (n_neighbors={args.nn}), colored+connected in position order", fontsize=12)
    plt.tight_layout()
    out = "/home/jan/projects/CIIRC/colabs/Alma/cognitiveMapsPlusPlus/factored_vis/mini15_umap.png"
    plt.savefig(out, dpi=110, bbox_inches="tight"); print("saved", out, flush=True)

if __name__ == "__main__":
    main()

"""PCA the individual component embeddings from the mini12 (multi-scale isometry) model: do the
per-agent position embeddings form RINGS (cyclic order preserved), and the key a short path?

For each agent i we take its position embedding table pos[i].weight (G x d), PCA to 2-D, and plot the
48 points connected in position order 0,1,...,47,0. A clean isometric cycle -> a circle traversed in
order. We also report: fraction of variance in PC1+PC2 (how planar), and 'ring monotonicity' = how
consistently the polar angle advances with position index (1.0 = perfect ring order).
"""
import numpy as np, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import mini11_estimators as m11
from mini12_gate_guided import train

def ringness(W):
    p = PCA(n_components=min(W.shape)).fit(W); ev = p.explained_variance_ratio_
    Y = p.transform(W)[:, :2]; c = Y.mean(0)
    ang = np.arctan2(Y[:, 1] - c[1], Y[:, 0] - c[0])
    dif = np.diff(np.unwrap(ang))                                   # angle advance per position step
    mono = max((dif > 0).mean(), (dif < 0).mean())                 # fraction advancing the same way
    return Y, ev, mono

def main():
    N, G, steps = 5, 48, 9000
    m11.N, m11.G, m11.Kk = N, G, N
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  training mini12 model (N={N} G={G})...", flush=True)
    rng = np.random.default_rng(0); torch.manual_seed(0); np.random.seed(0)
    head = train(m11.AttnDist1D(), steps, rng, device).cpu()

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for i in range(N):
        W = head.pos[i].weight.detach().numpy()
        Y, ev, mono = ringness(W)
        ax = axes.flat[i]
        loop = np.vstack([Y, Y[0]])
        ax.plot(loop[:, 0], loop[:, 1], '-', color="#0e8f8a", alpha=.45, lw=1)
        sc = ax.scatter(Y[:, 0], Y[:, 1], c=np.arange(G), cmap="twilight", s=36, zorder=3)
        ax.set_title(f"agent {i}   PC1+2 var={ev[:2].sum():.2f}   ring={mono:.2f}", fontsize=11)
        ax.set_aspect("equal"); ax.tick_params(labelsize=7)
        print(f"agent {i}: PC1+2 var={ev[:2].sum():.3f}  top5={np.round(ev[:5],3)}  ring-monotonicity={mono:.3f}", flush=True)
    # key embedding (a short gated path 0..4)
    Wk = head.key.weight.detach().numpy()
    Yk, evk, _ = ringness(Wk)
    ax = axes.flat[5]
    ax.plot(Yk[:, 0], Yk[:, 1], '-o', color="#d9622f")
    for k in range(len(Yk)): ax.annotate(str(k), Yk[k], fontsize=9)
    ax.set_title(f"key   PC1+2 var={evk[:2].sum():.2f}", fontsize=11); ax.set_aspect("equal")
    print(f"key: PC1+2 var={evk[:2].sum():.3f}  top5={np.round(evk[:5],3)}", flush=True)

    plt.suptitle("Per-component embeddings (PCA to 2-D), points colored+connected in position order", fontsize=12)
    plt.tight_layout()
    out = "/home/jan/projects/CIIRC/colabs/Alma/cognitiveMapsPlusPlus/factored_vis/mini12_component_pca.png"
    plt.savefig(out, dpi=110, bbox_inches="tight")
    print("saved", out, flush=True)

    # local-isometry check: embedding distance vs cyclic graph distance for agent 0 (should be ~linear up to K)
    W0 = head.pos[0].weight.detach().numpy()
    gd = np.array([min(k, G - k) for k in range(G)])
    ed = np.linalg.norm(W0 - W0[0], axis=1)
    fig2, ax2 = plt.subplots(figsize=(6, 5))
    ax2.scatter(gd, ed, s=24, color="#0e8f8a")
    ax2.set_xlabel("cyclic graph distance from position 0"); ax2.set_ylabel("embedding distance")
    ax2.set_title("agent 0: local isometry (linear up to K=6, then off)")
    out2 = "/home/jan/projects/CIIRC/colabs/Alma/cognitiveMapsPlusPlus/factored_vis/mini12_isometry_check.png"
    plt.savefig(out2, dpi=110, bbox_inches="tight"); print("saved", out2, flush=True)

if __name__ == "__main__":
    main()

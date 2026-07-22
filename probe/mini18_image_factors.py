"""Do the IMAGE encoder's internal factors reflect the expected behavior? Loads a trained ImageDist
and checks, for the per-component vectors it extracts from one shared canvas:

  1. POSITION ENCODING: vary one agent over all G*G cells (others fixed); its component vector should
     trace the 2-D grid -> fraction of cells whose nearest component-neighbour is a grid-adjacent cell.
  2. DISENTANGLEMENT: while agent i moves, agent j's (j!=i) component vector should stay ~constant
     -> report leakage (std of comp_j across agent-i's sweep, relative to comp_i's own spread).
  3. GATE: the gate weight for agent i should be small when free, large when frozen (per key).
  4. ATTENTION: query_i should attend to marker i's cell -> argmax attention cell vs true agent cell.

Saves a PCA/UMAP grid of each agent's component-vs-position manifold.
"""
import argparse, numpy as np, torch, torch.nn.functional as F, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import mini17_image as im

CKPT = "/home/jan/projects/CIIRC/colabs/Alma/cognitiveMapsPlusPlus/factored_vis/mini17_image.pt"

def grid_adj(p, G):
    r, c = p // G, p % G
    return {((r + 1) % G) * G + c, ((r - 1) % G) * G + c, r * G + (c + 1) % G, r * G + (c - 1) % G}

def nn_grid_adjacent(comp, G):
    NP = comp.shape[0]; D = np.linalg.norm(comp[:, None] - comp[None], axis=2); np.fill_diagonal(D, np.inf)
    nn = D.argmin(1)
    return float(np.mean([nn[p] in grid_adj(p, G) for p in range(NP)]))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--G", type=int, default=8); ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--encoder", default="mha"); args = ap.parse_args()
    im.set_G(args.G); N, G, NP = im.N, im.G, im.NP
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    head = im.ImageDist(d=args.d, heads=args.heads, encoder=args.encoder); head.load_state_dict(torch.load(args.ckpt, map_location="cpu")); head.to(dev).eval()
    print(f"loaded {args.ckpt}  N={N} G={G} d={args.d}\n")

    base = np.array([[3, 20, 40][:N] + [2]])          # key=2 -> all free, distinct positions
    others_by = {i: [base[0, j] for j in range(N) if j != i] for i in range(N)}
    fig, axes = plt.subplots(1, N, figsize=(5 * N, 5))
    with torch.no_grad():
        for i in range(N):
            cells = np.array([p for p in range(NP) if p not in others_by[i]])           # COLLISION-FREE sweep
            states = np.tile(base, (len(cells), 1)); states[:, i] = cells
            comp = head.components(torch.as_tensor(states, device=dev)).cpu().numpy()    # (M, N+1, d)
            ci = comp[:, i]
            adj = nn_grid_adjacent(ci, G) if len(cells) == NP else _adj_on(ci, cells, G)
            leaks = [comp[:, j].std(0).mean() / max(ci.std(0).mean(), 1e-6) for j in range(N) if j != i]
            print(f"agent {i}: grid-adjacent-NN={adj:.2f}   leakage(other agents' drift / own spread)={np.mean(leaks):.3f}")
            Y = PCA(2).fit_transform(ci)
            axes[i].scatter(Y[:, 0], Y[:, 1], c=cells % G + (cells // G) * G, cmap="twilight", s=28)
            axes[i].set_title(f"agent {i}: comp vs position (grid-NN={adj:.2f})", fontsize=10); axes[i].set_aspect("equal")
        plt.suptitle("Image-extracted component vectors as one agent sweeps the grid (PCA-2D)", fontsize=12)
        plt.tight_layout(); out = "/home/jan/projects/CIIRC/colabs/Alma/cognitiveMapsPlusPlus/factored_vis/mini18_image_factors.png"
        plt.savefig(out, dpi=110, bbox_inches="tight"); print("saved", out)

        # 3. gate behaviour on a REAL move pair: w_i actually applied when agent k moves 1 step
        print("\n[gate] w_i applied to agent k's own 1-step move (free should be small, frozen large):")
        for m in range(N):
            p = np.array([3, 20, 40][:N] + [m]); pt = torch.as_tensor(p[None], device=dev); cx = head.components(pt)
            row = []
            for k in range(N):
                q = p.copy(); q[k] = im.step2d(q[k], 1, 0); cy = head.components(torch.as_tensor(q[None], device=dev))
                ax = [cx[:, t] for t in range(N)]; ay = [cy[:, t] for t in range(N)]; kx, ky = cx[:, N], cy[:, N]
                ctx = torch.stack([a + b for a, b in zip(ax + [kx], ay + [ky])], 0).mean(0)
                w = float(F.softplus(head.gate(torch.cat([ax[k] + ay[k], ctx], -1)))[0])
                dc = float(torch.norm(ax[k] - ay[k]))                                    # component change magnitude
                fr = "free" if (k == 0 or m >= k) else "FROZ"
                row.append(f"a{k}[{fr}] w={w:.2f} dcomp={dc:.2f} w*dc={w*dc:.1f}")
            print(f"  key={m}: " + "  ".join(row))

        # 4. attention localization via REAL attention weights (encoder-aware)
        print("\n[attn] query_i argmax-attention cell vs agent i's true cell:")
        p = np.array([[3, 20, 40][:N] + [2]]); img = im.render(torch.as_tensor(p, device=dev), dev)
        vale = head.val(img); tok = vale + head.pe(torch.arange(NP + 1, device=dev))[None]; ids = torch.arange(N + 1, device=dev)
        if args.encoder == "gather":
            aw = torch.softmax(torch.einsum('cd,bnd->bcn', head.mk(ids), head.vproj(vale)) / (args.d ** 0.5), -1)
        elif args.encoder == "gather2":
            aw = torch.softmax(torch.einsum('cd,bnd->bcn', head.mk(ids[:N]), head.vproj(vale[:, :NP])) / (args.d ** 0.5), -1)
        else:
            q = head.query[None] if args.encoder == "mha" else (head.query + head.mk(ids))[None]
            _, aw = head.attn(q, tok, tok, need_weights=True, average_attn_weights=True)
        for i in range(N):
            amax = int(aw[0, i].argmax()); print(f"  agent {i}: true cell={p[0, i]}, attn-argmax cell={amax}, match={amax == p[0, i]}, wt={float(aw[0,i,amax]):.2f}")

def _adj_on(comp, cells, G):
    idx = {int(c): k for k, c in enumerate(cells)}; D = np.linalg.norm(comp[:, None] - comp[None], axis=2); np.fill_diagonal(D, np.inf)
    nn = D.argmin(1); ok = 0
    for k, c in enumerate(cells):
        nbr = grid_adj(int(c), G) & set(idx)
        if cells[nn[k]] in nbr: ok += 1
    return ok / len(cells)

if __name__ == "__main__":
    main()

"""
Image-input variant of the bridged-tori factored-attention probe.

Same graph / dynamics / distance head / losses / eval as `bridged_tori_probe.py`.
ONLY the input representation changes: a state is rendered as a 15x15 image with
per-pixel values in {0,1,2} and the 2 factor tokens [z_pos, z_id] are EXTRACTED from
that image by cross-attention (two learned query vectors over the 225 pixels).

Image encoding (per pixel in {0,1,2}):
  0 = background, 1 = agent (exactly one pixel = the position), 2 = "on torus 2" marker.
  Torus identity is carried by TWO fixed marker pixels (redundant): value 2 on torus 2,
  0 on torus 1. Two of them so that if the agent sits on one marker (renders as 1), the
  other still shows 2 -> identity always visible.

Encoder: pixel token = value_emb + learned per-pixel positional_emb. Two learned query
vectors cross-attend the 225 pixel tokens:
  Q_pos should learn to focus on the agent pixel (value 1) -> position,
  Q_id  should learn to focus on the marker pixels (value 2 vs not) -> which torus.
Their outputs are [z_pos, z_id], fed to the SAME self_norm distance head + dynamics.

Dynamics operate on the post-attention factor tokens (option A) -- pure encoder swap.
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bridged_tori_probe import (
    SIDE, N_POS, N_STATES, BRIDGE_POS, gid,
    build_graph, build_transitions, all_pairs_geodesic, torus_geo_to_bridge,
    FactoredAttentionModel, train_factored, train_baseline, final_eval,
)

MARKERS = (0, N_POS - 1)   # two fixed identity-marker pixels (corners: (0,0) and (14,14))
OUT = os.path.join(os.path.dirname(__file__), "..", "factored_vis")


def render_batch(pos, torus, device, markers=MARKERS):
    """(B,) pos + (B,) torus -> (B, 225) long pixel grid in {0,1,2}."""
    B = pos.shape[0]
    grid = torch.zeros(B, N_POS, dtype=torch.long, device=device)
    is_t2 = (torus == 1)
    if is_t2.any():
        for m in markers:
            grid[is_t2, m] = 2          # markers first
    grid[torch.arange(B, device=device), pos] = 1   # agent overwrites (occlusion)
    return grid


class ImageFactoredAttentionModel(FactoredAttentionModel):
    """Extracts [z_pos, z_id] from the rendered image via 2-query cross-attention,
    then reuses the inherited self_norm distance head + dynamics."""

    def __init__(self, d_model=32, n_layers=2, n_heads=4, emb_dim=16, p=1.5, markers=MARKERS):
        super().__init__(head="self_norm", d_model=d_model, n_layers=n_layers,
                         n_heads=n_heads, emb_dim=emb_dim, p=p)
        self.markers = markers
        self.pixel_val_emb = nn.Embedding(3, d_model)          # {0,1,2}
        self.pixel_pos_emb = nn.Embedding(N_POS, d_model)      # per-pixel location
        self.img_query = nn.Parameter(torch.randn(2, d_model) * 0.02)  # [Q_pos, Q_id]
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.enc_norm = nn.LayerNorm(d_model)
        self._last_attn = None                                 # (B,2,225) for viz/stats

    def render(self, pos, torus):
        return render_batch(pos, torus, pos.device, self.markers)

    def encode(self, grid):
        """(B,225) long -> (B,2,d) factor tokens; caches attention weights."""
        B = grid.shape[0]
        idx = torch.arange(N_POS, device=grid.device)
        pix = self.pixel_val_emb(grid) + self.pixel_pos_emb(idx).unsqueeze(0)  # (B,225,d)
        Q = self.img_query.unsqueeze(0).expand(B, -1, -1)                      # (B,2,d)
        z, attn = self.cross_attn(Q, pix, pix, need_weights=True, average_attn_weights=True)
        self._last_attn = attn.detach()                                       # (B,2,225)
        return self.enc_norm(z)                                               # (B,2,d)

    def state_tokens(self, pos, torus):        # overrides embedding-lookup version
        return self.encode(self.render(pos, torus))


class ImageMLPBaseline(nn.Module):
    """MLP over the flattened one-hot image + L-p distance. No attention."""

    def __init__(self, emb_dim=16, hidden=128, p=1.5, markers=MARKERS):
        super().__init__()
        self.p = p
        self.markers = markers
        self.net = nn.Sequential(
            nn.Linear(N_POS * 3, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, emb_dim),
        )

    def embed_ids(self, ids):
        pos, torus = ids % N_POS, ids // N_POS
        grid = render_batch(pos, torus, ids.device, self.markers)
        onehot = F.one_hot(grid, 3).float().view(grid.shape[0], -1)
        return self.net(onehot)

    def distance_ids(self, a_ids, b_ids):
        return torch.norm(self.embed_ids(a_ids) - self.embed_ids(b_ids), p=self.p, dim=-1)


# ---------- attention diagnostics ----------
@torch.no_grad()
def attention_stats(model, device):
    """For each of the 450 states, how much attention each query puts on the agent
    pixel vs the marker pixels. Reveals whether the 2 queries specialize."""
    model.eval()
    ids = torch.arange(N_STATES, device=device)
    pos, torus = ids % N_POS, ids // N_POS
    model.encode(render_batch(pos, torus, device, model.markers))
    attn = model._last_attn                                   # (450,2,225)
    ar = torch.arange(N_STATES, device=device)
    agent_mass = attn[:, :, :].gather(2, pos.view(-1, 1, 1).expand(-1, 2, 1)).squeeze(-1)  # (450,2)
    marker_mass = sum(attn[:, :, m] for m in model.markers)   # (450,2)
    # focus accuracy: does the query's argmax land on agent / a marker?
    amax = attn.argmax(dim=2)                                 # (450,2)
    pos_hit = (amax == pos.view(-1, 1)).float().mean(0)       # per query
    marker_hit = torch.zeros(2, device=device)
    for m in model.markers:
        marker_hit += (amax == m).float().mean(0)
    return dict(
        agent_mass=agent_mass.mean(0).cpu().numpy(),
        marker_mass=marker_mass.mean(0).cpu().numpy(),
        pos_argmax_acc=pos_hit.cpu().numpy(),
        marker_argmax_acc=marker_hit.cpu().numpy(),
    )


@torch.no_grad()
def save_attention_maps(model, device, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model.eval()
    # example states: interior torus1, interior torus2, agent occluding a marker on torus2
    examples = [
        (112, 0, "torus x, pos 112"),
        (112, 1, "torus y, pos 112"),
        (model.markers[0], 1, "torus y, agent ON marker"),
    ]
    fig, axes = plt.subplots(len(examples), 3, figsize=(9, 3 * len(examples)))
    for r, (pos, torus, label) in enumerate(examples):
        p = torch.tensor([pos], device=device)
        t = torch.tensor([torus], device=device)
        grid = render_batch(p, t, device, model.markers)
        model.encode(grid)
        attn = model._last_attn[0].cpu().numpy()              # (2,225)
        g = grid[0].cpu().numpy().reshape(SIDE, SIDE)
        axes[r, 0].imshow(g, cmap="viridis", vmin=0, vmax=2)
        axes[r, 0].set_title(f"input  ({label})", fontsize=9)
        for q in range(2):
            ax = axes[r, 1 + q]
            ax.imshow(attn[q].reshape(SIDE, SIDE), cmap="magma")
            ax.set_title(f"query {q} attention", fontsize=9)
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    fig.suptitle("Image input (col 0: 0=bg,1=agent,2=marker) and per-query attention", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--json_out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(OUT, exist_ok=True)
    print(f"device={device} IMAGE input, markers={MARKERS} steps={args.steps} seed={args.seed}")

    G = build_graph()
    trans = build_transitions()
    geo = all_pairs_geodesic(G)
    tg_bridge = torus_geo_to_bridge(G)

    print("\ntraining image factored-attention (self_norm head) ...")
    model_f = ImageFactoredAttentionModel().to(device)
    train_factored(model_f, trans, geo, tg_bridge, args.steps, args.batch, args.lr, device,
                   eval_every=args.eval_every)

    print("\ntraining image MLP baseline ...")
    model_b = ImageMLPBaseline().to(device)
    train_baseline(model_b, trans, geo, args.steps, args.batch, args.lr, device,
                   eval_every=args.eval_every)

    res = final_eval(model_f, model_b, geo, tg_bridge, device, seed=args.seed)

    st = attention_stats(model_f, device)
    print("\n---- attention specialization (per query [q0, q1]) ----")
    print(f"agent-pixel attention mass : {st['agent_mass'].round(3)}")
    print(f"marker-pixel attention mass: {st['marker_mass'].round(3)}")
    print(f"argmax-on-agent  accuracy  : {st['pos_argmax_acc'].round(3)}")
    print(f"argmax-on-marker accuracy  : {st['marker_argmax_acc'].round(3)}")
    save_attention_maps(model_f, device, os.path.join(OUT, "image_attention_maps.png"))

    if args.json_out:
        res.update({k: v.tolist() for k, v in st.items()})
        with open(args.json_out, "w") as f:
            json.dump(res, f, indent=2)
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()

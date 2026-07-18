"""
Experiment A: equivariant / constrained dynamics (NO disentanglement loss).

Same image encoder + self_norm head as bridged_tori_image_probe, but the latent
dynamics for the 4 MOVE actions are constrained to a SHARED, state-independent LINEAR
map z' = M[a] @ z (one learned matrix per move action). next/prev stay a free MLP
(their effect is conditional on being at the bridge, so they can't be a global map).

Idea (group-theoretic / grid-cell view): if every "move right" is the same linear
transform everywhere, the only latent geometry that predicts moves on a wraparound
torus is one where position lives on a rotation orbit (a torus) and identity lies in
the move-fixed subspace. So position/identity factorize from the dynamics constraint
alone -- no disentanglement penalty.

Measured: geometry (full geodesic + detour) and per-token identity/position sensitivity.
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn

from bridged_tori_probe import (
    build_graph, build_transitions, all_pairs_geodesic, torus_geo_to_bridge,
    _sample_pairs, spearman_geo, detour_signature,
)
from bridged_tori_image_probe import (
    ImageFactoredAttentionModel, train_image_factored, attention_stats,
    save_attention_maps, OUT,
)
from bridged_tori_multitask_probe import disentangle_stats


class EquivariantModel(ImageFactoredAttentionModel):
    """Move actions act as a shared state-independent linear map on the latent."""

    def __init__(self, **kw):
        super().__init__(**kw)
        D = 2 * self.d_model
        # one learned matrix per move action, initialized near identity
        eye = torch.eye(D).unsqueeze(0).repeat(4, 1, 1)
        self.move_maps = nn.Parameter(eye + 0.01 * torch.randn(4, D, D))
        self.switch_dyn = nn.Sequential(nn.Linear(D, 128), nn.GELU(), nn.Linear(128, D))

    def dynamics(self, tok, action):
        B = tok.shape[0]
        D = 2 * self.d_model
        z = tok.reshape(B, D)
        move_out = torch.bmm(self.move_maps[action.clamp(max=3)], z.unsqueeze(-1)).squeeze(-1)
        switch_out = z + self.switch_dyn(z)
        is_move = (action < 4).unsqueeze(-1)
        out = torch.where(is_move, move_out, switch_out)
        return out.reshape(B, 2, self.d_model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
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
    print(f"device={device} EQUIVARIANT dynamics (moves = shared linear map), seed={args.seed}")

    G = build_graph()
    trans = build_transitions()
    geo = all_pairs_geodesic(G)
    tg_bridge = torus_geo_to_bridge(G)

    model = EquivariantModel().to(device)
    train_image_factored(model, trans, geo, tg_bridge, args.steps, args.batch, args.lr, device,
                         aux_mode="none", eval_every=args.eval_every)

    full_sp = spearman_geo(model, geo, *_sample_pairs(6000, seed=7), device)
    det, _ = detour_signature(model, tg_bridge, device)
    id_s, pos_s = disentangle_stats(model, device)
    st = attention_stats(model, device)

    print("\n================ EQUIVARIANT RESULTS ================")
    print(f"full Spearman(D, geodesic) = {full_sp:.3f}   detour = {det:.3f}")
    print(f"token identity-sensitivity [q0,q1] = {id_s.round(3)}")
    print(f"token position-sensitivity [q0,q1] = {pos_s.round(3)}")
    print(f"attention argmax-on-agent  [q0,q1] = {st['pos_argmax_acc'].round(3)}")
    print(f"attention argmax-on-marker [q0,q1] = {st['marker_argmax_acc'].round(3)}")
    print("=====================================================")
    save_attention_maps(model, device, os.path.join(OUT, "image_attention_maps_equivariant.png"))

    if args.json_out:
        res = dict(full_sp=float(full_sp), det=float(det),
                   id_sens=id_s.tolist(), pos_sens=pos_s.tolist(),
                   agent_argmax=st['pos_argmax_acc'].tolist(),
                   marker_argmax=st['marker_argmax_acc'].tolist())
        with open(args.json_out, "w") as f:
            json.dump(res, f, indent=2)
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()

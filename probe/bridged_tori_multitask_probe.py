"""
Multi-task disentanglement probe (NO disentanglement loss / "trick").

Same image encoder as bridged_tori_image_probe (2-query cross-attention -> [z_pos,z_id]),
but with TWO readout heads, both reading BOTH factors:
  - full head : bridged-tori geodesic (needs position AND identity).
  - pos  head : position-only metric -- move steps cost 1, torus-switch costs 0
                (identity irrelevant). Needs ONLY position.

Bet: to serve the position-only head, the encoder must expose position identity-free;
to serve the full head it must also expose identity. If the two factors specialize as a
result, disentanglement emerged from TASK STRUCTURE, not from a disentanglement penalty.

Disentanglement is measured attention-free:
  identity_sensitivity(token) = mean_p || tok(p, torus0) - tok(p, torus1) ||
  position_sensitivity(token) = mean over move-neighbors || tok(a) - tok(b) ||
A disentangled encoder has one token high-identity/low-position (= z_id) and the other
low-identity/high-position (= z_pos).
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bridged_tori_probe import (
    SIDE, N_POS, N_STATES, BRIDGE_POS, NEXT, PREV, gid,
    build_graph, build_transitions, all_pairs_geodesic, torus_geo_to_bridge,
    _sample_pairs, spearman_geo, detour_signature,
)
from bridged_tori_image_probe import (
    ImageFactoredAttentionModel, render_batch, attention_stats, save_attention_maps,
    MARKERS, OUT,
)
from scipy.stats import spearmanr


def torus_geo_matrix():
    """within-torus geodesic (toroidal Manhattan), 225x225, identity ignored."""
    r = np.arange(N_POS) // SIDE
    c = np.arange(N_POS) % SIDE
    dr = np.abs(r[:, None] - r[None, :]); dr = np.minimum(dr, SIDE - dr)
    dc = np.abs(c[:, None] - c[None, :]); dc = np.minimum(dc, SIDE - dc)
    return (dr + dc).astype(np.float32)


class MultiTaskModel(ImageFactoredAttentionModel):
    def __init__(self, **kw):
        super().__init__(**kw)
        d = self.d_model
        self.pos_head = nn.Sequential(nn.Linear(2 * d, 128), nn.GELU(), nn.Linear(128, 16))
        self.id_head = nn.Sequential(nn.Linear(2 * d, 128), nn.GELU(), nn.Linear(128, 16))

    def dist_pos(self, tok_u, tok_v):
        return torch.norm(self.pos_head(tok_u.flatten(1)) - self.pos_head(tok_v.flatten(1)), p=1.5, dim=-1)

    def dist_id(self, tok_u, tok_v):
        return torch.norm(self.id_head(tok_u.flatten(1)) - self.id_head(tok_v.flatten(1)), p=1.5, dim=-1)


@torch.no_grad()
def disentangle_stats(model, device):
    """attention-free: per-token identity vs position sensitivity."""
    model.eval()
    pos = torch.arange(N_POS, device=device)
    t0 = model.state_tokens(pos, torch.zeros_like(pos))       # (225,2,d) torus 0
    t1 = model.state_tokens(pos, torch.ones_like(pos))        # (225,2,d) torus 1
    id_sens = (t0 - t1).norm(dim=2).mean(0)                   # (2,) per token
    # position sensitivity: move to right-neighbor on torus 0
    r = pos // SIDE; c = pos % SIDE
    cn = (c + 1) % SIDE
    posn = r * SIDE + cn
    tn = model.state_tokens(posn, torch.zeros_like(pos))
    pos_sens = (t0 - tn).norm(dim=2).mean(0)                  # (2,)
    return id_sens.cpu().numpy(), pos_sens.cpu().numpy()


def train_multitask(model, trans, geo, tgeo, tg_bridge, steps, batch, lr, device,
                    rep_offset=10.0, eval_every=500):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    A, ACT, B = trans[:, 0].to(device), trans[:, 1].to(device), trans[:, 2].to(device)
    is_move = (A != B) & (ACT < 4)                # true grid moves
    is_cross = (A != B) & (ACT >= 4)              # real torus crossings
    n = trans.shape[0]
    ea, eb = _sample_pairs(3000, seed=123)
    # fixed bridge crossings for the pos-head "switch costs 0" anchor
    bcross = torch.tensor([[gid(BRIDGE_POS, 0), gid(BRIDGE_POS, 1)],
                           [gid(BRIDGE_POS, 1), gid(BRIDGE_POS, 0)]], device=device)
    model.train()
    for s in range(steps):
        idx = torch.randint(0, n, (batch,), device=device)
        a_id, act, b_id = A[idx], ACT[idx], B[idx]
        ap, at = model.pos_torus(a_id); bp, bt = model.pos_torus(b_id)
        tok_a = model.state_tokens(ap, at)
        tok_b = model.state_tokens(bp, bt)
        mv = is_move[idx]
        r_id = torch.randint(0, N_STATES, (batch,), device=device)
        rp, rt = model.pos_torus(r_id)
        tok_r = model.state_tokens(rp, rt)

        # ---- FULL task (position + identity), self_norm head ----
        pred = model.dynamics(tok_a, act)
        loss_dyn = model.distance(pred, tok_b).square().mean()
        mm = (a_id != b_id)
        loss_anc = ((model.distance(tok_a[mm], tok_b[mm]) - 1.0).square().mean()
                    if mm.any() else torch.zeros((), device=device))
        loss_rep = F.softplus(rep_offset - model.distance(tok_a, tok_r)).mean()
        loss_full = loss_dyn + loss_anc + loss_rep

        # same-position-different-torus tokens (identity-only difference)
        same_p = torch.randint(0, N_POS, (batch,), device=device)
        tok_p0 = model.state_tokens(same_p, torch.zeros_like(same_p))
        tok_p1 = model.state_tokens(same_p, torch.ones_like(same_p))

        # ---- POSITION-ONLY task (identity irrelevant): pos head ----
        # move -> 1 ; same position across tori -> 0 ; random -> pushed apart
        loss_pos = F.softplus(rep_offset - model.dist_pos(tok_a, tok_r)).mean()
        if mv.any():
            loss_pos = loss_pos + (model.dist_pos(tok_a[mv], tok_b[mv]) - 1.0).square().mean()
        loss_pos = loss_pos + model.dist_pos(tok_p0, tok_p1).square().mean()   # identity irrelevant

        # ---- IDENTITY-ONLY task (position irrelevant): id head ----
        # move -> 0 (same torus) ; same position across tori -> 1 ; random by torus label
        loss_id = model.dist_id(tok_p0, tok_p1).sub(1.0).square().mean()       # diff torus -> 1
        if mv.any():
            loss_id = loss_id + model.dist_id(tok_a[mv], tok_b[mv]).square().mean()  # same torus -> 0
        same_torus = (at == rt).float()
        loss_id = loss_id + (model.dist_id(tok_a, tok_r) - (1.0 - same_torus)).square().mean()

        loss = loss_full + loss_pos + loss_id
        opt.zero_grad(); loss.backward(); opt.step()

        if (s + 1) % eval_every == 0 or s == 0:
            sp_full = spearman_geo(model, geo, ea, eb, device)
            det, _ = detour_signature(model, tg_bridge, device)
            id_s, pos_s = disentangle_stats(model, device)
            print(f"[mt] step {s+1}/{steps} full {loss_full.item():.2f} pos {loss_pos.item():.2f} "
                  f"| full_sp {sp_full:.3f} detour {det:.3f} "
                  f"| id_sens {id_s.round(2)} pos_sens {pos_s.round(2)}")
    return model


@torch.no_grad()
def eval_id_head(model, device, n_pairs=6000, seed=8):
    a, b = _sample_pairs(n_pairs, seed)
    ta = model.state_tokens(torch.tensor(a % N_POS, device=device), torch.tensor(a // N_POS, device=device))
    tb = model.state_tokens(torch.tensor(b % N_POS, device=device), torch.tensor(b // N_POS, device=device))
    d = model.dist_id(ta, tb).cpu().numpy()
    same = (a // N_POS) == (b // N_POS)
    return float(d[same].mean()), float(d[~same].mean())


@torch.no_grad()
def eval_pos_head(model, tgeo, device, n_pairs=6000, seed=7):
    a, b = _sample_pairs(n_pairs, seed)
    pa, pb = a % N_POS, b % N_POS
    tg = tgeo[pa, pb]                                # position-only geodesic (torus ignored)
    ta = model.state_tokens(torch.tensor(pa, device=device), torch.tensor(a // N_POS, device=device))
    tb = model.state_tokens(torch.tensor(pb, device=device), torch.tensor(b // N_POS, device=device))
    d = model.dist_pos(ta, tb).cpu().numpy()
    return spearmanr(d, tg).statistic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
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
    print(f"device={device} MULTITASK (full + position-only), no disentangle loss, seed={args.seed}")

    G = build_graph()
    trans = build_transitions()
    geo = all_pairs_geodesic(G)
    tgeo = torus_geo_matrix()
    tg_bridge = torus_geo_to_bridge(G)

    model = MultiTaskModel().to(device)
    train_multitask(model, trans, geo, tgeo, tg_bridge, args.steps, args.batch, args.lr, device,
                    eval_every=args.eval_every)

    full_sp = spearman_geo(model, geo, *_sample_pairs(6000, seed=7), device)
    det, _ = detour_signature(model, tg_bridge, device)
    pos_sp = eval_pos_head(model, tgeo, device)
    id_same, id_diff = eval_id_head(model, device)
    id_s, pos_s = disentangle_stats(model, device)
    st = attention_stats(model, device)

    print("\n================ MULTITASK RESULTS ================")
    print(f"full-head  Spearman(D, full geodesic)     = {full_sp:.3f}   detour = {det:.3f}")
    print(f"pos-head   Spearman(D, position geodesic) = {pos_sp:.3f}")
    print(f"id-head    mean dist  same-torus={id_same:.3f}  diff-torus={id_diff:.3f}  (want 0 vs 1)")
    print(f"token identity-sensitivity [q0,q1] = {id_s.round(3)}")
    print(f"token position-sensitivity [q0,q1] = {pos_s.round(3)}")
    print(f"attention argmax-on-agent  [q0,q1] = {st['pos_argmax_acc'].round(3)}")
    print(f"attention argmax-on-marker [q0,q1] = {st['marker_argmax_acc'].round(3)}")
    print("===================================================")
    save_attention_maps(model, device, os.path.join(OUT, "image_attention_maps_multitask.png"))

    if args.json_out:
        res = dict(full_sp=float(full_sp), det=float(det), pos_sp=float(pos_sp),
                   id_same=id_same, id_diff=id_diff,
                   id_sens=id_s.tolist(), pos_sens=pos_s.tolist(),
                   agent_argmax=st['pos_argmax_acc'].tolist(),
                   marker_argmax=st['marker_argmax_acc'].tolist())
        with open(args.json_out, "w") as f:
            json.dump(res, f, indent=2)
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()

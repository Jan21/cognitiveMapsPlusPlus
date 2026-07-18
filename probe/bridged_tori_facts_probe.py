"""
FACTS-lite: recurrent slot factoring from TEMPORAL consistency (no disentangle loss).

Inspired by FACTS (Li et al., ICLR 2025): a small memory of k slots is carried
recurrently across a trajectory; each step the slots cross-attend the observation
(routing) and update (gated / element-wise). Training objective is next-observation
prediction ONLY -- no invariance, sparsity, or independence penalty.

Bet (the transferable trick): a slot carried across a random walk naturally becomes the
thing that is STABLE over time. Identity is constant as you walk (only the rare bridge
crossing changes it); position updates every move. So identity should collect in one slot
and position in another -- the same signal `invar` used, but obtained for free from time.

Env: 2-factor bridged tori (position + id), image observations (15x15, values {0,1,2}).
Measured: per-slot identity vs position sensitivity (which slot is move-invariant = id).
"""

import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from bridged_tori_probe import SIDE, N_POS, N_STATES, BRIDGE_POS, gid
from bridged_tori_image_probe import render_batch, OUT


def step_vec(pos, torus, a):
    """vectorized environment step. pos,torus,a: (N,) int arrays -> next pos,torus."""
    npos = pos.copy(); ntor = torus.copy()
    for act, (dr, dc) in [(0, (-1, 0)), (1, (1, 0)), (2, (0, -1)), (3, (0, 1))]:
        m = a == act
        r = (pos[m] // SIDE + dr) % SIDE
        c = (pos[m] % SIDE + dc) % SIDE
        npos[m] = r * SIDE + c
    ntor[(a == 4) & (pos == BRIDGE_POS) & (torus == 0)] = 1     # next
    ntor[(a == 5) & (pos == BRIDGE_POS) & (torus == 1)] = 0     # prev
    return npos, ntor


def generate_walks(n, T, seed):
    rng = np.random.default_rng(seed)
    states = np.zeros((n, T), dtype=np.int64)
    actions = np.zeros((n, T - 1), dtype=np.int64)
    pos = rng.integers(0, N_POS, n); torus = rng.integers(0, 2, n)
    for t in range(T):
        states[:, t] = torus * N_POS + pos
        if t < T - 1:
            a = rng.integers(0, 6, n)
            actions[:, t] = a
            pos, torus = step_vec(pos, torus, a)
    return torch.tensor(states), torch.tensor(actions)


class FactsLite(nn.Module):
    def __init__(self, k=2, d_model=32, n_heads=4):
        super().__init__()
        self.k = k
        self.d = d_model
        self.pixel_val_emb = nn.Embedding(3, d_model)          # values {0,1,2}
        self.pixel_pos_emb = nn.Embedding(N_POS, d_model)
        self.slots_init = nn.Parameter(torch.randn(k, d_model) * 0.02)
        # slot-attention routing: inputs COMPETE over slots (softmax over slots) -> specialization
        self.to_q = nn.Linear(d_model, d_model)
        self.to_k = nn.Linear(d_model, d_model)
        self.to_v = nn.Linear(d_model, d_model)
        self.norm_slots = nn.LayerNorm(d_model)
        self.gru = nn.GRUCell(d_model, d_model)                # shared across slots
        self.act_emb = nn.Embedding(6, d_model)
        self.dyn = nn.Sequential(nn.Linear(k * d_model + d_model, 128), nn.GELU(),
                                 nn.Linear(128, k * d_model))
        self.decoder = nn.Sequential(nn.Linear(k * d_model, 256), nn.GELU(),
                                     nn.Linear(256, N_POS * 3))

    def pixels(self, states):
        pos, torus = states % N_POS, states // N_POS
        grid = render_batch(pos, torus, states.device)         # (B,225) in {0,1,2}
        return self.pixel_val_emb(grid) + self.pixel_pos_emb(
            torch.arange(N_POS, device=states.device)).unsqueeze(0), grid

    def update(self, Z, states):
        """one recurrent step: slot-attention routing (inputs compete over slots), gated update."""
        B = states.shape[0]
        X, grid = self.pixels(states)
        q = self.to_q(self.norm_slots(Z))                      # (B,k,d)
        kk = self.to_k(X); v = self.to_v(X)                    # (B,N,d)
        dots = torch.einsum("bkd,bnd->bkn", q, kk) * (self.d ** -0.5)
        attn = dots.softmax(dim=1)                             # compete OVER SLOTS (dim=1)
        attn = attn / (attn.sum(dim=2, keepdim=True) + 1e-8)   # weighted mean over pixels
        R = torch.einsum("bkn,bnd->bkd", attn, v)              # (B,k,d) routed updates
        Znew = self.gru(R.reshape(B * self.k, self.d), Z.reshape(B * self.k, self.d))
        return Znew.view(B, self.k, self.d), grid

    def decode(self, Z):
        B = Z.shape[0]
        return self.decoder(Z.reshape(B, -1)).view(B, N_POS, 3)

    def predict(self, Z, action):
        B = Z.shape[0]
        inp = torch.cat([Z.reshape(B, -1), self.act_emb(action)], dim=-1)
        return Z + self.dyn(inp).view(B, self.k, self.d)

    def rollout_loss(self, states, actions):
        # upweight informative pixels (agent=1, marker=2) so background doesn't dominate
        w = torch.tensor([1.0, 50.0, 50.0], device=states.device)
        B, T = states.shape
        Z = self.slots_init.unsqueeze(0).expand(B, -1, -1).contiguous()
        recon = 0.0; pred = 0.0
        for t in range(T):
            Z, grid = self.update(Z, states[:, t])
            recon = recon + F.cross_entropy(self.decode(Z).reshape(-1, 3), grid.reshape(-1), weight=w)
            if t < T - 1:
                predZ = self.predict(Z, actions[:, t])
                _, gnext = self.pixels(states[:, t + 1])
                pred = pred + F.cross_entropy(self.decode(predZ).reshape(-1, 3), gnext.reshape(-1), weight=w)
        return recon / T + pred / (T - 1)

    @torch.no_grad()
    def slot_of(self, states, iters=8):
        """settled slot encoding of a static observation (iterate routing to convergence)."""
        B = states.shape[0]
        Z = self.slots_init.unsqueeze(0).expand(B, -1, -1).contiguous()
        for _ in range(iters):
            Z, _ = self.update(Z, states)
        return Z


@torch.no_grad()
def rollout_slot_stats(model, device, T=20, n=256, seed=5):
    """Read slots AS USED during rollouts; per-slot modulation depth by id vs position."""
    rng = np.random.default_rng(seed)
    pos = rng.integers(0, N_POS, n); torus = rng.integers(0, 2, n)
    Z = model.slots_init.unsqueeze(0).expand(n, -1, -1).contiguous()
    Zs, P, TT = [], [], []
    for t in range(T):
        states = torch.tensor(torus * N_POS + pos, device=device)
        Z, _ = model.update(Z, states)
        if t >= T // 2:                                # after warmup, collect
            Zs.append(Z.cpu().numpy()); P.append(pos.copy()); TT.append(torus.copy())
        a = rng.integers(0, 6, n)
        pos, torus = step_vec(pos, torus, a)
    Z = np.concatenate(Zs, 0); P = np.concatenate(P); TT = np.concatenate(TT)
    k = Z.shape[1]
    id_mod = np.zeros(k); pos_mod = np.zeros(k)
    for j in range(k):
        zj = Z[:, j, :]
        std = zj.std(0).mean() + 1e-6
        id_mod[j] = np.abs(zj[TT == 0].mean(0) - zj[TT == 1].mean(0)).mean() / std
        pm = np.stack([zj[P == p].mean(0) if (P == p).any() else zj.mean(0) for p in range(N_POS)])
        pos_mod[j] = (pm.max(0) - pm.min(0)).mean() / std
    return id_mod, pos_mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2500)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--T", type=int, default=12)
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--n_walks", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--json_out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(OUT, exist_ok=True)
    print(f"device={device} FACTS-lite k={args.k} T={args.T} (temporal factoring, no disentangle loss)")

    states, actions = generate_walks(args.n_walks, args.T, seed=args.seed)
    states, actions = states.to(device), actions.to(device)
    model = FactsLite(k=args.k).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    model.train()
    for s in range(args.steps):
        idx = torch.randint(0, args.n_walks, (args.batch,), device=device)
        loss = model.rollout_loss(states[idx], actions[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % args.eval_every == 0 or s == 0:
            id_m, pos_m = rollout_slot_stats(model, device)
            print(f"step {s+1}/{args.steps} loss {loss.item():.3f} | "
                  f"id_mod {id_m.round(2)} pos_mod {pos_m.round(2)}")

    id_m, pos_m = rollout_slot_stats(model, device)
    print("\n================ FACTS-LITE RESULTS ================")
    for j in range(args.k):
        role = "identity" if id_m[j] > 2 * pos_m[j] else ("position" if pos_m[j] > 2 * id_m[j] else "mixed")
        print(f"  slot{j}: id_mod={id_m[j]:.2f}  pos_mod={pos_m[j]:.2f}  -> {role}")
    clean = (id_m.argmax() != pos_m.argmax())
    print(f"id-dominant slot = {id_m.argmax()}, position-dominant slot = {pos_m.argmax()}  "
          f"-> {'SPLIT' if clean else 'same slot (entangled)'}")
    print("====================================================")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(dict(id_mod=id_m.tolist(), pos_mod=pos_m.tolist(), clean=bool(clean)), f, indent=2)
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()

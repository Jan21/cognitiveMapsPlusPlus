"""
Gated-knob stratified probe (image input). A knob can be changed ONLY when the agent stands
on its control cell (next to the knob) — you cannot switch movement-mode at will. So for a
state far from all knobs, the local neighbourhood is *genuinely* just the current movement,
which can be truly low-dimensional (0 if a frozen agent isn't near its knob).

Knob VALUE lives in a separate always-visible pixel outside the movable grid — the agent can
never occupy it, so it stays observable. The agent reaches a control cell (a normal grid cell)
to flip its knob; the flip costs one step, same as a move.

Dead states (nobody can move AND nobody is next to a knob) are excluded from sampling; no
transition can create one (a move needs an unfrozen agent; a flip keeps its agent on-control).

We do NOT visualise the whole space. We pick representative states, sample only each one's
local neighbourhood by legal random walks, embed just those, and check the volume-growth
dimension is right.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NAGENTS = 2
G = 15
NPOS = G * G
NKNOB = 4                                   # 0 all, 1 horiz, 2 vert, 3 none
NONE = 3
KDOF = np.array([2, 1, 1, 0])
# control cells (distinct interior cells); agent i flips knob i only when standing here
CTRL = [((2 + 3 * i) % G) * G + ((2 + 3 * i) % G) for i in range(4)][:NAGENTS]
P = NPOS + NAGENTS                          # grid pixels + one always-visible knob pixel per agent
VOCAB = max(NAGENTS + 1, NKNOB)
OUT = os.path.join(os.path.dirname(__file__), "..", "factored_vis")


def allowed_dirs(k):
    if k == 0: return [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if k == 1: return [(0, -1), (0, 1)]     # horizontal
    if k == 2: return [(-1, 0), (1, 0)]     # vertical
    return []                               # none


def is_dead(pos, knob):
    return bool((knob == NONE).all() and not any(pos[i] == CTRL[i] for i in range(NAGENTS)))


def legal_transitions(pos, knob):
    """all one-step legal neighbours of a single state (numpy)."""
    out = []
    for i in range(NAGENTS):
        r, c = pos[i] // G, pos[i] % G
        for dr, dc in allowed_dirs(knob[i]):
            p2 = pos.copy(); p2[i] = ((r + dr) % G) * G + ((c + dc) % G)
            out.append((p2, knob.copy()))
        if pos[i] == CTRL[i]:               # can flip knob i only here
            for kv in range(NKNOB):
                if kv != knob[i]:
                    k2 = knob.copy(); k2[i] = kv
                    out.append((pos.copy(), k2))
    return out


# ---------- image encoder ----------
def render(pos, knob):
    B, dev = pos.shape[0], pos.device
    grid = torch.zeros(B, P, dtype=torch.long, device=dev); ar = torch.arange(B, device=dev)
    for i in range(NAGENTS):
        grid[ar, pos[:, i]] = i + 1
    for i in range(NAGENTS):
        grid[:, NPOS + i] = knob[:, i]       # always-visible knob value pixel
    return grid


class ImageEnc(nn.Module):
    def __init__(self, d_model=64, D=48, n_slots=8, n_heads=4):
        super().__init__()
        self.val_emb = nn.Embedding(VOCAB, d_model)
        self.pos_emb = nn.Embedding(P, d_model)
        self.slots = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(n_slots * d_model, 128), nn.GELU(), nn.Linear(128, D))

    def embed(self, pos, knob):
        grid = render(pos, knob); B = grid.shape[0]
        pix = self.val_emb(grid) + self.pos_emb(torch.arange(P, device=grid.device)).unsqueeze(0)
        z, _ = self.attn(self.slots.unsqueeze(0).expand(B, -1, -1), pix, pix)
        return self.proj(z.reshape(B, -1))


# ---------- training (vectorized) ----------
def rand_valid(n, device):
    pos = torch.randint(0, NPOS, (n, NAGENTS), device=device)
    knob = torch.randint(0, NKNOB, (n, NAGENTS), device=device)
    ctrl_t = torch.tensor(CTRL, device=device)
    for _ in range(6):
        allfrozen = (knob == NONE).all(1)
        atctrl = torch.zeros(n, dtype=torch.bool, device=device)
        for i in range(NAGENTS):
            atctrl |= (pos[:, i] == CTRL[i])
        dead = allfrozen & ~atctrl
        if not dead.any():
            break
        knob[dead, 0] = torch.randint(0, NONE, (int(dead.sum()),), device=device)  # unfreeze agent 0
    return pos, knob


def move_pair(pos, knob, K):
    """isometry target within the same config; dist = flat-torus L2 geodesic."""
    n, dev = pos.shape[0], pos.device
    dsq = torch.zeros(n, device=dev); tgt = pos.clone()
    for i in range(NAGENTS):
        ki = knob[:, i]; r, c = pos[:, i] // G, pos[:, i] % G
        row_ok = ((ki == 0) | (ki == 2)).long(); col_ok = ((ki == 0) | (ki == 1)).long()
        drow = torch.randint(-K, K + 1, (n,), device=dev) * row_ok
        dcol = torch.randint(-K, K + 1, (n,), device=dev) * col_ok
        tgt[:, i] = ((r + drow) % G) * G + ((c + dcol) % G)
        dsq = dsq + drow.float() ** 2 + dcol.float() ** 2
    return tgt, dsq.sqrt()


def knob_edge(pos, knob):
    """put a random agent on its control cell, flip its knob -> a unit-cost gated edge."""
    n, dev = pos.shape[0], pos.device
    ctrl_t = torch.tensor(CTRL, device=dev)
    a = torch.randint(0, NAGENTS, (n,), device=dev); ar = torch.arange(n, device=dev)
    pos2 = pos.clone(); pos2[ar, a] = ctrl_t[a]
    knob_v = knob.clone(); knob_v[ar, a] = (knob[ar, a] + torch.randint(1, NKNOB, (n,), device=dev)) % NKNOB
    return pos2, knob, knob_v


def train(model, steps, batch, lr, device, K=2, rep_offset=10.0, eval_every=2000):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for s in range(steps):
        pos, knob = rand_valid(batch, device)
        eu = model.embed(pos, knob)
        tgt, dist = move_pair(pos, knob, K)                       # isometry (moves)
        loss_iso = (torch.norm(eu - model.embed(tgt, knob), dim=-1) - dist).square().mean()
        p2, k_u, k_v = knob_edge(pos, knob)                       # gated knob flip = distance 1
        loss_knob = (torch.norm(model.embed(p2, k_u) - model.embed(p2, k_v), dim=-1) - 1.0).square().mean()
        rp, rk = rand_valid(batch, device)
        loss_rep = F.softplus(rep_offset - torch.norm(eu - model.embed(rp, rk), dim=-1)).mean()
        loss = loss_iso + loss_knob + loss_rep
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % eval_every == 0 or s == 0:
            print(f"step {s+1}/{steps} loss {loss.item():.3f} (iso {loss_iso.item():.3f} "
                  f"knob {loss_knob.item():.3f} rep {loss_rep.item():.3f})", flush=True)
    return model


# ---------- local-neighbourhood dimension ----------
def vgt_slope(dist, window=3, rmax_frac=0.6, min_count=6, num_radii=30):
    d = np.sort(dist[dist > 1e-9])
    if d.size < 12:
        return 0.0 if d.size == 0 else np.nan
    rmax = d[int(len(d) * rmax_frac)]
    if rmax <= d[0]:
        return np.nan
    radii = np.logspace(np.log10(d[0]), np.log10(rmax), num_radii)
    counts = np.searchsorted(d, radii, side="left").astype(float)
    valid = counts >= min_count
    if valid.sum() < window + 1:
        return np.nan
    lr, lc = np.log(radii[valid]), np.log(counts[valid])
    sl = [np.polyfit(lr[i:i + window], lc[i:i + window], 1)[0] for i in range(len(lr) - window)]
    return float(np.median(sl)) if sl else np.nan


def local_ball(pos0, knob0, R, N, rng):
    """vectorized: N legal random walks in parallel, collecting every step (all radii), deduped.
    respects gating (move only if the knob allows the axis; flip only on the control cell)."""
    ctrl = np.array(CTRL)
    pos = np.tile(pos0.astype(np.int64), (N, 1)); knob = np.tile(knob0.astype(np.int64), (N, 1))
    cp, ck = [pos.copy()], [knob.copy()]
    ar = np.arange(N)
    for _ in range(R):
        a = rng.integers(0, NAGENTS, N)
        ka = knob[ar, a]; pa = pos[ar, a]; r, c = pa // G, pa % G
        do_flip = rng.random(N) < 0.3
        d = rng.integers(0, 4, N)                                # 0 up,1 down (vert), 2 left,3 right (horiz)
        nr = np.where(d == 0, (r - 1) % G, np.where(d == 1, (r + 1) % G, r))
        nc = np.where(d == 2, (c - 1) % G, np.where(d == 3, (c + 1) % G, c))
        move_ok = ((d < 2) & ((ka == 0) | (ka == 2))) | ((d >= 2) & ((ka == 0) | (ka == 1)))
        at_ctrl = pa == ctrl[a]
        new_knob = (ka + rng.integers(1, NKNOB, N)) % NKNOB
        pos = pos.copy(); knob = knob.copy()
        fmask = do_flip & at_ctrl
        mmask = (~do_flip) & move_ok
        knob[ar[fmask], a[fmask]] = new_knob[fmask]
        pos[ar[mmask], a[mmask]] = (nr * G + nc)[mmask]
        cp.append(pos.copy()); ck.append(knob.copy())
    allp = np.concatenate(cp); allk = np.concatenate(ck)
    _, idx = np.unique(np.concatenate([allp, allk], axis=1), axis=0, return_index=True)
    return allp[idx], allk[idx]


@torch.no_grad()
def measure_point(model, device, pos0, knob0, R, N, rng):
    ps, ks = local_ball(pos0, knob0, R, N, rng)
    if len(ps) < 12:
        return 0.0, len(ps)                              # tiny neighbourhood -> ~0-dim
    E = []
    for i in range(0, len(ps), 8192):
        E.append(model.embed(torch.tensor(ps[i:i + 8192], device=device),
                             torch.tensor(ks[i:i + 8192], device=device)).cpu().numpy())
    E = np.concatenate(E)
    q = model.embed(torch.tensor(pos0[None], device=device),
                    torch.tensor(knob0[None], device=device)).cpu().numpy()[0]
    return vgt_slope(np.linalg.norm(E - q, axis=1)), len(ps)


def far_pos(rng, avoid_ctrl=True):
    while True:
        p = int(rng.integers(0, NPOS))
        if not avoid_ctrl or p not in CTRL:
            return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--N", type=int, default=5000)
    ap.add_argument("--reps", type=int, default=6)
    ap.add_argument("--eval_every", type=int, default=4000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device); os.makedirs(OUT, exist_ok=True)
    print(f"device={device} GATED image, {NAGENTS} agents G={G} ctrl={CTRL}", flush=True)

    model = ImageEnc().to(device)
    train(model, args.steps, args.batch, args.lr, device, K=args.K, eval_every=args.eval_every)

    rng = np.random.default_rng(args.seed)
    # representative scenarios: (name, knob, positions-builder, expected note)
    def st(k, at_ctrl):    # at_ctrl: list of agent indices placed on their control cell
        pos = np.array([CTRL[i] if i in at_ctrl else far_pos(rng) for i in range(NAGENTS)])
        return pos, np.array(k)
    scenarios = [
        ("all,all  far",          lambda: st([0, 0], []),  "move 2+2=4, no flip"),
        ("all,none far",          lambda: st([0, 3], []),  "move 2 (a1 stuck), no flip"),
        ("horiz,vert far",        lambda: st([1, 2], []),  "move 1+1=2, no flip"),
        ("none,all  far",         lambda: st([3, 0], []),  "move 2 (a0 stuck)"),
        ("none,none a0@ctrl",     lambda: st([3, 3], [0]), "a0 can flip (bridge), a1 stuck"),
        ("all,all   a0@ctrl",     lambda: st([0, 0], [0]), "move 4 + a0 flip bridge"),
    ]
    print("\n================ GATED — local dimension at representative points ================", flush=True)
    print(f"{'scenario':<24}{'measured':>10}{'ball':>8}   expected")
    for name, build, note in scenarios:
        vals, balls = [], []
        for _ in range(args.reps):
            pos0, knob0 = build()
            d, nb = measure_point(model, device, pos0, knob0, args.R, args.N, rng)
            if not np.isnan(d):
                vals.append(d); balls.append(nb)
        md = np.mean(vals) if vals else float("nan")
        print(f"{name:<24}{md:>10.2f}{int(np.mean(balls)) if balls else 0:>8}   {note}", flush=True)
    print("==================================================================================", flush=True)


if __name__ == "__main__":
    main()

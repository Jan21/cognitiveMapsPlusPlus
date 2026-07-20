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


def move_pair(pos, knob, Kmax):
    """isometry target within the same config; dist = flat-torus L2 geodesic. A per-sample scale
    s in [1, Kmax] is drawn so pairs span all offset magnitudes (not just the max) — this trains
    the metric to be faithful at short range (adjacent states must stay ~1 apart, not collapse to
    0) as well as out toward the measurement radius."""
    n, dev = pos.shape[0], pos.device
    s = torch.randint(1, Kmax + 1, (n,), device=dev).float()
    dsq = torch.zeros(n, device=dev); tgt = pos.clone()
    for i in range(NAGENTS):
        ki = knob[:, i]; r, c = pos[:, i] // G, pos[:, i] % G
        row_ok = ((ki == 0) | (ki == 2)).float(); col_ok = ((ki == 0) | (ki == 1)).float()
        drow = (torch.round((torch.rand(n, device=dev) * 2 - 1) * s) * row_ok).long()
        dcol = (torch.round((torch.rand(n, device=dev) * 2 - 1) * s) * col_ok).long()
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
def vgt_slope(dist, lo_frac=0.05, hi_frac=0.5, min_lo=8):
    """Correlation-dimension slope: log(rank k) vs log(k-th sorted distance), fit over the band
    [lo_frac, hi_frac] of the cumulative curve. Evaluating growth AT the observed distances (not
    at log-spaced bins) means there are no empty bins to drop when the embedding squeezes points
    into a narrow band, and starting at lo_frac skips any collapsed near-neighbours at the bottom.
    The hi_frac cap keeps the fit below saturation (the flat tail of a fully-enclosed ball)."""
    d = np.sort(dist[dist > 1e-9]); N = d.size
    if N < 20:
        return np.nan
    lo = max(min_lo, int(lo_frac * N)); hi = int(hi_frac * N)
    if hi - lo < 6:
        return np.nan
    ld, lk = np.log(d[lo:hi]), np.log(np.arange(1, N + 1, dtype=float)[lo:hi])
    if ld[-1] - ld[0] < 1e-6:
        return np.nan
    return float(np.polyfit(ld, lk, 1)[0])


def local_ball(pos0, knob0, R, cap=6000):
    """Exact legal ball to radius R hops (BFS), capped. Dense and complete for low-dim
    neighbourhoods (where a random walk under-samples the one free agent), so the count-vs-radius
    curve has a clean power-law region. Respects gating via legal_transitions()."""
    start = (tuple(int(x) for x in pos0), tuple(int(x) for x in knob0))
    seen = {start}; frontier = [start]; order = [start]
    for _ in range(R):
        nxt = []
        for pos, knob in frontier:
            for p2, k2 in legal_transitions(np.array(pos), np.array(knob)):
                s = (tuple(int(x) for x in p2), tuple(int(x) for x in k2))
                if s not in seen:
                    seen.add(s); nxt.append(s); order.append(s)
                    if len(seen) >= cap:
                        nxt = []; break
            if not nxt and len(seen) >= cap:
                break
        frontier = nxt
        if not frontier:
            break
    ps = np.array([o[0] for o in order], dtype=np.int64)
    ks = np.array([o[1] for o in order], dtype=np.int64)
    return ps, ks


@torch.no_grad()
def measure_point(model, device, pos0, knob0, R, cap=6000, dbg=False):
    ps, ks = local_ball(pos0, knob0, R, cap)
    nconf = len(set(map(tuple, ks)))
    if len(ps) < 20:
        return np.nan, len(ps), nconf                    # neighbourhood too small to fit
    E = []
    for i in range(0, len(ps), 8192):
        E.append(model.embed(torch.tensor(ps[i:i + 8192], device=device),
                             torch.tensor(ks[i:i + 8192], device=device)).cpu().numpy())
    E = np.concatenate(E)
    q = model.embed(torch.tensor(pos0[None], device=device),
                    torch.tensor(knob0[None], device=device)).cpu().numpy()[0]
    dist = np.linalg.norm(E - q, axis=1)
    if dbg:
        d = np.sort(dist[dist > 1e-9])
        radii = np.logspace(np.log10(d[0]), np.log10(d[-1]), 40)
        counts = np.searchsorted(d, radii, side="right").astype(float)
        nmask = int(((counts >= 8) & (counts <= 0.5 * d.size)).sum())
        print(f"      [dbg] N={d.size} d0={d[0]:.3f} d25={np.percentile(d,25):.3f} "
              f"dmed={np.median(d):.3f} dmax={d[-1]:.3f} fit_bins={nmask}", flush=True)
    return vgt_slope(dist), len(ps), nconf


def tor1(a, b):
    d = abs(a - b); return min(d, G - d)


def far_from_own_ctrl(i, rng, R):
    """random cell whose toroidal distance from agent i's own control cell is > R,
    so an R-ball there cannot reach the control cell -> stays a single movement config."""
    cr, cc = CTRL[i] // G, CTRL[i] % G
    while True:
        p = int(rng.integers(0, NPOS))
        if p in CTRL:
            continue
        if tor1(p // G, cr) + tor1(p % G, cc) > R:
            return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--K", type=int, default=7)     # max isometry scale (per-sample draws 1..K)
    ap.add_argument("--R", type=int, default=7)     # measurement-ball radius (hops)
    ap.add_argument("--cap", type=int, default=6000)
    ap.add_argument("--reps", type=int, default=4)
    ap.add_argument("--eval_every", type=int, default=4000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device); os.makedirs(OUT, exist_ok=True)
    print(f"device={device} GATED image, {NAGENTS} agents G={G} ctrl={CTRL} R={args.R}", flush=True)

    model = ImageEnc().to(device)
    train(model, args.steps, args.batch, args.lr, device, K=args.K, eval_every=args.eval_every)

    rng = np.random.default_rng(args.seed)
    # at_ctrl: agents pinned to their control cell (a stratum junction); others placed far from
    # their own control cell so the ball is a single, genuinely low-dim movement patch.
    def st(k, at_ctrl):
        pos = np.array([CTRL[i] if i in at_ctrl else far_from_own_ctrl(i, rng, args.R)
                        for i in range(NAGENTS)])
        return pos, np.array(k)
    scenarios = [
        ("all,all  far",       lambda: st([0, 0], []),  "both move 2D -> 4"),
        ("all,none far",       lambda: st([0, 3], []),  "a1 frozen -> 2 (a0 only)"),
        ("horiz,vert far",     lambda: st([1, 2], []),  "1D+1D -> 2"),
        ("none,all  far",      lambda: st([3, 0], []),  "a0 frozen -> 2 (a1 only)"),
        ("none,none a0@ctrl",  lambda: st([3, 3], [0]), "junction: a0 flips modes -> ~1.8"),
        ("all,all   a0@ctrl",  lambda: st([0, 0], [0]), "4D bulk + flip bridges -> ~4"),
    ]
    print("\n================ GATED — local dimension at representative points ================", flush=True)
    print(f"{'scenario':<22}{'measured':>10}{'ball':>7}{'cfg':>5}   expected")
    for name, build, note in scenarios:
        vals, balls, cfgs = [], [], []
        for r in range(args.reps):
            pos0, knob0 = build()
            d, nb, nc = measure_point(model, device, pos0, knob0, args.R, args.cap, dbg=(r == 0))
            balls.append(nb); cfgs.append(nc)
            if not np.isnan(d):
                vals.append(d)
        md = np.mean(vals) if vals else float("nan")
        sd = np.std(vals) if len(vals) > 1 else 0.0
        print(f"{name:<22}{md:>7.2f}±{sd:.2f}{int(np.mean(balls)):>7}{int(np.mean(cfgs)):>5}   {note}",
              flush=True)
    print("==================================================================================", flush=True)


if __name__ == "__main__":
    main()

"""Minimal reproduction of the 4-agent emb-NN failure + fix. TWO agents on a 12x12 torus, knob per
agent in {all=2D, horiz=1D, vert=1D, none=0D}. A DOF-2 state = agent0 'all' (movable, 2D) + agent1
'none' (frozen, 2 frozen axes) -> enough frozen axes to trigger the sampling-density problem.

We measure the DOF-2 state's local dimension two ways and compare baseline vs collapse:
  (a) LEGAL ball    - enumerate only legal neighbours (agent0 moves, agent1 fixed): the true stratum.
  (b) jitter-pool emb-NN - graph-free pool jittering BOTH agents, k nearest in embedding.
If (a)=2 for both but (b) is inflated for baseline and =2 for collapse, the culprit is the
measurement (density), and collapse is the fix.
"""
import argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

G = 12; NA = 2; NK = 4; NPOS = G * G; P = NPOS + NA; VOCAB = max(NA + 1, NK)
def axes(k): return (k == 0) | (k == 2), (k == 0) | (k == 1)   # row_ok, col_ok

def render(pos, knob):
    B, dev = pos.shape[0], pos.device
    grid = torch.zeros(B, P, dtype=torch.long, device=dev); ar = torch.arange(B, device=dev)
    for i in range(NA): grid[ar, pos[:, i]] = i + 1
    for i in range(NA): grid[:, NPOS + i] = knob[:, i]
    return grid

class Enc(nn.Module):
    def __init__(self, d=48, D=32, slots=8):
        super().__init__()
        self.val = nn.Embedding(VOCAB, d); self.pe = nn.Embedding(P, d)
        self.slots = nn.Parameter(torch.randn(slots, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(slots * d, 128), nn.GELU(), nn.Linear(128, D))
    def embed(self, pos, knob):
        grid = render(pos, knob); B = grid.shape[0]
        pix = self.val(grid) + self.pe(torch.arange(P, device=grid.device))[None]
        z, _ = self.attn(self.slots[None].expand(B, -1, -1), pix, pix)
        return self.proj(z.reshape(B, -1))

def rand_state(n, dev):
    return torch.randint(0, NPOS, (n, NA), device=dev), torch.randint(0, NK, (n, NA), device=dev)

def move_pair(pos, knob, K):
    n, dev = pos.shape[0], pos.device; tgt = pos.clone(); dsq = torch.zeros(n, device=dev)
    for i in range(NA):
        r, c = pos[:, i] // G, pos[:, i] % G; ro, co = axes(knob[:, i])
        dr = torch.randint(-K, K + 1, (n,), device=dev) * ro.long()
        dc = torch.randint(-K, K + 1, (n,), device=dev) * co.long()
        tgt[:, i] = ((r + dr) % G) * G + ((c + dc) % G); dsq += dr.float() ** 2 + dc.float() ** 2
    return tgt, dsq.sqrt()

def illegal_move(pos, knob):
    n, dev = pos.shape[0], pos.device; ar = torch.arange(n, device=dev)
    a = torch.randint(0, NA, (n,), device=dev); ka = knob[ar, a]; pa = pos[ar, a]; r, c = pa // G, pa % G
    ro, co = axes(ka); mask = (~ro | ~co).float()
    nr = torch.where(~ro, (r + 1) % G, r); nc = torch.where(ro & ~co, (c + 1) % G, c)
    pos2 = pos.clone(); pos2[ar, a] = nr * G + nc
    return pos2, mask

def train(model, steps, dev, lam_col=0.0):
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        pos, knob = rand_state(256, dev); eu = model.embed(pos, knob)
        tgt, dist = move_pair(pos, knob, 2)
        loss = (torch.norm(eu - model.embed(tgt, knob), dim=-1) - dist).square().mean()
        rp, rk = rand_state(256, dev)
        loss = loss + F.softplus(10.0 - torch.norm(eu - model.embed(rp, rk), dim=-1)).mean()
        pim, m = illegal_move(pos, knob)
        d_im = torch.norm(eu - model.embed(pim, knob), dim=-1)
        if lam_col > 0: loss = loss + lam_col * (d_im.square() * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
    return model

def vgt(dist, lo=0.05, hi=0.6, mlo=8):
    d = np.sort(dist[dist > 1e-9]); N = d.size
    if N < 20: return np.nan
    a, b = max(mlo, int(lo * N)), int(hi * N)
    if b - a < 5 or d[b - 1] - d[a] < 1e-6: return np.nan
    return float(np.polyfit(np.log(d[a:b]), np.log(np.arange(1, N + 1, dtype=float)[a:b]), 1)[0])

@torch.no_grad()
def legal_ball(model, dev, pos0, R):   # agent0 'all' moves 2D, agent1 frozen
    r0, c0 = pos0[0] // G, pos0[0] % G; pts = []
    for dr in range(-R, R + 1):
        for dc in range(-R, R + 1):
            if dr * dr + dc * dc <= R * R:
                pts.append([((r0 + dr) % G) * G + ((c0 + dc) % G), pos0[1]])
    pos = torch.tensor(pts, device=dev); knob = torch.tensor([[0, 3]] * len(pts), device=dev)
    E = model.embed(pos, knob).cpu().numpy()
    q = model.embed(torch.tensor([pos0], device=dev), torch.tensor([[0, 3]], device=dev)).cpu().numpy()[0]
    return vgt(np.linalg.norm(E - q, axis=1))

@torch.no_grad()
def jitter_dim(model, dev, pos0, W, M, k, rng):
    pos = np.tile(np.array(pos0), (M, 1))
    for i in range(NA):
        dr = rng.integers(-W, W + 1, M); dc = rng.integers(-W, W + 1, M)
        pos[:, i] = ((pos[:, i] // G + dr) % G) * G + ((pos[:, i] % G + dc) % G)
    knob = np.tile(np.array([0, 3]), (M, 1))
    E = model.embed(torch.tensor(pos, device=dev), torch.tensor(knob, device=dev)).cpu().numpy()
    q = model.embed(torch.tensor([pos0], device=dev), torch.tensor([[0, 3]], device=dev)).cpu().numpy()[0]
    dist = np.linalg.norm(E - q, axis=1)
    return vgt(np.partition(dist, k)[:k])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=4000); args = ap.parse_args()
    dev = torch.device("cpu"); rng = np.random.default_rng(0)
    pos0 = [3 * G + 3, 9 * G + 9]                      # a0 far from a1
    print("DOF-2 state: agent0 'all' (2D), agent1 'none' (frozen). true local dim = 2")
    print(f"{'variant':<14}{'legal-ball':>12}{'jitter-embNN':>14}   d_illegal(a1 move)")
    for tag, lam in [("baseline", 0.0), ("collapse", 1.0)]:
        torch.manual_seed(0); np.random.seed(0)
        model = train(Enc(), args.steps, dev, lam_col=lam)
        lb = legal_ball(model, dev, pos0, R=6)
        jd = jitter_dim(model, dev, pos0, W=6, M=40000, k=2000, rng=rng)
        # d_illegal: move frozen agent1 one cell
        with torch.no_grad():
            q = model.embed(torch.tensor([pos0], device=dev), torch.tensor([[0, 3]], device=dev))
            p2 = [pos0[0], ((pos0[1] // G + 1) % G) * G + pos0[1] % G]
            e2 = model.embed(torch.tensor([p2], device=dev), torch.tensor([[0, 3]], device=dev))
            dill = float(torch.norm(q - e2))
        print(f"{tag:<14}{lb:>12.2f}{jd:>14.2f}   {dill:.2f}")

if __name__ == "__main__":
    main()

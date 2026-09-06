"""Maze distance probe: evolve start/goal token embeddings by a recurrent transformer
over a maze layout; distance between the final embeddings must equal the BFS shortest
path. Validation on UNSEEN layouts (map split).

Inputs per example: maze layout (G x G walls), start cell id, goal cell id.
Architectures (--arch):
  frozen    maze tokens = wall-bit + position embeddings, FIXED; only the start/goal
            tokens evolve, cross-attending to maze tokens + each other every iteration
  frozenenc same, but maze tokens are contextualized ONCE by a small self-attention
            encoder, then frozen
  full      all tokens (maze + start + goal) evolve under full self-attention
  fullinteg like full, but distance = integration readout (accumulated motion of the
            start/goal tokens across iterations) instead of final-embedding L2

Readout (frozen/frozenenc/full): D = softplus(scale) * ||z_s - z_g||_2 at the last
iteration. fullinteg: D = softplus(scale) * sum_t mean(||dz_s||, ||dz_g||).

Usage: python maze_probe.py --arch frozen --G 15 --nmaps 3000 --steps 40000
"""
import argparse, collections, json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def carve_maze(G, rng):
    """perfect maze on an odd lattice: rooms at odd coords, DFS backtracker carving."""
    wall = np.ones((G, G), bool)
    rooms = [(r, c) for r in range(1, G, 2) for c in range(1, G, 2)]
    for r, c in rooms:
        wall[r, c] = False
    start = rooms[rng.integers(len(rooms))]
    stack = [start]; seen = {start}
    while stack:
        r, c = stack[-1]
        nbrs = [(r + dr, c + dc) for dr, dc in ((2, 0), (-2, 0), (0, 2), (0, -2))
                if 0 < r + dr < G and 0 < c + dc < G and (r + dr, c + dc) not in seen]
        if not nbrs:
            stack.pop(); continue
        nr, nc = nbrs[rng.integers(len(nbrs))]
        wall[(r + nr) // 2, (c + nc) // 2] = False
        seen.add((nr, nc)); stack.append((nr, nc))
    if rng.random() < 0.5:                      # braid a little: open a few extra walls
        for _ in range(int(rng.integers(1, 4))):
            r, c = rng.integers(1, G - 1), rng.integers(1, G - 1)
            if wall[r, c] and ((not wall[r - 1, c] and not wall[r + 1, c]) or
                               (not wall[r, c - 1] and not wall[r, c + 1])):
                wall[r, c] = False
    return wall


def bfs(wall, src):
    G = wall.shape[0]
    dist = {src: 0}; dq = collections.deque([src])
    while dq:
        r, c = dq.popleft()
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            n = (r + dr, c + dc)
            if 0 <= n[0] < G and 0 <= n[1] < G and not wall[n] and n not in dist:
                dist[n] = dist[(r, c)] + 1; dq.append(n)
    return dist


def build_pool(a, rng, walls, ids, per_map):
    S, T_, D, M = [], [], [], []
    for m in ids:
        wall = walls[m]
        free = [(r, c) for r in range(a.G) for c in range(a.G) if not wall[r, c]]
        for _ in range(per_map // 10):
            src = free[rng.integers(len(free))]
            dist = bfs(wall, src)
            byd = collections.defaultdict(list)
            for cell, d in dist.items():
                if d > 0: byd[d].append(cell)
            ds = list(byd)
            for _ in range(10):
                d = ds[rng.integers(len(ds))]
                tgt = byd[d][rng.integers(len(byd[d]))]
                S.append(src[0] * a.G + src[1]); T_.append(tgt[0] * a.G + tgt[1])
                D.append(d); M.append(m)
    return (np.array(S), np.array(T_), np.array(D, np.float32), np.array(M))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", choices=["frozen", "frozenenc", "full", "fullinteg"], required=True)
    ap.add_argument("--G", type=int, default=15)
    ap.add_argument("--nmaps", type=int, default=3000)
    ap.add_argument("--ntest", type=int, default=300)
    ap.add_argument("--permap", type=int, default=40, help="pairs sampled per map")
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--T", type=int, default=8)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--gradclip", type=float, default=1.0)
    ap.add_argument("--evalevery", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    ncell = a.G * a.G

    walls = [carve_maze(a.G, np.random.default_rng(a.seed * 100003 + m))
             for m in range(a.nmaps + a.ntest)]
    tr_ids = list(range(a.nmaps)); te_ids = list(range(a.nmaps, a.nmaps + a.ntest))
    S1, S2, D, M = build_pool(a, rng, walls, tr_ids, a.permap)
    E1, E2, ED, EM = build_pool(a, np.random.default_rng(a.seed + 99), walls, te_ids, a.permap)
    print(f"pool train={len(S1)} test={len(E1)} dmean={D.mean():.1f} dmax={int(D.max())}", flush=True)
    WALL = torch.as_tensor(np.stack([w.reshape(-1) for w in walls]).astype(np.float32), device=dev)
    S1t, S2t, Dt, Mt = (torch.as_tensor(x, device=dev) for x in (S1, S2, D, M))
    E1t, E2t, EDt, EMt = (torch.as_tensor(x, device=dev) for x in (E1, E2, ED, EM))

    class Block(nn.Module):
        def __init__(s, d, heads):
            super().__init__()
            s.at = nn.MultiheadAttention(d, heads, batch_first=True)
            s.l1 = nn.LayerNorm(d); s.l2 = nn.LayerNorm(d)
            s.mlp = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        def forward(s, q, kv):
            h, _ = s.at(s.l1(q), s.l1(kv), s.l1(kv))
            q = q + h
            return q + s.mlp(s.l2(q))

    class Model(nn.Module):
        def __init__(s):
            super().__init__()
            d = a.d
            s.pos = nn.Embedding(ncell, d)
            s.wemb = nn.Embedding(2, d)                    # wall / floor
            s.role = nn.Embedding(2, d)                    # start / goal token roles
            s.b1 = Block(d, a.heads); s.b2 = Block(d, a.heads)
            if a.arch == "frozenenc":
                s.enc1 = Block(d, a.heads); s.enc2 = Block(d, a.heads)
            s.scale = nn.Parameter(torch.zeros(()))
        def maze_tokens(s, m):
            wt = s.wemb((WALL[m] > 0.5).long())            # (B,ncell,d)
            return wt + s.pos.weight[None]
        def forward(s, x1, x2, m):
            B = x1.shape[0]
            mz = s.maze_tokens(m)
            if a.arch == "frozenenc":
                mz = s.enc2(mz, mz); mz = s.enc1(mz, mz)
            dyn = torch.stack([s.pos(x1) + s.role.weight[0], s.pos(x2) + s.role.weight[1]], 1)
            acc = torch.zeros(B, device=x1.device)
            if a.arch in ("frozen", "frozenenc"):
                for _ in range(a.T):
                    kv = torch.cat([mz, dyn], 1)
                    prev = dyn
                    dyn = s.b2(s.b1(dyn, kv), torch.cat([mz, dyn], 1))
                    acc = acc + (dyn - prev).norm(dim=-1).mean(1)
            else:
                tok = torch.cat([mz, dyn], 1)
                for _ in range(a.T):
                    prev = tok
                    tok = s.b2(s.b1(tok, tok), tok)
                    acc = acc + (tok[:, -2:] - prev[:, -2:]).norm(dim=-1).mean(1)
                dyn = tok[:, -2:]
            if a.arch == "fullinteg":
                return F.softplus(s.scale) * acc
            return F.softplus(s.scale) * (dyn[:, 0] - dyn[:, 1]).norm(dim=-1)

    model = Model().to(dev)
    print(f"{a.arch} params {sum(p.numel() for p in model.parameters())}", flush=True)
    opt = torch.optim.Adam(model.parameters(), a.lr)
    best_c, best_m = float("-inf"), float("inf")
    for step in range(a.steps):
        for gp in opt.param_groups:
            gp["lr"] = a.lr * min(1.0, (step + 1) / a.warmup)
        b = torch.randint(0, len(S1t), (a.bs,), device=dev)
        loss = F.smooth_l1_loss(model(S1t[b], S2t[b], Mt[b]), Dt[b])
        opt.zero_grad(); loss.backward()
        if a.gradclip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), a.gradclip)
        opt.step()
        if step % max(1, a.steps // 8) == 0:
            print(f"step {step} loss {loss.item():.3f}", flush=True)
        if a.evalevery and step > 0 and step % a.evalevery == 0:
            model.eval()
            with torch.no_grad():
                pr = torch.cat([model(E1t[i:i + 2048], E2t[i:i + 2048], EMt[i:i + 2048])
                                for i in range(0, len(E1t), 2048)])
            model.train()
            c = float(np.corrcoef(pr.cpu().numpy(), ED)[0, 1]); mm = float((pr - EDt).abs().mean())
            best_c, best_m = max(best_c, c), min(best_m, mm)
            print(f"step {step} evalcorr {c:.3f} evalmae {mm:.3f}", flush=True)
    model.eval()
    with torch.no_grad():
        pr = torch.cat([model(E1t[i:i + 2048], E2t[i:i + 2048], EMt[i:i + 2048])
                        for i in range(0, len(E1t), 2048)])
    c = float(np.corrcoef(pr.cpu().numpy(), ED)[0, 1]); mm = float((pr - EDt).abs().mean())
    print("RESULT " + json.dumps(dict(tag=a.tag, arch=a.arch, G=a.G, T=a.T, d=a.d,
          nmaps=a.nmaps, steps=a.steps, seed=a.seed, test_corr=round(c, 4),
          test_mae=round(mm, 3), best_corr=round(max(best_c, c), 4),
          best_mae=round(min(best_m, mm), 3))), flush=True)


if __name__ == "__main__":
    main()

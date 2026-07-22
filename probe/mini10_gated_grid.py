"""Scaled-up, more pronounced, and with a GATED key and NO wired knowledge of which factor is the key.

Env (factored, grid agents, no images): N agents on a GxG torus (each a 2-D component) + a key.
  - Agent 0 is ALWAYS free (the mover).
  - Agent j>=1 is free iff key >= j.
  - The key changes (+-1) ONLY when the mover (agent 0) stands on the control cell CTRL.
So at a GENERIC state (mover not on CTRL) you cannot change the key: the local neighbourhood is locked
inside its own stratum, and strata are separated by a real detour (walk the mover to CTRL, flip). DOF
at a generic state = 2*(1 + #free agents among 1..N-1) = 2*(1+min(key,N-1)): a pronounced 2/4/6 ladder.

Model (AttnDistV2): d(x,y) = sum_i w_i * ||agent_i(x)-agent_i(y)|| + w_key*||key(x)-key(y)||.
The per-agent weight w_i is produced by a SHARED gate that reads [the agent's own token, a POOLED
context over ALL components]. The key is NOT designated: it enters only anonymously through the pool,
so the model must DISCOVER that the key gates the agents. (Agents are distinguishable by an id
embedding; that is not the key.)

Training: the original local signal only (legal neighbour -> 1; illegal move or off-control key change
or random -> repelled). Measurement: rollout bank -> nearest-K -> gap-cut -> rank-based growth slope.
"""
import argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from mini7_local_vgt import vgt
from mini8_bank import dist_all, dim_from_bank

N, G, Kk = 3, 8, 3                      # 3 agents on 8x8 torus, key 0..2 ; agent j>=1 free iff key>=j
NP = G * G; CTRL = 0; MARGIN = 20.0     # control cell = index 0 (row 0, col 0)

def dof_of(key): return 2 * (1 + min(int(key), N - 1))       # generic-state DOF

def step(p, dr, dc):
    r, c = p // G, p % G
    return ((r + dr) % G) * G + ((c + dc) % G)

def rand_states(n, rng):
    pos = rng.integers(0, NP, (n, N)); key = rng.integers(0, Kk, (n,))
    return np.concatenate([pos, key[:, None]], 1)

DIRS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

def legal_neighbour(s, rng):
    out = s.copy()
    for b in range(s.shape[0]):
        key = s[b, N]; moves = []
        for j in range(N):
            if j == 0 or key >= j:
                for dr, dc in DIRS: moves.append(("a", j, dr, dc))
        if s[b, 0] == CTRL:
            if key - 1 >= 0: moves.append(("k", -1))
            if key + 1 <= Kk - 1: moves.append(("k", +1))
        m = moves[rng.integers(0, len(moves))]
        if m[0] == "a": out[b, m[1]] = step(s[b, m[1]], m[2], m[3])
        else: out[b, N] = key + m[1]
    return out

def illegal_neighbour(s, rng):
    out = s.copy(); ok = np.zeros(s.shape[0], bool)
    for b in range(s.shape[0]):
        key = s[b, N]; opts = []
        frozen = [j for j in range(1, N) if key < j]
        for j in frozen:
            dr, dc = DIRS[rng.integers(0, 4)]; opts.append(("a", j, dr, dc))
        if s[b, 0] != CTRL:                                   # off-control key change is illegal
            if key - 1 >= 0: opts.append(("k", -1))
            if key + 1 <= Kk - 1: opts.append(("k", +1))
        if opts:
            m = opts[rng.integers(0, len(opts))]; ok[b] = True
            if m[0] == "a": out[b, m[1]] = step(s[b, m[1]], m[2], m[3])
            else: out[b, N] = key + m[1]
    return out, ok

class AttnDistV2(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.row = nn.ModuleList([nn.Embedding(G, d) for _ in range(N)])
        self.col = nn.ModuleList([nn.Embedding(G, d) for _ in range(N)])
        self.aid = nn.Embedding(N, d); self.key = nn.Embedding(Kk, d)
        self.gate = nn.Sequential(nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, 1))   # [agent token, pooled context]
        self.wk = nn.Parameter(torch.zeros(()))
    def _embs(self, s):
        ea = []
        for i in range(N):
            p = s[:, i]; ea.append(self.row[i](p // G) + self.col[i](p % G) + self.aid(torch.full_like(p, i)))
        return ea, self.key(s[:, N])
    def forward(self, x, y):
        ax, kx = self._embs(x); ay, ky = self._embs(y)
        cx, cy = ax + [kx], ay + [ky]                          # N+1 components (key NOT flagged)
        sum_pool = torch.stack([a + b for a, b in zip(cx, cy)], 0).mean(0)
        dif_pool = torch.stack([(a - b).abs() for a, b in zip(cx, cy)], 0).mean(0)
        g = torch.cat([sum_pool, dif_pool], -1)                # pooled context over ALL components
        d = F.softplus(self.wk) * torch.norm(kx - ky, dim=-1)
        for i in range(N):
            ti = torch.cat([ax[i] + ay[i], (ax[i] - ay[i]).abs()], -1)
            wi = F.softplus(self.gate(torch.cat([ti, g], -1))).squeeze(-1)
            d = d + wi * torch.norm(ax[i] - ay[i], dim=-1)
        return d

def train(head, steps, rng, lr=3e-3, bs=192):
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(steps):
        s = rand_states(bs, rng); nb = legal_neighbour(s, rng); il, ok = illegal_neighbour(s, rng); rd = rand_states(bs, rng)
        st, nt, it, rt = (torch.tensor(a) for a in (s, nb, il, rd))
        loss = ((head(st, nt) - 1.0) ** 2).mean()
        loss = loss + F.softplus(MARGIN - head(st, rt)).mean()
        okm = torch.tensor(ok)
        if okm.any(): loss = loss + F.softplus(MARGIN - head(st, it)[okm]).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return head

def collect_bank(n_traj, length, rng):
    s = rand_states(n_traj, rng); frames = []
    for _ in range(length):
        frames.append(s.copy()); s = legal_neighbour(s, rng)
    return np.concatenate(frames, 0)

def jitter_pool(probe, W, M, rng):
    """per-probe LOCAL pool: jitter every agent's grid position +-W (key fixed) -> guaranteed local
    density even in high dimension, unlike a shared global bank."""
    pool = np.tile(probe, (M, 1))
    for i in range(N):
        r = (pool[:, i] // G + rng.integers(-W, W + 1, M)) % G
        c = (pool[:, i] % G + rng.integers(-W, W + 1, M)) % G
        pool[:, i] = r * G + c
    return pool

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(); rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    head = train(AttnDistV2(), args.steps, rng)
    bank = collect_bank(2500, 45, rng)
    print(f"N={N} G={G} Kk={Kk}; DOF = 2*(1+free); bank={len(bank)} rollout states\n")
    rj = np.random.default_rng(args.seed + 1)                                         # separate rng: stable probes
    def clean(a):                                                                     # reject VGT runaways (dim>>2N)
        a = np.array([x if (np.isfinite(x) and 0 < x < 10) else np.nan for x in a]); return a
    print(f"{'':<8}{'DOF true':>9}{'dim[bank]':>11}{'dim[jitter]':>13}{'valid j':>9}")
    for m in (0, 1, 2):
        idx = np.where((bank[:, N] == m) & (bank[:, 0] != CTRL))[0]                    # generic states, key=m
        probes = bank[rj.choice(idx, size=min(10, idx.size), replace=False)]
        db, dj = [], []
        for p in probes:
            db.append(dim_from_bank(head, p, bank)[0])
            dj.append(dim_from_bank(head, p, jitter_pool(p, W=3, M=60000, rng=rj))[0])
        db, dj = clean(db), clean(dj); valid = int(np.isfinite(dj).sum())
        f = lambda a: np.nanmean(a) if np.isfinite(a).any() else np.nan
        print(f"key={m}  {dof_of(m):>8}{f(db):>11.2f}{f(dj):>13.2f}{valid:>7}/{len(probes)}")

    # gate diagnostic: does the model make frozen-agent moves far without being told which is the key?
    print("\n[diag] 1-step distances at a generic probe per key (free agents vs frozen)")
    with torch.no_grad():
        for m in (0, 1, 2):
            idx = np.where((bank[:, N] == m) & (bank[:, 0] != CTRL))[0]
            p = bank[idx[0]]; pt = torch.tensor(p[None]); ds = []
            for j in range(N):
                q = p.copy(); q[j] = step(q[j], 1, 0)
                fr = (j == 0 or m >= j)
                ds.append(f"a{j}:{float(head(pt, torch.tensor(q[None]))[0]):.2f}({'free' if fr else 'froz'})")
            print(f"  key={m}: " + "  ".join(ds))

if __name__ == "__main__":
    main()

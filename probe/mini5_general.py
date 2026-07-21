"""Generalize the attn_sum detour distance to arbitrary GUARDED-GRAPH product spaces.

A state is a tuple (s_0, ..., s_{F-1}). Component f has its own internal graph (any edges). An
internal edge of component f is traversable (cost 1) only if its GUARD holds: a chosen other
component g is in an allowed set of its own nodes. Free components have no guard. The global state
graph is the product with guarded edges; we enumerate it and compute the EXACT all-pairs geodesic
by BFS, then regress it with distance heads.

Geodesic intuition (the 'pieces'): to move component f you pay f's internal move cost PLUS, if f is
guarded, a detour to bring the guard component into an enabling state and (usually) back. So:
    d(x,y) ~ sum_f  move_f(sx_f, sy_f)  +  sum_f gate(f moves) * detour_f(guard states at x, y)
Each move_f / detour_f is a learned scalar; an independent sigmoid gate turns each 'piece' on; the
terms are SUMMED (as in attn_sum) so costs accumulate. Nested guards need more pieces than we give;
the approximation then degrades gracefully, which is acceptable.

Heads:
  mds_l1     free per-state embedding, d = ||Ex-Ey||_1              (representability ceiling)
  nodetour   sum_f move_f only (no detour terms)                    (baseline: shows detours matter)
  attnsumG   the general recipe above                              (ours)
"""
import argparse, collections, itertools, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

# ---------- environment ----------
def path(n): return [(i, i + 1) for i in range(n - 1)]
def cycle(n): return path(n) + [(n - 1, 0)]
def grid(r, c):
    e = []
    for i in range(r):
        for j in range(c):
            u = i * c + j
            if j + 1 < c: e.append((u, i * c + j + 1))
            if i + 1 < r: e.append((u, (i + 1) * c + j))
    return e

class Env:
    def __init__(self, sizes, edges, guards):
        self.sizes = sizes; self.F = len(sizes)
        self.edges = [[(a, b) for (a, b) in ef] + [(b, a) for (a, b) in ef] for ef in edges]  # both dirs
        self.guards = guards                      # guards[f] = (g, frozenset) or None
        self.rad = np.array(sizes); self.N = int(np.prod(sizes))
    def decode(self, i):
        out = []
        for s in reversed(self.sizes): out.append(i % s); i //= s
        return list(reversed(out))
    def encode(self, st):
        i = 0
        for s, v in zip(self.sizes, st): i = i * s + v
        return i
    def neighbours(self, i):
        st = self.decode(i); out = []
        for f in range(self.F):
            gd = self.guards[f]
            if gd is not None and st[gd[0]] not in gd[1]:
                continue                          # guard not satisfied: edges of f frozen
            for a, b in self.edges[f]:
                if st[f] == a:
                    st2 = list(st); st2[f] = b; out.append(self.encode(st2))
        return out
    def geodesic(self):
        INF = 10 ** 9; geo = np.full((self.N, self.N), INF, np.int64)
        for s in range(self.N):
            dq = collections.deque([s]); geo[s, s] = 0
            while dq:
                u = dq.popleft()
                for v in self.neighbours(u):
                    if geo[s, v] == INF: geo[s, v] = geo[s, u] + 1; dq.append(v)
        return geo, INF

# ---------- heads ----------
class MDS(nn.Module):
    def __init__(self, N, D=24): super().__init__(); self.E = nn.Embedding(N, D)
    def forward(self, i, j, env): return torch.norm(self.E(i) - self.E(j), p=1, dim=-1)

class SumHead(nn.Module):
    """nodetour: sum of per-component learned internal distances. attnsumG adds gated detour terms."""
    def __init__(self, env, d=24, detour=True):
        super().__init__(); self.env = env; self.F = env.F; self.detour = detour
        self.emb = nn.ModuleList([nn.Embedding(s, d) for s in env.sizes])
        self.move = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1)) for _ in range(self.F)])
        self.guarded = [f for f in range(self.F) if env.guards[f] is not None]
        self.det = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1)) for _ in self.guarded])
        ntok = self.F + (len(self.guarded) if detour else 0)
        self.key = nn.Parameter(torch.randn(ntok, d) * 0.1)
        self.q = nn.Linear(2 * d * self.F + self.F, d); self.d = d
    def _decode(self, idx):
        st = []; rem = idx.clone()
        for s in reversed(self.env.sizes): st.append(rem % s); rem = rem // s
        return list(reversed(st))
    def _dist(self, i, j):
        sx = self._decode(i); sy = self._decode(j)
        ex = [self.emb[f](sx[f]) for f in range(self.F)]; ey = [self.emb[f](sy[f]) for f in range(self.F)]
        dmag = torch.stack([torch.norm(ex[f] - ey[f], dim=-1) for f in range(self.F)], -1)   # (B,F)
        mval = torch.stack([F.softplus(self.move[f](torch.cat([ex[f], ey[f]], -1))).squeeze(-1)
                            for f in range(self.F)], -1)                                      # (B,F)
        vals = [mval[:, f] for f in range(self.F)]
        if self.detour:
            for t, f in enumerate(self.guarded):
                g = self.env.guards[f][0]
                vals.append(F.softplus(self.det[t](torch.cat([ex[g], ey[g]], -1))).squeeze(-1))
        V = torch.stack(vals, -1)                                                            # (B,ntok)
        ctx = torch.cat(ex + ey + [dmag], -1)
        q = self.q(ctx)[:, None]                                                             # (B,1,d)
        gate = torch.sigmoid((q * self.key[None]).sum(-1) / self.d ** 0.5)                   # (B,ntok)
        return (gate * V).sum(-1)
    def forward(self, i, j, env): return 0.5 * (self._dist(i, j) + self._dist(j, i))

def split(env, rng, frac_test=0.3):
    I, J = np.meshgrid(np.arange(env.N), np.arange(env.N)); I = I.ravel(); J = J.ravel()
    perm = rng.permutation(len(I)); ntest = int(frac_test * len(I))
    te, tr = perm[:ntest], perm[ntest:]
    return (I[tr], J[tr]), (I[te], J[te])

class SumHeadLD(nn.Module):
    """Learned dependency structure: NO guard graph wired in. A detour token for EVERY ordered
    component pair (f,g); the gate must learn which (f,g) dependencies are real (fire) and which are
    fake (stay off). Everything else as SumHead."""
    def __init__(self, env, d=24, rigid=False, tied=False):
        super().__init__(); self.env = env; self.F = env.F; self.rigid = rigid or tied; self.tied = tied
        self.emb = nn.ModuleList([nn.Embedding(s, d) for s in env.sizes])
        self.move = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1)) for _ in range(self.F)])
        self.pairs = [(f, g) for f in range(self.F) for g in range(self.F) if f != g]
        self.det = nn.ModuleList([nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1)) for _ in self.pairs])
        ntok = self.F + len(self.pairs)
        self.gate = nn.Sequential(nn.Linear(2 * d * self.F + self.F, d), nn.GELU(), nn.Linear(d, ntok))
        self.register_buffer("dmask", torch.ones(len(self.pairs)))     # 1=piece active, 0=ablated
        self.register_buffer("det_f", torch.tensor([f for (f, g) in self.pairs]))   # which comp triggers piece
        self.gw = nn.Parameter(torch.ones(len(self.pairs))); self.gb = nn.Parameter(torch.zeros(len(self.pairs)))
    def _decode(self, idx):
        st = []; rem = idx.clone()
        for s in reversed(self.env.sizes): st.append(rem % s); rem = rem // s
        return list(reversed(st))
    def _parts(self, i, j):
        sx = self._decode(i); sy = self._decode(j)
        ex = [self.emb[f](sx[f]) for f in range(self.F)]; ey = [self.emb[f](sy[f]) for f in range(self.F)]
        dmag = torch.stack([torch.norm(ex[f] - ey[f], dim=-1) for f in range(self.F)], -1)
        vals = [F.softplus(self.move[f](torch.cat([ex[f], ey[f]], -1))).squeeze(-1) for f in range(self.F)]
        for t, (f, g) in enumerate(self.pairs):
            vals.append(self.dmask[t] * F.softplus(self.det[t](torch.cat([ex[g], ey[g]], -1))).squeeze(-1))
        V = torch.stack(vals, -1)
        gate = torch.sigmoid(self.gate(torch.cat(ex + ey + [dmag], -1)))
        if self.rigid:                                                 # move-terms always on, ungated:
            gate = torch.cat([torch.ones_like(gate[:, :self.F]), gate[:, self.F:]], -1)  # can't absorb detours
        if self.tied:                                                  # detour gate depends ONLY on its own
            tg = torch.sigmoid(self.gw[None] * dmag[:, self.det_f] + self.gb[None])       # component's motion
            gate = torch.cat([gate[:, :self.F], tg], -1)
        moved = torch.stack([(sx[f] != sy[f]) for f in range(self.F)], -1)
        return V, gate, moved
    def _dist(self, i, j):
        V, gate, _ = self._parts(i, j); return (gate * V).sum(-1)
    def forward(self, i, j, env): return 0.5 * (self._dist(i, j) + self._dist(j, i))
    def reg(self, i, j):
        V, gate, _ = self._parts(i, j)
        c = gate[:, self.F:] * V[:, self.F:]                           # (B, ndetour) contributions
        return torch.sqrt((c ** 2).mean(0) + 1e-9).sum()              # group lasso: prune whole pieces
    @torch.no_grad()
    def dep_matrix(self, idx):
        V, gate, moved = self._parts(torch.tensor(idx[0]), torch.tensor(idx[1]))
        contrib = gate * V                                    # actual cost each piece adds
        M = np.full((self.F, self.F), np.nan)
        for t, (f, g) in enumerate(self.pairs):
            mv = moved[:, f]                                  # pairs where component f actually moves
            if mv.any(): M[f, g] = float(contrib[:, self.F + t][mv].mean())
        return M


def train(head, geo, env, tr, steps, lr=3e-3, l1=0.0):
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    it = torch.tensor(tr[0]); jt = torch.tensor(tr[1]); gt = torch.tensor(geo[tr[0], tr[1]], dtype=torch.float32)
    for _ in range(steps):
        d = head(it, jt, env); loss = ((d - gt) ** 2).mean()
        if l1 > 0 and hasattr(head, "reg"): loss = loss + l1 * head.reg(it, jt)
        opt.zero_grad(); loss.backward(); opt.step()
    return head

@torch.no_grad()
def rmse_on(head, geo, env, idx):
    d = head(torch.tensor(idx[0]), torch.tensor(idx[1]), env).numpy(); t = geo[idx[0], idx[1]].astype(float)
    return float(np.sqrt(((d - t) ** 2).mean())), float(np.corrcoef(d, t)[0, 1])

ENVS = {
    # F=2: cyclic internal component guarded by a linear key (non-linear internal geometry)
    "cycle_key": lambda: Env([6, 5], [cycle(6), path(5)], [(1, frozenset({4})), None]),
    # F=2: 2-D grid internal component guarded by a linear key
    "grid_key":  lambda: Env([9, 5], [grid(3, 3), path(5)], [(1, frozenset({4})), None]),
    # F=3: nested guards -- move A needs B in place, move B needs C in place (deep detours)
    "nested":    lambda: Env([5, 4, 4], [path(5), path(4), path(4)],
                             [(1, frozenset({3})), (2, frozenset({3})), None]),
}

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--env", default="all"); ap.add_argument("--seed", type=int, default=0); args = ap.parse_args()
    names = list(ENVS) if args.env == "all" else [args.env]
    for nm in names:
        env = ENVS[nm](); geo, INF = env.geodesic()
        assert (geo < INF).all(), f"{nm}: disconnected"
        rng = np.random.default_rng(args.seed); tr, te = split(env, rng)
        print(f"\n=== {nm}  (F={env.F}, states={env.N}, max geo={int(geo.max())}, "
              f"train pairs={len(tr[0])}, held-out={len(te[0])}) ===")
        print(f"{'head':<11}{'train':>7}{'TEST':>7}{'testcorr':>9}")
        heads = {"mds_l1": lambda: MDS(env.N),
                 "attnsumG": lambda: SumHead(env, detour=True),        # guard graph WIRED (reference)
                 "learndep": lambda: SumHeadLD(env)}                   # guard graph LEARNED
        trained = {}
        for hn, mk in heads.items():
            torch.manual_seed(args.seed); np.random.seed(args.seed)
            h = train(mk(), geo, env, tr, args.steps); trained[hn] = h
            rtr, _ = rmse_on(h, geo, env, tr); rte, cte = rmse_on(h, geo, env, te)
            print(f"{hn:<11}{rtr:>7.2f}{rte:>7.2f}{cte:>9.3f}")
        M = trained["learndep"].dep_matrix(te)                        # learned detour-gate per (f<-g)
        true = {f: (env.guards[f][0] if env.guards[f] else None) for f in range(env.F)}
        print("learned detour  M[f,g] = mean CONTRIBUTION (gate*value) of 'move f via g' when f moves "
              f"(true guards f<-g: {[(f, true[f]) for f in range(env.F) if true[f] is not None]})")
        print("        " + "".join(f"g={g:<5}" for g in range(env.F)))
        for f in range(env.F):
            row = "".join(f"{M[f,g]:<6.2f}" if not np.isnan(M[f, g]) else f"{'--':<6}" for g in range(env.F))
            star = "".join((" *" if true[f] == g else "  ") for g in range(env.F))
            print(f"  f={f}  {row}   true:{star}")

if __name__ == "__main__":
    main()

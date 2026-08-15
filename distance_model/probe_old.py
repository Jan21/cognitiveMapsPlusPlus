"""path_integration probe harness. n factors (cycles C_m) with CHAIN GUARDS: factor 0 always free;
factor i free iff p_{i-1} in keyset. Legal move = +/-1 on one free factor. Product graph -> BFS geodesic
= regression target (cost). A looped transformer flows start factor-embeddings toward the goal, integrating
per-factor displacement into the cost; we test whether that beats simpler baselines and EXTRAPOLATES to
longer distances than trained.

Models (--model): integrator | mlp | transformer | static
  integrator - looped weight-shared 3-layer (attn+MLP) block; cost = sum_t sum_i ||dz_i(t)||   (the bet)
  mlp        - MLP([emb(s);emb(g)]) -> scalar                                                   (no loop/integr)
  transformer- 3-layer transformer over 2n tokens, readout -> scalar                            (no integr)
  static     - sum_i ||emb(s_i)-emb(g_i)|| * scale                                              (prior static distance)
Flow (--flow, integrator only): attractor | velocity | joint
Splits: train on dist<=D (minus held-out); IN-DIST test = held-out dist<=D; EXTRAP test = dist>D.
"""
import argparse, collections, json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F


def enc(p, m):
    e = 0
    for j in range(len(p) - 1, -1, -1): e = e * m + int(p[j])
    return e
def dec(e, n, m):
    p = []
    for _ in range(n): p.append(e % m); e //= m
    return p

def build_guards(n, mode, rng):
    """guard DAG: parents[i] = factors that must ALL be in keyset for factor i to move. chain = [i-1]."""
    if mode == "dag":
        parents = [[]]
        for i in range(1, n):
            k = int(rng.integers(1, i + 1)); parents.append(sorted(int(x) for x in rng.choice(i, size=min(k, i), replace=False)))
        return parents
    return [[i - 1] for i in range(n)]          # chain (parents[0] unused; factor 0 always free)

def free_i(p, i, parents, keyset, guarded):
    if not guarded or i == 0: return True
    return all(p[par] in keyset for par in parents[i])

def fmoves(p, gt, ms):
    """per-factor graph neighbours: cycle C_ms, path P_ms, or ms x ms grid (node = r*ms+c)."""
    if gt == "grid":
        r, c = p // ms, p % ms; out = []
        if r + 1 < ms: out.append((r + 1) * ms + c)
        if r - 1 >= 0: out.append((r - 1) * ms + c)
        if c + 1 < ms: out.append(r * ms + c + 1)
        if c - 1 >= 0: out.append(r * ms + c - 1)
        return out
    if gt == "path":
        return ([p + 1] if p + 1 < ms else []) + ([p - 1] if p - 1 >= 0 else [])
    return [(p + 1) % ms, (p - 1) % ms]                 # cycle
def radix(gt, ms): return ms * ms if gt == "grid" else ms

def neighbours(e, n, R, gt, ms, parents, keyset, guarded):
    p = dec(e, n, R); out = []
    for i in range(n):
        if not free_i(p, i, parents, keyset, guarded): continue
        for q in fmoves(p[i], gt, ms):
            nb = p.copy(); nb[i] = q; out.append(enc(nb, R))
    return out

def bfs(src, n, R, gt, ms, parents, keyset, guarded):
    dist = {src: 0}; dq = collections.deque([src])
    while dq:
        u = dq.popleft()
        for v in neighbours(u, n, R, gt, ms, parents, keyset, guarded):
            if v not in dist: dist[v] = dist[u] + 1; dq.append(v)
    return dist

def build_pool(n, R, gt, ms, parents, keyset, guarded, n_start, rng, per_start=60):
    """triples (s_pos, g_pos, dist) from single-source BFS at random starts, goals sampled across distances."""
    S, Gg, Dd = [], [], []
    for _ in range(n_start):
        src = int(rng.integers(0, R ** n)); dist = bfs(src, n, R, gt, ms, parents, keyset, guarded)
        items = [(g, dd) for g, dd in dist.items() if dd > 0]
        if not items: continue
        by_d = collections.defaultdict(list)
        for g, dd in items: by_d[dd].append(g)
        picks = []
        for dd, gs in by_d.items():
            k = min(len(gs), max(1, per_start // max(1, len(by_d))))
            picks += [(g, dd) for g in rng.choice(gs, size=k, replace=False)]
        sp = dec(src, n, R)
        for g, dd in picks:
            S.append(sp); Gg.append(dec(g, n, R)); Dd.append(dd)
    return np.array(S), np.array(Gg), np.array(Dd, np.float32)


class Block(nn.Module):
    def __init__(self, d, heads, layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d, heads, d * 2, dropout=0.0,
                                     activation="gelu", batch_first=True, norm_first=True) for _ in range(layers)])
    def forward(self, z):
        for l in self.layers: z = l(z)
        return z

class Integrator(nn.Module):
    def __init__(self, n, m, d=64, heads=4, layers=3, T=6, share=True, flow="attractor", disp="l2", stepcap=0.0):
        super().__init__()
        self.n, self.T, self.flow, self.disp, self.share, self.stepcap = n, T, flow, disp, share, stepcap
        self.recall = False; self.cost = "length"
        self.pos = nn.Embedding(m, d); self.fid = nn.Embedding(n, d); self.role = nn.Embedding(3, d)   # start / goal / recalled-start
        self.input = "index"; self.R = m
        self.icnv = nn.Sequential(nn.Conv2d(1, 16, 3, padding=1), nn.GELU(), nn.Conv2d(16, 32, 3, padding=1), nn.GELU())  # IMAGE encoder
        self.iproj = nn.Linear(32, d)
        nb = 1 if share else T
        self.blocks = nn.ModuleList([Block(d, heads, layers) for _ in range(nb)])
        if flow == "velocity": self.merge = nn.Linear(2 * d, d)
        if flow == "joint": self.readout = nn.Linear(d, 1)
        if flow == "act": self.halt = nn.Linear(2 * d, 1)      # learned halting; sees [state, state-goal] to halt on arrival
        if flow == "fp":                                       # goal-gate: update vanishes as state -> goal (fixed point)
            self.ggate = nn.Linear(2 * d, 1); nn.init.constant_(self.ggate.bias, -2.0)   # start near-stationary (gate~0.12), grow as needed
        if flow == "qsweep":                                   # EqR-style: cost sweeps up, a CORRECTNESS-supervised halt head reads it when right
            self.qhead = nn.Linear(2 * d, 1); nn.init.zeros_(self.qhead.weight); nn.init.constant_(self.qhead.bias, -5.0)   # conservative: start by not halting
        self.scale = nn.Parameter(torch.zeros(()))
        self.steps = None; self.arrive = None; self.eps = 0.05; self.haltmode = "converge"; self.eps_arr = 0.5
    def _emb(self, x, role):
        ids = torch.arange(self.n, device=x.device)
        if self.input == "image":                              # render state -> n x R pixel grid; CNN reads it -> per-factor tokens
            img = F.one_hot(x.long(), self.R).float()[:, None]  # (B,1,n,R): row = factor, marked column = position
            feat = self.icnv(img).mean(-1).transpose(1, 2)      # CNN over the grid, pool positions -> (B,n,32)
            return self.iproj(feat) + self.fid(ids)[None] + self.role(torch.tensor(role, device=x.device))
        return self.pos(x) + self.fid(ids)[None] + self.role(torch.tensor(role, device=x.device))
    def forward(self, s, g):
        n = self.n; zs = self._emb(s, 0); zg = self._emb(g, 1)
        blk = lambda t: self.blocks[0] if self.share else self.blocks[t]
        if self.flow == "act":                                 # adaptive computation time: flow halts itself
            B = s.shape[0]; dev = s.device
            tok = torch.cat([zs, zg], 1)
            halt_sum = torch.zeros(B, device=dev); D = torch.zeros(B, device=dev)
            cost = torch.zeros(B, device=dev); n_steps = torch.zeros(B, device=dev)
            still = torch.ones(B, dtype=torch.bool, device=dev)
            for t in range(self.T):                            # self.T = MAX step budget (raise at test via Ttest)
                z = self.blocks[0](tok); new_zs = z[:, :n]
                step = new_zs - tok[:, :n]
                if self.stepcap > 0:                           # bound per-factor step -> far goals need MORE steps
                    sn = step.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                    step = step * (self.stepcap / sn).clamp(max=1.0); new_zs = tok[:, :n] + step
                dz = step.norm(dim=-1).sum(-1)
                sf = still.float(); D = D + dz * sf
                h = torch.sigmoid(self.halt(torch.cat([new_zs.mean(1), (new_zs - zg).mean(1)], -1)).squeeze(-1))
                over = (halt_sum + h) >= (1.0 - 1e-3)
                p = torch.where(over, 1.0 - halt_sum, h) * sf   # ACT remainder at the halting step
                cost = cost + p * D; n_steps = n_steps + sf
                halt_sum = halt_sum + h * sf
                upd = still.view(B, 1, 1)
                tok = torch.cat([torch.where(upd, new_zs, tok[:, :n]), tok[:, n:]], 1)
                still = still & (~over)
                if not bool(still.any()): break
            self.steps = n_steps; self.arrive = (tok[:, :n] - zg).norm(dim=-1).sum(-1)
            return F.softplus(self.scale) * cost
        if self.flow == "qsweep":                              # EqR-style read-at-halt: cost sweeps up; learned halt reads it when correct
            B = s.shape[0]; dev = s.device; z = zs; tok = torch.cat([zs, zg], 1)
            cost = torch.zeros(B, device=dev); costs = []; qs = []
            for t in range(self.T):
                out = self.blocks[0](tok)[:, :n]
                cost = cost + (out - z).norm(dim=-1).sum(-1)
                z = out; tok = torch.cat([z, zg], 1)
                qs.append(self.qhead(torch.cat([z.mean(1), (z - zg).mean(1)], -1)).squeeze(-1))
                costs.append(F.softplus(self.scale) * cost)
            self.cost_traj = torch.stack(costs, 0); self.q_traj = torch.stack(qs, 0)      # (T,B)
            halt = torch.sigmoid(self.q_traj) > 0.5
            idx = torch.where(halt.any(0), halt.float().argmax(0), torch.full((B,), self.T - 1, device=dev, dtype=torch.long))
            self.steps = idx.float() + 1; self.arrive = (z - zg).norm(dim=-1).sum(-1)
            return self.cost_traj[idx, torch.arange(B, device=dev)]                       # cost read at first halt (or last)
        if self.flow == "fp":                                  # FIXED-POINT halting: goal-gated update + convergence stop
            B = s.shape[0]; dev = s.device; z = zs
            cost = torch.zeros(B, device=dev); nsteps = torch.zeros(B, device=dev)
            still = torch.ones(B, dtype=torch.bool, device=dev)
            for t in range(self.T):
                h = self.blocks[0](torch.cat([z, zg], 1))[:, :n]
                gate = torch.sigmoid(self.ggate(torch.cat([z, z - zg], -1)))   # (B,n,1) -> 0 near goal
                upd = gate * (h - z)
                if self.stepcap > 0:                           # bound per-factor step -> ~1 edge/step (traversal)
                    un = upd.norm(dim=-1, keepdim=True).clamp(min=1e-6); upd = upd * (self.stepcap / un).clamp(max=1.0)
                z_new = z + upd
                dz = (z_new - z).norm(dim=-1).sum(-1)
                sf = still.float(); cost = cost + dz * sf; nsteps = nsteps + sf
                if self.haltmode == "arrive":                  # traversal: halt when ARRIVED at goal (bounded steps -> #steps ~ dist)
                    conv = (z_new - zg).norm(dim=-1).sum(-1) < self.eps_arr
                else:
                    conv = dz < self.eps                       # converge: state stopped moving -> freeze
                z = torch.where(still.view(B, 1, 1), z_new, z)
                still = still & (~conv)
                if not bool(still.any()): break
            self.steps = nsteps; self.arrive = (z - zg).norm(dim=-1).sum(-1)
            return F.softplus(self.scale) * cost
        if self.flow == "velocity":
            z = self.merge(torch.cat([zs, zg], -1)); cost = 0.0
            for t in range(self.T):
                nz = blk(t)(z); v = nz - z; z = nz
                cost = cost + (v.abs().sum(-1) if self.disp == "l1" else v.norm(dim=-1)).sum(-1)
            return F.softplus(self.scale) * cost
        zr = self._emb(s, 2) if self.recall else None              # recalled start tokens (Deep-Thinking recall)
        base = torch.cat([zg, zr], 1) if self.recall else zg       # fixed context re-fed every step
        tok = torch.cat([zs, base], 1); cost = torch.zeros(s.shape[0], device=s.device)
        w = min(getattr(self, "warm", 0) or 0, self.T)             # progressive loss: first w steps under no_grad
        def _step(bt, tok, cost):
            z = blk(bt)(tok); nz = z[:, :n]; dz = nz - tok[:, :n]
            per = dz.abs().sum(-1) if self.disp == "l1" else dz.norm(dim=-1)
            return torch.cat([nz, base], 1), cost + ((per ** 2) if self.cost == "energy" else per).sum(-1)
        if w > 0:
            with torch.no_grad():
                for t in range(w): tok, cost = _step(t, tok, cost)
            tok = tok.detach(); cost = cost.detach()
        for t in range(w, self.T): tok, cost = _step(t, tok, cost)
        self.arrive = (tok[:, :n] - zg).norm(dim=-1).sum(-1)       # how close the flow parked to the goal
        if self.flow == "joint":
            return F.softplus(self.readout(tok[:, :n].mean(1)).squeeze(-1)) + 0.0 * cost
        return F.softplus(self.scale) * cost                       # attractor

class MLPBaseline(nn.Module):
    def __init__(self, n, m, d=64, **kw):
        super().__init__()
        self.pos = nn.Embedding(m, d); self.fid = nn.Embedding(n, d); self.n = n
        self.net = nn.Sequential(nn.Linear(2 * n * d, 4 * d), nn.GELU(), nn.Linear(4 * d, 4 * d), nn.GELU(), nn.Linear(4 * d, 1))
    def _e(self, x):
        ids = torch.arange(self.n, device=x.device); return (self.pos(x) + self.fid(ids)[None]).flatten(1)
    def forward(self, s, g): return F.softplus(self.net(torch.cat([self._e(s), self._e(g)], -1)).squeeze(-1))

class TransformerBaseline(nn.Module):
    def __init__(self, n, m, d=64, heads=4, layers=3, **kw):
        super().__init__()
        self.pos = nn.Embedding(m, d); self.fid = nn.Embedding(n, d); self.role = nn.Embedding(2, d); self.n = n
        self.block = Block(d, heads, layers); self.readout = nn.Linear(d, 1)
    def _emb(self, x, r):
        ids = torch.arange(self.n, device=x.device); return self.pos(x) + self.fid(ids)[None] + self.role(torch.tensor(r, device=x.device))
    def forward(self, s, g):
        tok = torch.cat([self._emb(s, 0), self._emb(g, 1)], 1)
        return F.softplus(self.readout(self.block(tok).mean(1)).squeeze(-1))

class StaticBaseline(nn.Module):
    def __init__(self, n, m, d=64, **kw):
        super().__init__(); self.pos = nn.Embedding(m, d); self.fid = nn.Embedding(n, d); self.n = n; self.scale = nn.Parameter(torch.zeros(()))
    def forward(self, s, g):
        ids = torch.arange(self.n, device=s.device)
        zs = self.pos(s) + self.fid(ids)[None]; zg = self.pos(g) + self.fid(ids)[None]
        return F.softplus(self.scale) * (zs - zg).norm(dim=-1).sum(-1)


class SymWhole(nn.Module):
    """Whole-state symmetric embedding: D(s,g) = ||f(s) - f(g)||_1.
    attn: plain transformer, mean-pool.  flat: plain transformer, flattened tokens.
    rec: weight-shared block iterated T times with self-recall, mean-pool."""
    def __init__(self, n, m, variant, d=64, heads=4, layers=3, T=6, **kw):
        super().__init__()
        self.pos = nn.Embedding(m, d); self.fid = nn.Embedding(n, d); self.n = n
        self.variant = variant; self.T = T
        self.block = Block(d, heads, layers)
        self.proj = nn.Linear(n * d, d) if variant == "flat" else nn.Linear(d, d)
    def _f(self, x):
        ids = torch.arange(self.n, device=x.device)
        tok = self.pos(x) + self.fid(ids)[None]
        if self.variant == "attn":
            return self.proj(self.block(tok).mean(1))
        if self.variant == "flat":
            return self.proj(self.block(tok).flatten(1))
        cur, base = tok, tok
        for _ in range(self.T):
            cur = self.block(torch.cat([cur, base], 1))[:, :self.n]
        return self.proj(cur.mean(1))
    def forward(self, s, g):
        return (self._f(s) - self._f(g)).abs().sum(-1)


def fdist(a, b, gt, ms):
    """per-factor UNGUARDED graph distance (guard-blind): cycle=min(d,ms-d), path=|d|, grid=manhattan."""
    if gt == "grid":
        return ((a // ms - b // ms).abs() + (a % ms - b % ms).abs()).float()
    if gt == "path":
        return (a - b).abs().float()
    d = (a - b).abs(); return torch.minimum(d, ms - d).float()

class SumCyc(nn.Module):
    """BASELINE: scale * sum_i unguarded per-factor distance + bias. Guard-BLIND -> quantifies how much
    the guards matter (true guarded distance >= this)."""
    def __init__(self, n, ms, gt, **kw):
        super().__init__(); self.n = n; self.ms = ms; self.gt = gt
        self.a = nn.Parameter(torch.ones(())); self.b = nn.Parameter(torch.zeros(()))
    def forward(self, s, g):
        tot = sum(fdist(s[:, i], g[:, i], self.gt, self.ms) for i in range(self.n))
        return F.softplus(self.a) * tot + self.b

class OneHot(nn.Module):
    """BASELINE: scale * (# differing factors) + bias. Crudest floor (position-blind Hamming)."""
    def __init__(self, n, R, **kw):
        super().__init__(); self.a = nn.Parameter(torch.ones(())); self.b = nn.Parameter(torch.zeros(()))
    def forward(self, s, g): return F.softplus(self.a) * (s != g).float().sum(-1) + self.b

def metric_embedding(R, d, gt, ms, scale):
    """analytic per-factor metric embedding: circle (cycle), line (path), lattice (grid)."""
    W = torch.zeros(R, d)
    if gt == "cycle":
        for p in range(R): W[p, 0] = scale * math.cos(2 * math.pi * p / ms); W[p, 1] = scale * math.sin(2 * math.pi * p / ms)
    elif gt == "path":
        for p in range(R): W[p, 0] = scale * float(p)
    else:
        for p in range(R): W[p, 0] = scale * float(p // ms); W[p, 1] = scale * float(p % ms)
    return W

def make_model(a, n, R):
    if a.model == "integrator": return Integrator(n, R, a.d, a.heads, a.layers, a.T, bool(a.share), a.flow, a.disp, a.stepcap)
    if a.model == "mlp": return MLPBaseline(n, R, a.d)
    if a.model == "transformer": return TransformerBaseline(n, R, a.d, a.heads, a.layers)
    if a.model == "sumcyc": return SumCyc(n, a.m, a.graphtype)
    if a.model == "onehot": return OneHot(n, R)
    if a.model.startswith("sym_"): return SymWhole(n, R, a.model[4:], a.d, a.heads, a.layers, a.T)
    return StaticBaseline(n, R, a.d)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="integrator", choices=["integrator", "mlp", "transformer", "static", "sumcyc", "onehot",
                                                              "sym_attn", "sym_flat", "sym_rec"])
    ap.add_argument("--graphtype", default="cycle", choices=["cycle", "path", "grid"], help="per-factor graph (grid: each factor an m x m grid)")
    ap.add_argument("--input", default="index", choices=["index", "image"], help="index = embed factor id (clean); image = render state as n x R pixel grid, CNN reads it (test image input)")
    ap.add_argument("--flow", default="attractor", choices=["attractor", "velocity", "joint", "act", "fp", "qsweep"])
    ap.add_argument("--qtol", type=float, default=0.5, help="qsweep: |cost-true|<qtol counts as 'correct' for the halt-head BCE target")
    ap.add_argument("--eps", type=float, default=0.05, help="fp flow: convergence threshold (freeze sample when step < eps)")
    ap.add_argument("--haltmode", default="converge", choices=["converge", "arrive"], help="fp: converge (dz<eps) or arrive (||z-goal||<eps_arr, for stepcap traversal)")
    ap.add_argument("--eps_arr", type=float, default=0.6, help="arrive halt threshold")
    ap.add_argument("--curriculum", type=int, default=0, help="1 = grow distance cap AND iteration budget over training")
    ap.add_argument("--tbuf", type=int, default=4, help="curriculum: iteration budget = dist-cap + tbuf")
    ap.add_argument("--Tmin", type=int, default=-1, help="anytime training: if >=0, sample step budget ~ U[Tmin, T] each step so the flow learns to reach+stay at ANY budget -> one fixed test budget works without knowing the distance")
    ap.add_argument("--supervision", default="geodesic", choices=["geodesic", "bellman"], help="geodesic = regress true BFS dist (needs labels); bellman = SELF-SUPERVISED d(s,g)=1+min_nbr d(s',g), only local transitions")
    ap.add_argument("--selffrac", type=float, default=0.15, help="bellman: fraction of batch set to s==g (base case d=0)")
    ap.add_argument("--isoloss", type=float, default=0.0, help="aux loss: per-factor ||emb(s_i)-emb(g_i)|| ~ graph dist (metric embeddings)")
    ap.add_argument("--initemb", default="rand", choices=["rand", "metric"], help="metric = init pos embedding to circle/line/lattice")
    ap.add_argument("--embscale", type=float, default=1.0)
    ap.add_argument("--ponder", type=float, default=0.01, help="ACT ponder penalty (encourage halting); flow=act only")
    ap.add_argument("--stepcap", type=float, default=0.0, help="cap per-factor step norm (act); >0 forces far goals to need MORE steps")
    ap.add_argument("--recall", type=int, default=0, help="Deep-Thinking recall: re-inject the raw (start,goal) tokens EVERY step (fixes overthinking)")
    ap.add_argument("--progressive", type=int, default=0, help="Deep-Thinking progressive loss: warm k~U[0,T-1] steps no-grad then finish with grad -> steady state, no overthinking")
    ap.add_argument("--cost", default="length", choices=["length", "energy"], help="length = sum ||dz|| (path length); energy = sum ||dz||^2 (Benamou-Brenier kinetic action)")
    ap.add_argument("--disp", default="l2", choices=["l1", "l2"]); ap.add_argument("--share", type=int, default=1)
    ap.add_argument("--T", type=int, default=6); ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--Ttest", type=int, default=-1, help="eval loop count (share=1 only); -1 = same as --T. Tests running the flow LONGER at test time for farther goals.")
    ap.add_argument("--arrive", type=float, default=0.0, help="weight on arrival loss ||z_T - goal|| (attractor); makes the flow PARK at the goal so extra test-time steps add ~0 cost (halting).")
    ap.add_argument("--n", type=int, default=3); ap.add_argument("--m", type=int, default=6)
    ap.add_argument("--key", type=int, default=1, help="guard key set = {0..key-1}"); ap.add_argument("--guarded", type=int, default=1)
    ap.add_argument("--guardmode", default="chain", choices=["chain", "dag"], help="chain: factor i guarded by i-1; dag: by a random subset")
    ap.add_argument("--split", default="dist", choices=["dist", "combo"], help="dist: extrapolate to longer paths; combo: generalize to unseen GOAL configurations")
    ap.add_argument("--comboheld", type=float, default=0.2, help="fraction of goal states held out (split=combo)")
    ap.add_argument("--D", type=int, default=5, help="train on dist<=D, extrapolate on dist>D (split=dist)")
    ap.add_argument("--d", type=int, default=64); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--n_start", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--valevery", type=int, default=0, help=">0: every N steps print a VAL line with MULTI-AXIS generalization (in-range, distance-extrap by bin, compositional). Also forces dual holdout (train excludes dist>D AND held-out goal configs).")
    ap.add_argument("--save", default="", help="path to save {state_dict, args, result} checkpoint")
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed)
    gt = args.graphtype; ms = args.m; R = radix(gt, ms)
    keyset = set(range(args.key))
    parents = build_guards(args.n, args.guardmode, np.random.default_rng(args.seed + 99))
    S, Gg, Dd = build_pool(args.n, R, gt, ms, parents, keyset, bool(args.guarded), args.n_start, rng)
    if len(S) < 200: print("RESULT " + json.dumps({"error": "tiny pool", "npool": int(len(S))})); return
    # splits: dist -> extrap = dist>D ; combo -> extrap = pairs whose GOAL config is held out (unseen)
    if args.split == "combo":
        genc = np.array([enc(g, R) for g in Gg]); rh = np.random.default_rng(args.seed + 777); ug = np.unique(genc)
        held = set(int(x) for x in rh.choice(ug, size=max(1, int(args.comboheld * len(ug))), replace=False))
        ex = np.array([int(g) in held for g in genc])
    else:
        ex = Dd > args.D
    le = ~ex; idx = np.where(le)[0]; rng.shuffle(idx)
    n_ho = max(50, int(0.2 * len(idx))); ho = set(idx[:n_ho].tolist())
    tr = np.array([i for i in idx if i not in ho]); ind = np.array(sorted(ho)); exi = np.where(ex)[0]
    St, Gt, Dt = (torch.as_tensor(a, device=dev) for a in (S, Gg, Dd))
    ax_far = ax_combo = None
    if args.valevery > 0:                                       # DUAL holdout: monitor distance-extrap AND compositional together
        genc = np.array([enc(g, R) for g in Gg]); rh = np.random.default_rng(args.seed + 777); ug = np.unique(genc)
        held_g = set(int(x) for x in rh.choice(ug, size=max(1, int(args.comboheld * len(ug))), replace=False))
        is_combo = np.array([int(g) in held_g for g in genc]); is_far = Dd > args.D
        tri = np.where((~is_combo) & (~is_far))[0]; rng.shuffle(tri)     # train: in-range AND not held-out config
        n_ho = max(50, int(0.15 * len(tri))); ind = np.array(sorted(tri[:n_ho])); tr = tri[n_ho:]
        ax_combo = np.where(is_combo & (~is_far))[0]            # compositional: unseen config, in trained distance range
        ax_far = np.where(is_far)[0]                            # distance extrapolation: unseen (longer) distances
    MAXDEG = 4 * args.n
    def neighbor_pack(s_np):                                    # legal 1-step neighbours per state (self-supervised signal)
        B = len(s_np); nb = np.repeat(s_np[:, None, :], MAXDEG, 1); mask = np.zeros((B, MAXDEG), np.float32)
        for b in range(B):
            for j, v in enumerate(neighbours(enc(s_np[b], R), args.n, R, gt, ms, parents, keyset, bool(args.guarded))[:MAXDEG]):
                nb[b, j] = dec(v, args.n, R); mask[b, j] = 1.0
        return nb, mask
    model = make_model(args, args.n, R).to(dev)
    if hasattr(model, "recall"): model.recall = bool(args.recall); model.cost = args.cost
    if hasattr(model, "input"): model.input = args.input; model.R = R
    if getattr(model, "flow", None) == "fp": model.eps = args.eps; model.haltmode = args.haltmode; model.eps_arr = args.eps_arr
    if args.initemb == "metric" and hasattr(model, "pos"):
        with torch.no_grad(): model.pos.weight[:R] = metric_embedding(R, args.d, gt, ms, args.embscale).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    trt = torch.as_tensor(tr, device=dev); Dtr = Dt[trt]
    @torch.no_grad()
    def multival(step):                                        # intermediate multi-axis generalization report
        model.eval(); tv = model.T; model.T = args.Ttest if args.Ttest > 0 else args.T
        def mm(ix):
            if ix is None or len(ix) == 0: return None
            ix = torch.as_tensor(ix[:3000], device=dev); return round(float((model(St[ix], Gt[ix]) - Dt[ix]).abs().mean()), 3)
        fb = {}
        if ax_far is not None:
            for dd in sorted(set(int(x) for x in Dd[ax_far])): fb[dd] = mm(ax_far[Dd[ax_far] == dd])
        print("VAL step=%d inrange=%s combo=%s extrap=%s" % (step, mm(ind), mm(ax_combo), json.dumps(fb)), flush=True)
        model.T = tv; model.train()
    for step in range(args.steps):
        if args.valevery > 0 and step % args.valevery == 0: multival(step)
        if args.curriculum and args.model == "integrator":     # grow distance cap + iteration budget together
            frac = min(1.0, step / (0.7 * args.steps)); Dcap = 2.0 + frac * (max(2, args.D) - 2.0)
            model.T = int(round(Dcap + args.tbuf))
            elig = trt[Dtr <= Dcap]; elig = elig if len(elig) >= args.bs else trt
            b = elig[torch.randint(0, len(elig), (args.bs,), device=dev)]
        else:
            b = trt[torch.randint(0, len(trt), (args.bs,), device=dev)]
        if args.Tmin >= 0 and args.model == "integrator" and not args.curriculum:   # anytime: random budget each step
            model.T = int(np.random.randint(args.Tmin, args.T + 1))
        if args.progressive and args.model == "integrator":       # Deep-Thinking progressive: warm k~U[0,T-1] no-grad
            model.warm = int(np.random.randint(0, max(1, model.T)))
        if args.flow == "qsweep" and args.model == "integrator":   # EqR-style: min-reg (cost passes through true) + correctness-supervised halt
            tgt = Dt[b]; model(St[b], Gt[b])                   # populates cost_traj / q_traj
            ct = model.cost_traj                               # (T,B)
            reg = ((ct - tgt[None]) ** 2).min(0).values.mean()
            correct = ((ct.detach() - tgt[None]).abs() < args.qtol).float()
            hl = F.binary_cross_entropy_with_logits(model.q_traj, correct)
            loss = reg + 0.5 * hl
            opt.zero_grad(); loss.backward(); opt.step(); continue
        if args.supervision == "bellman":                      # SELF-SUPERVISED: no BFS labels, only local transitions
            bi = tr[np.random.randint(0, len(tr), args.bs)]; s_np = S[bi].copy(); g_np = Gg[bi].copy()
            nself = int(args.selffrac * args.bs); g_np[:nself] = s_np[:nself]   # base case s==g -> target 0
            nb, mask = neighbor_pack(s_np)
            sb = torch.as_tensor(s_np, device=dev); gb = torch.as_tensor(g_np, device=dev)
            nbf = torch.as_tensor(nb.reshape(-1, args.n), device=dev)
            gg = torch.as_tensor(np.repeat(g_np, MAXDEG, 0), device=dev); mk = torch.as_tensor(mask, device=dev)
            with torch.no_grad():
                dn = model(nbf, gg).reshape(args.bs, MAXDEG).masked_fill(mk < 0.5, 1e9)
                same = (sb == gb).all(-1)
                tgt = torch.where(same, torch.zeros(args.bs, device=dev), 1.0 + dn.min(1).values)
            loss = F.mse_loss(model(sb, gb), tgt)
        else:
            pred = model(St[b], Gt[b]); loss = F.mse_loss(pred, Dt[b])
        if args.arrive > 0 and args.model == "integrator" and args.flow in ("attractor", "act", "fp"):
            loss = loss + args.arrive * model.arrive.mean()
        if args.flow in ("act", "fp") and args.model == "integrator":
            loss = loss + args.ponder * model.steps.mean()     # ponder: encourage fewer / efficient steps
        if args.isoloss > 0 and hasattr(model, "pos"):         # metric embeddings: per-factor emb dist ~ graph dist
            Sb, Gb = St[b], Gt[b]
            ed = torch.stack([(model.pos(Sb[:, i]) - model.pos(Gb[:, i])).norm(dim=-1) for i in range(args.n)], -1)
            gd = torch.stack([fdist(Sb[:, i], Gb[:, i], gt, ms) for i in range(args.n)], -1)
            loss = loss + args.isoloss * ((ed - gd) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    if hasattr(model, "warm"): model.warm = 0                      # no warm-up at test
    if args.Ttest > 0 and args.model == "integrator" and args.share:   # run the flow LONGER at test time
        model.T = args.Ttest
    @torch.no_grad()
    def mae(ix):
        if len(ix) == 0: return None
        ix = torch.as_tensor(ix, device=dev); p = model(St[ix], Gt[ix])
        return float((p - Dt[ix]).abs().mean())
    @torch.no_grad()
    def extrap_curve():
        out = {}
        for dd in sorted(set(int(x) for x in Dd[exi])):
            sel = exi[Dd[exi] == dd]
            if len(sel): out[dd] = round(mae(sel), 3)
        return out
    @torch.no_grad()
    def act_diag():
        """ACT diagnostics: does the flow take MORE steps for farther goals, and does it PARK (arrive~0)?"""
        if args.flow not in ("act", "fp", "qsweep"): return {}, {}   # step diagnostics for self-halting flows
        steps_by_d, arr_by_d = {}, {}
        for dd in sorted(set(int(x) for x in Dd)):
            sel = np.where(Dd == dd)[0]
            if len(sel) < 5: continue
            ix = torch.as_tensor(sel[:3000], device=dev); model(St[ix], Gt[ix])
            steps_by_d[dd] = round(float(model.steps.mean()), 2); arr_by_d[dd] = round(float(model.arrive.mean()), 2)
        return steps_by_d, arr_by_d
    steps_by_d, arr_by_d = act_diag()
    @torch.no_grad()
    def by_d_all():                                             # FIXED-budget honesty: MAE at each distance (near AND far), one Ttest
        held = np.concatenate([ind, exi]) if len(exi) else ind; out = {}
        for dd in sorted(set(int(x) for x in Dd[held])):
            sel = held[Dd[held] == dd]
            if len(sel) >= 3: out[dd] = round(mae(sel), 3)
        return out
    @torch.no_grad()
    def iso_diag():
        """does per-factor EMBEDDING distance reflect GRAPH distance? (the metric-embedding hypothesis)"""
        if not hasattr(model, "pos"): return None, None
        ix = torch.arange(min(6000, len(S)), device=dev); Sb, Gb = St[ix], Gt[ix]
        ed = torch.cat([(model.pos(Sb[:, i]) - model.pos(Gb[:, i])).norm(dim=-1) for i in range(args.n)])
        gd = torch.cat([fdist(Sb[:, i], Gb[:, i], gt, ms) for i in range(args.n)])
        if ed.std() < 1e-6: return 0.0, None
        corr = float(torch.corrcoef(torch.stack([ed, gd]))[0, 1])
        A = torch.stack([gd, torch.ones_like(gd)], 1); sol = torch.linalg.lstsq(A, ed).solution
        return round(corr, 3), round(float((A @ sol - ed).abs().mean()), 3)
    emb_corr, emb_mae = iso_diag()
    res = dict(model=args.model, flow=args.flow, disp=args.disp, share=args.share, T=args.T, Ttest=(args.Ttest if args.Ttest > 0 else args.T), arrive=args.arrive, ponder=args.ponder, stepcap=args.stepcap, eps=args.eps, curriculum=args.curriculum, tbuf=args.tbuf, isoloss=args.isoloss, initemb=args.initemb, recall=args.recall, progressive=args.progressive, cost=args.cost, input=args.input, layers=args.layers,
               n=args.n, m=args.m, graphtype=args.graphtype, guarded=args.guarded, key=args.key, guardmode=args.guardmode, split=args.split, D=args.D, d=args.d, steps=args.steps, seed=args.seed, lr=args.lr,
               npool=int(len(S)), maxd=int(Dd.max()), n_extrap=int(ex.sum()),
               train_mae=round(mae(tr), 3), indist_mae=round(mae(ind), 3),
               extrap_mae=round(mae(exi), 3) if len(exi) else None, extrap_by_d=extrap_curve(),
               supervision=args.supervision, Tmin=args.Tmin, emb_corr=emb_corr, emb_mae=emb_mae,
               by_d_all=by_d_all(), act_steps_by_d=steps_by_d, act_arrive_by_d=arr_by_d)
    print("RESULT " + json.dumps(res), flush=True)
    if args.save:
        torch.save({"state_dict": model.state_dict(), "args": vars(args), "R": R, "result": res}, args.save)
        print("SAVED " + args.save, flush=True)

if __name__ == "__main__":
    main()

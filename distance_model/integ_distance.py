"""Integration-based distance model on a coupling gridworld.

Trains a recall-flow integrator to predict the BFS geodesic distance between two states, and reports DISTANCE
ACCURACY (MAE, correlation) and GENERALIZATION to held-out constraint configurations. Input is either factored
(--enc factored) or an IMAGE (--enc bmask|marker: agents rendered onto a GxG canvas, constraints in a key token).

State = [positions(N), mobility_key (base-4 over gated agents 1..N-1), link_key (bitmask over agent pairs)]. Linked
agents move as a rigid group, so coupling couples factors and reduces the reachable set. --heldout {combo,links2,dofhi,
dofhi2} holds out constraint configurations to test generalization. --Rtrain caps the distance of TRAINING pairs while
evaluation keeps the full Rmax ball (length extrapolation: train short, test long; reports mae_beyond/mae_within).
No latent-space / DOF analysis -- distance only.
"""
import argparse, collections, itertools, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

CTRL = 0
ENC = "factored"
INJECT = False
HELDMODE = ""
_HELD = set()

AXES = {0: frozenset({"H", "V"}), 1: frozenset({"H"}), 2: frozenset({"V"}), 3: frozenset()}   # free/H/V/lock
AX2DIR = {"H": ((0, 1), (0, -1)), "V": ((1, 0), (-1, 0))}

def PAIRS(N): return list(itertools.combinations(range(1, N), 2))
def knob(mk, i): return (mk // (4 ** (i - 1))) % 4
def knobs_tuple(mk, N): return tuple(knob(mk, i) for i in range(1, N))
def mobmax(N): return 4 ** (N - 1)
def linkmax(N): return 1 << len(PAIRS(N))

def components(N, lk):
    parent = list(range(N))
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for b, (i, j) in enumerate(PAIRS(N)):
        if (lk >> b) & 1: parent[find(i)] = find(j)
    comp = collections.defaultdict(list)
    for a in range(1, N): comp[find(a)].append(a)
    return list(comp.values())

def comp_axes(members, mk):
    ax = frozenset({"H", "V"})
    for a in members: ax &= AXES[knob(mk, a)]
    return ax

def state_dof(state, N):                                         # only used for stratified sampling + the dofhi split
    mk, lk = int(state[N]), int(state[N + 1])
    return 2 + sum(len(comp_axes(m, mk)) for m in components(N, lk))

def neighbours(state, N, G):
    mk, lk = int(state[N]), int(state[N + 1]); out = []
    r0, c0 = state[0] // G, state[0] % G                          # agent 0 = free mover
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r0 + dr, c0 + dc
        if 0 <= nr < G and 0 <= nc < G:
            s = state.copy(); s[0] = nr * G + nc; out.append(s)
    for members in components(N, lk):                            # linked components move as a rigid group
        ax = comp_axes(members, mk)
        for a in ax:
            for dr, dc in AX2DIR[a]:
                ok = True; s = state.copy()
                for m in members:
                    nr, nc = state[m] // G + dr, state[m] % G + dc
                    if not (0 <= nr < G and 0 <= nc < G): ok = False; break
                    s[m] = nr * G + nc
                if ok: out.append(s)
    if state[0] == CTRL:                                          # reconfigure constraints at the control cell
        for i in range(1, N):
            for nk in range(4):
                if nk != knob(mk, i):
                    s = state.copy(); s[N] = mk + (nk - knob(mk, i)) * (4 ** (i - 1)); out.append(s)
        for b in range(len(PAIRS(N))):
            s = state.copy(); s[N + 1] = lk ^ (1 << b); out.append(s)
    return out

MAXN = 8000

def bfs_local(src, N, G, Rmax, maxnodes=None):
    maxnodes = maxnodes or MAXN
    src = tuple(src); dist = {src: 0}; dq = collections.deque([src])
    while dq and len(dist) < maxnodes:
        u = dq.popleft(); du = dist[u]
        if du >= Rmax: continue
        for v in neighbours(list(u), N, G):
            tv = tuple(v)
            if tv not in dist:
                dist[tv] = du + 1; dq.append(tv)
                if len(dist) >= maxnodes: break
    return dist

def proxy_np(A, B, N, G):
    """Exact admissible factorized lower bound, any range: per-agent Manhattan + knob toggles + link toggles."""
    A, B = np.asarray(A), np.asarray(B); d = np.zeros(len(A), np.float32)
    for i in range(N):
        d += np.abs(A[:, i] // G - B[:, i] // G) + np.abs(A[:, i] % G - B[:, i] % G)
    for i in range(1, N):
        d += ((A[:, N] // 4 ** (i - 1)) % 4 != (B[:, N] // 4 ** (i - 1)) % 4)
    x = A[:, N + 1] ^ B[:, N + 1]
    for b in range(len(PAIRS(N))): d += (x >> b) & 1
    return d

def greedy_upper(s, g, N, G, budget=600):
    """Suboptimal-search upper bound on d(s,g) via best-first on proxy heuristic; None if not found."""
    import heapq
    h0 = float(proxy_np(np.array(s)[None], np.array(g)[None], N, G)[0])
    pq = [(h0, 0, tuple(s))]; seen = {tuple(s): 0}; gt = tuple(g)
    for _ in range(budget):
        if not pq: break
        _, cost, u = heapq.heappop(pq)
        if u == gt: return cost
        for v in neighbours(list(u), N, G):
            tv = tuple(v); nc = cost + 1
            if seen.get(tv, 1 << 30) > nc:
                seen[tv] = nc
                h = float(proxy_np(np.array(v)[None], np.array(g)[None], N, G)[0])
                heapq.heappush(pq, (nc + h, nc, tv))
    return None

def is_test(state, N):
    if HELDMODE == "links2": return bin(int(state[N + 1])).count("1") >= 2       # train <=1 link, test >=2 (unseen multi-link)
    if HELDMODE in ("dofhi", "dofhi2"): return state_dof(state, N) >= (7 if HELDMODE == "dofhi" else 6)   # unseen DOF levels
    return (knobs_tuple(int(state[N]), N), int(state[N + 1])) in _HELD             # held-out (mobility,link) combos

def _rand_state(N, G, rng):
    st = np.zeros(N + 2, np.int64)
    for i in range(N): st[i] = rng.integers(G * G)
    st[N] = rng.integers(mobmax(N)); st[N + 1] = rng.integers(linkmax(N))
    return st

def make_state(N, G, rng, part="all", dof=None):                 # DOF-stratified (random states skew low-DOF)
    for _ in range(800):
        st = _rand_state(N, G, rng)
        okp = part == "all" or (part == "test") == is_test(st, N)
        okd = dof is None or state_dof(st, N) == dof
        if okp and okd: return st
    return st

def build_pool(N, G, Rmax, nq, rng):
    S1, S2, D = [], [], []
    for _ in range(nq):
        dof = int(rng.integers(2, 7)) if HELDMODE in ("dofhi", "dofhi2") else int(rng.integers(2, 2 * N + 1))
        q = make_state(N, G, rng, part="train", dof=dof); dist = bfs_local(q, N, G, Rmax)
        byd = collections.defaultdict(list)
        for tv, dv in dist.items():
            if dv > 0: byd[dv].append(tv)
        per = max(1, 60 // max(1, len(byd))); pick = []
        for dv, lst in byd.items():
            idx = rng.choice(len(lst), min(per, len(lst)), replace=False)
            pick += [(lst[j], dv) for j in idx]
        for tv, dv in pick:
            S1.append(q); S2.append(np.array(tv)); D.append(dv)
    return np.array(S1), np.array(S2), np.array(D, np.float32)


class Block(nn.Module):
    def __init__(self, d, heads, layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, activation="gelu",
                                     batch_first=True, norm_first=True) for _ in range(layers)])
    def forward(self, z):
        for l in self.layers: z = l(z)
        return z


class Enc(nn.Module):
    """N position tokens (factored | image canvas) + one additive constraint token."""
    def __init__(self, N, G, d, heads=4):
        super().__init__()
        self.N, self.G, self.d = N, G, d; self.P = len(PAIRS(N)); self.NP = G * G
        self.posf = nn.Embedding(G * G, d)
        self.mob_emb = nn.Embedding((N - 1) * 4, d)                 # additive constraint token: per (agent,knob)
        self.link_emb = nn.Embedding(self.P, d)                     # ... + per active link pair
        self.cbase = nn.Parameter(torch.randn(d) * 0.02)
        # image position encoding: agents on a GxG lossless canvas (cell = base + Σ present-agent embeddings)
        self.pe = nn.Embedding(self.NP, d)
        self.cellemb = nn.Embedding(N, d)
        self.cellbase = nn.Parameter(torch.randn(d) * 0.02)
        self.query = nn.Parameter(torch.randn(N, d) * 0.02)        # marker: per-agent-slot query
        self.slotid = nn.Embedding(N, d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
    def ctoken(self, mk, lk):
        tok = self.cbase[None].expand(mk.shape[0], self.d).clone()
        for i in range(1, self.N):
            tok = tok + self.mob_emb.weight[(i - 1) * 4 + ((mk // (4 ** (i - 1))) % 4)]
        for b in range(self.P):
            tok = tok + ((lk >> b) & 1).float()[..., None] * self.link_emb.weight[b]
        return tok
    def _canvas(self, s):
        B, dev = s.shape[0], s.device; ar = torch.arange(B, device=dev)
        bits = torch.zeros(B, self.NP, dtype=torch.long, device=dev)
        for i in range(self.N): bits[ar, s[:, i]] |= (1 << i)
        tok = self.cellbase[None, None].expand(B, self.NP, self.d).clone()
        for i in range(self.N):
            tok = tok + ((bits >> i) & 1).float()[..., None] * self.cellemb.weight[i]
        return tok + self.pe(torch.arange(self.NP, device=dev))[None], bits
    def _positions(self, s):
        if ENC == "factored":
            return [self.posf(s[:, i]) for i in range(self.N)]
        tok, bits = self._canvas(s)
        if ENC == "bmask":                                         # lossless: recover agent i cell deterministically
            return [self.posf(((bits >> i) & 1).float().argmax(1)) for i in range(self.N)]
        ids = torch.arange(self.N, device=s.device)                # marker: learn to bind agent i from the shared canvas
        q = (self.query + self.slotid(ids))[None].expand(s.shape[0], -1, -1)
        bound, _ = self.attn(q, tok, tok)
        return [bound[:, i] for i in range(self.N)]
    def forward(self, s):
        mk, lk = s[:, self.N], s[:, self.N + 1]; comps = []; pos = self._positions(s)
        for i in range(self.N):
            t = pos[i]
            if INJECT and i >= 1:                                  # optional per-agent constraint injection
                t = t + self.mob_emb.weight[(i - 1) * 4 + ((mk // (4 ** (i - 1))) % 4)]
                for b, (p, q) in enumerate(PAIRS(self.N)):
                    if i in (p, q): t = t + ((lk >> b) & 1).float()[..., None] * self.link_emb.weight[b]
            comps.append(t)
        comps.append(self.ctoken(mk, lk))
        return torch.stack(comps, 1)


class Integrator(nn.Module):
    """Recall-flow integrator: distance = accumulated path length over T weight-shared steps (goal+start re-injected)."""
    def __init__(self, N, G, d=64, heads=4, layers=3, T=14):
        super().__init__()
        self.n = N + 1; self.T = T
        self.enc = Enc(N, G, d, heads)
        self.fid = nn.Embedding(N + 1, d); self.role = nn.Embedding(3, d)
        self.block = Block(d, heads, layers); self.scale = nn.Parameter(torch.zeros(()))
        self.gbase = None; self.softcount = False; self.capval = 0.0
    def emb(self, s, role):
        ids = torch.arange(self.n, device=s.device)
        return self.enc(s) + self.fid(ids)[None] + self.role(torch.tensor(role, device=s.device))
    def forward(self, s, g, Trun=None, ret_arr=False, Tper=None):
        n = self.n; zs = self.emb(s, 0); zg = self.emb(g, 1); base = torch.cat([zg, self.emb(s, 2)], 1)
        gtok = self.emb(g, 0)
        tok = torch.cat([zs, base], 1); cost = torch.zeros(s.shape[0], device=s.device)
        self.cappen = torch.zeros((), device=s.device)
        Tmax = int(Tper.max().item()) if Tper is not None else (Trun or self.T)
        for t in range(Tmax):
            z = self.block(tok); step = (z[:, :n] - tok[:, :n]).norm(dim=-1)
            if self.capval > 0: self.cappen = self.cappen + F.relu(step - self.capval).mean()
            if self.softcount:                                     # count not-yet-arrived steps instead of path length
                inc = torch.sigmoid(((z[:, :n] - gtok).norm(dim=-1).mean(-1) - 1.0) / 0.25)
            else:
                inc = step.sum(-1)
            if Tper is not None: inc = inc * (t < Tper).float()
            cost = cost + inc
            tok = torch.cat([z[:, :n], base], 1)                   # re-inject goal + start (recall)
        out = F.softplus(self.scale) * cost
        out = out + F.softplus(self.gbase) if self.gbase is not None else out
        if ret_arr:                                                # arrival: park the flow at the goal's tokens
            return out, (tok[:, :n] - gtok).norm(dim=-1).mean(-1)
        return out


class LinMix(nn.Module):
    """Linear head over [flow cost, proxy, per-factor coordinate gaps, config diffs]: the nonlinear flow
    supplies shape, linear extrapolating features supply scale."""
    def __init__(self, N, G, d=64, heads=4, layers=3, T=14):
        super().__init__()
        self.N, self.G = N, G
        self.flow = Integrator(N, G, d, heads, layers, T)
        self.flow.scale.data.fill_(0.0)
        self.w = nn.Linear(2 * N + 3, 1)
    def feats(self, s, g):
        f = []
        for i in range(self.N):
            f.append((s[:, i] // self.G - g[:, i] // self.G).abs().float())
            f.append((s[:, i] % self.G - g[:, i] % self.G).abs().float())
        kd = torch.zeros_like(f[0])
        for i in range(1, self.N):
            kd = kd + ((s[:, self.N] // 4 ** (i - 1)) % 4 != (g[:, self.N] // 4 ** (i - 1)) % 4).float()
        ld = torch.zeros_like(f[0]); x = s[:, self.N + 1] ^ g[:, self.N + 1]
        for b in range(len(PAIRS(self.N))): ld = ld + ((x >> b) & 1).float()
        return torch.stack(f + [kd, ld], 1)
    def forward(self, s, g, **kw):
        cost = self.flow(s, g)
        return F.softplus(self.w(torch.cat([cost[:, None], self.feats(s, g)], 1)).squeeze(-1))


class QmetHead(nn.Module):
    """torchqmet baselines: shared attn encoder -> latent pair -> IQE or MRNFixed quasimetric head."""
    def __init__(self, N, G, kind, d=64, heads=4, layers=3):
        super().__init__()
        import torchqmet
        self.n = N + 1
        self.enc = Enc(N, G, d, heads); self.fid = nn.Embedding(N + 1, d)
        self.block = Block(d, heads, layers); self.proj = nn.Linear(d, d)
        self.head = torchqmet.IQE(d, dim_per_component=16) if kind == "iqe" else torchqmet.MRNFixed(d)
    def embed(self, s):
        ids = torch.arange(self.n, device=s.device)
        return self.proj(self.block(self.enc(s) + self.fid(ids)[None]).mean(1))
    def forward(self, s, g, **kw):
        return self.head(self.embed(s), self.embed(g))


class SymEmbed(nn.Module):
    """Symmetric-embedding baseline: D(s,g) = ||f(s) - f(g)||_1 with three encoder variants.
    mlp: flatten tokens -> MLP.  attn: plain (non-shared) transformer, mean-pooled.
    flat: plain transformer, FLATTENED tokens -> linear (slot-preserving readout).
    rec: OUR recurrent weight-shared block iterated T times with self-recall, mean-pooled."""
    def __init__(self, N, G, variant, d=64, heads=4, layers=3, T=14):
        super().__init__()
        self.n = N + 1; self.T = T; self.variant = variant
        self.enc = Enc(N, G, d, heads); self.fid = nn.Embedding(N + 1, d)
        if variant == "mlp":
            self.head = nn.Sequential(nn.Linear(self.n * d, 2 * d), nn.GELU(), nn.Linear(2 * d, 2 * d),
                                      nn.GELU(), nn.Linear(2 * d, d))
        else:
            self.block = Block(d, heads, layers)
            self.proj = nn.Linear(self.n * d, d) if variant == "flat" else nn.Linear(d, d)
    def embed(self, s):
        ids = torch.arange(self.n, device=s.device)
        tok = self.enc(s) + self.fid(ids)[None]
        if self.variant == "mlp":
            return self.head(tok.flatten(1))
        if self.variant == "attn":
            return self.proj(self.block(tok).mean(1))
        if self.variant == "flat":
            return self.proj(self.block(tok).flatten(1))
        cur, base = tok, tok                                       # rec: weight-shared iterations + self-recall
        for _ in range(self.T):
            cur = self.block(torch.cat([cur, base], 1))[:, :self.n]
        return self.proj(cur.mean(1))
    def forward(self, s, g):
        return (self.embed(s) - self.embed(g)).abs().sum(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nag", type=int, default=4); ap.add_argument("--G", type=int, default=6)
    ap.add_argument("--Rmax", type=int, default=12); ap.add_argument("--T", type=int, default=14)
    ap.add_argument("--Rtrain", type=int, default=0, help="cap training-pair distance (0 = Rmax); eval always uses Rmax")
    ap.add_argument("--arch", choices=["integ", "sym_mlp", "sym_attn", "sym_flat", "sym_rec", "iqe", "mrn"], default="integ")
    ap.add_argument("--globalbase", type=int, default=0, help="integ: add learned base once to the output")
    ap.add_argument("--maxnodes", type=int, default=8000, help="BFS ball node cap (raise for Rmax > 12)")
    ap.add_argument("--arrive", type=float, default=0.0, help="integ: arrival-loss weight (park at goal)")
    ap.add_argument("--Tmin", type=int, default=-1, help="integ: anytime training, per-step budget ~ U[Tmin, T]")
    ap.add_argument("--Ttest", type=int, default=-1, help="integ: eval loop count (-1 = T); large + parking = budget-free")
    ap.add_argument("--bellman", type=float, default=0.0, help="weight of Bellman self-consistency on UNLABELED pairs")
    ap.add_argument("--bellpairs", type=int, default=20000); ap.add_argument("--bellbs", type=int, default=64)
    ap.add_argument("--bellwalk", type=int, default=0, help="multi-step Bellman: K-step sampled-walk targets")
    ap.add_argument("--bellw", type=int, default=8, help="walks per pair for bellwalk")
    ap.add_argument("--resprox", type=int, default=0, help="predict d - proxy (exact factorized lower bound carries scale)")
    ap.add_argument("--logd", type=int, default=0, help="regress log1p(d), invert at eval")
    ap.add_argument("--softcount", type=int, default=0, help="readout = soft count of not-yet-arrived steps")
    ap.add_argument("--stepcap", type=float, default=0.0, help="soft per-factor step-norm cap penalty weight")
    ap.add_argument("--capval", type=float, default=1.5, help="cap threshold for --stepcap")
    ap.add_argument("--proxybudget", type=float, default=0.0, help="eval budget per pair = ceil(alpha * proxy)")
    ap.add_argument("--tri", type=float, default=0.0, help="soft triangle-inequality regularizer on random triples")
    ap.add_argument("--bounds", type=float, default=0.0, help="hinge to [proxy, greedy-upper] on unlabeled far pairs")
    ap.add_argument("--boundpairs", type=int, default=4000)
    ap.add_argument("--distill", type=int, default=0, help="stitching self-distillation rounds (radius doubles per round)")
    ap.add_argument("--waypoint_eval", type=int, default=0, help="eval far pairs by Dijkstra over sampled waypoints")
    ap.add_argument("--linmix", type=int, default=0, help="linear head over [flow cost, proxy features]")
    ap.add_argument("--ematarget", type=int, default=0, help="bellman: bootstrap from an EMA shadow model (TD stabilizer)")
    ap.add_argument("--emam", type=float, default=0.995)
    ap.add_argument("--d", type=int, default=128); ap.add_argument("--layers", type=int, default=4); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=40000); ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--poolq", type=int, default=2000); ap.add_argument("--nquery", type=int, default=80); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enc", choices=["factored", "bmask", "marker"], default="factored")
    ap.add_argument("--inject", type=int, default=0, help="per-agent constraint injection into position tokens")
    ap.add_argument("--heldout", choices=["", "combo", "links2", "dofhi", "dofhi2"], default="")
    a = ap.parse_args()
    global ENC, INJECT, HELDMODE, _HELD, MAXN; ENC = a.enc; INJECT = bool(a.inject); HELDMODE = a.heldout; N = a.nag
    MAXN = a.maxnodes
    if a.heldout == "combo":
        _HELD = {((0, 0, 0), 1), ((1, 1, 1), 3), ((3, 0, 0), 2), ((0, 2, 0), 4), ((2, 2, 2), 7), ((1, 0, 2), 5)}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    S1, S2, D = build_pool(N, a.G, a.Rtrain or a.Rmax, a.poolq, rng)                  # training ball capped at Rtrain
    S1t, S2t, Dt = (torch.as_tensor(x, device=dev) for x in (S1, S2, D))
    print(f"nag={N} G={a.G} enc={a.enc} heldout={a.heldout} Rtrain={a.Rtrain or a.Rmax} pool={len(S1)} maxd={int(D.max())}", flush=True)
    if a.linmix:
        model = LinMix(N, a.G, a.d, a.heads, a.layers, a.T).to(dev)
    elif a.arch in ("iqe", "mrn"):
        model = QmetHead(N, a.G, a.arch, a.d, a.heads, a.layers).to(dev)
    elif a.arch == "integ":
        model = Integrator(N, a.G, a.d, a.heads, a.layers, a.T).to(dev)
        if a.globalbase: model.gbase = nn.Parameter(torch.zeros((), device=dev))
        model.softcount = bool(a.softcount); model.capval = a.capval if a.stepcap > 0 else 0.0
    else:
        model = SymEmbed(N, a.G, a.arch[4:], a.d, a.heads, a.layers, a.T).to(dev)
    # target transforms: resprox trains the residual over the exact proxy; logd trains log1p(d)
    PXtr = torch.as_tensor(proxy_np(S1, S2, N, a.G), device=dev)
    def enc_t(dd, px): return (dd - px) if a.resprox else (torch.log1p(dd) if a.logd else dd)
    def dec_t(pp, px): return (pp + px) if a.resprox else (torch.expm1(pp) if a.logd else pp)
    opt = torch.optim.Adam(model.parameters(), a.lr)
    shadow = None
    if a.ematarget and a.bellman > 0:                             # EMA shadow model for bootstrap targets (TD stabilizer)
        import copy
        shadow = copy.deepcopy(model)
        for p in shadow.parameters(): p.requires_grad_(False)
    if a.bellman > 0:                                             # unlabeled pairs + neighbor/walk sets (no BFS labels)
        U0, UG, UN, UM = [], [], [], []
        nmax = 0; tmp = []
        for _ in range(a.bellpairs):
            s = _rand_state(N, a.G, rng); g = _rand_state(N, a.G, rng)
            if a.bellwalk > 0:                                    # K-step sampled-walk endpoints
                nb = []
                for _w in range(a.bellw):
                    cur = list(s)
                    for _k in range(a.bellwalk):
                        cn = neighbours(cur, N, a.G)
                        if not cn: break
                        cur = list(cn[rng.integers(len(cn))])
                    nb.append(np.array(cur))
            else:
                nb = neighbours(list(s), N, a.G)
            if not nb or tuple(s) == tuple(g): continue
            tmp.append((s, g, nb)); nmax = max(nmax, len(nb))
        for s, g, nb in tmp:
            U0.append(s); UG.append(g)
            UN.append(np.stack(list(nb) + [nb[0]] * (nmax - len(nb))))
            UM.append([1.0] * len(nb) + [0.0] * (nmax - len(nb)))
        U0t, UGt = torch.as_tensor(np.array(U0), device=dev), torch.as_tensor(np.array(UG), device=dev)
        UNt = torch.as_tensor(np.array(UN), device=dev); UMt = torch.as_tensor(np.array(UM), device=dev)
        UPx = torch.as_tensor(proxy_np(np.array(U0), np.array(UG), N, a.G), device=dev)
        print(f"bellman pool {len(U0)} nmax {nmax} kstep {max(1, a.bellwalk)}", flush=True)
    if a.tri > 0:                                                 # random state universe for triangle triples
        TRI = torch.as_tensor(np.array([_rand_state(N, a.G, rng) for _ in range(6000)]), device=dev)
    if a.bounds > 0:                                              # unlabeled far pairs with [proxy, greedy-upper] bounds
        BL0, BLG, BLlo, BLhi = [], [], [], []
        for _ in range(a.boundpairs):
            s = _rand_state(N, a.G, rng); g = _rand_state(N, a.G, rng)
            lo = float(proxy_np(np.array(s)[None], np.array(g)[None], N, a.G)[0])
            up = greedy_upper(s, g, N, a.G)
            if up is None: continue
            BL0.append(s); BLG.append(g); BLlo.append(lo); BLhi.append(float(up))
        BL0t = torch.as_tensor(np.array(BL0), device=dev); BLGt = torch.as_tensor(np.array(BLG), device=dev)
        BLlot = torch.as_tensor(np.array(BLlo, np.float32), device=dev); BLhit = torch.as_tensor(np.array(BLhi, np.float32), device=dev)
        print(f"bounds pool {len(BL0)} mean_gap {float((BLhit - BLlot).mean()):.2f}", flush=True)
    parking = a.arch == "integ" and (a.arrive > 0 or a.Tmin >= 0)
    for step in range(a.steps):
        b = torch.randint(0, len(S1t), (a.bs,), device=dev)
        if parking:                                                # anytime budget + arrival loss (parking recipe)
            Trun = int(torch.randint(max(1, a.Tmin), a.T + 1, (1,)).item()) if a.Tmin >= 0 else None
            cost, arr = model(S1t[b], S2t[b], Trun=Trun, ret_arr=True)
            loss = F.smooth_l1_loss(cost, enc_t(Dt[b], PXtr[b])) + a.arrive * arr.mean()
        else:
            loss = F.smooth_l1_loss(model(S1t[b], S2t[b]), enc_t(Dt[b], PXtr[b]))
        if a.stepcap > 0 and hasattr(model, "cappen"): loss = loss + a.stepcap * model.cappen
        if a.bellman > 0:                                         # k + min over neighbors/walk-ends, bootstrap target
            tgt_model = shadow if shadow is not None else model
            ub = torch.randint(0, len(U0t), (a.bellbs,), device=dev)
            with torch.no_grad():
                nb = UNt[ub]; B_, K_, _ = nb.shape
                px_n = None
                dn_raw = tgt_model(nb.reshape(B_ * K_, -1), UGt[ub].repeat_interleave(K_, 0)).reshape(B_, K_)
                if a.resprox or a.logd:                            # decode to distance space for the min
                    pxn = torch.as_tensor(proxy_np(nb.reshape(B_ * K_, -1).cpu().numpy(),
                                                   UGt[ub].repeat_interleave(K_, 0).cpu().numpy(), N, a.G),
                                          device=dev).reshape(B_, K_)
                    dn_raw = dec_t(dn_raw, pxn)
                dn = torch.where(UMt[ub] > 0, dn_raw, torch.full_like(dn_raw, 1e9))
                tgt = max(1, a.bellwalk) + dn.min(-1).values
            loss = loss + a.bellman * F.smooth_l1_loss(model(U0t[ub], UGt[ub]), enc_t(tgt, UPx[ub]))
        if a.tri > 0:                                              # soft triangle inequality on random triples
            ti = torch.randint(0, len(TRI), (3, a.bellbs), device=dev)
            A_, B2_, C_ = TRI[ti[0]], TRI[ti[1]], TRI[ti[2]]
            p3 = model(torch.cat([A_, A_, B2_]), torch.cat([C_, B2_, C_]))
            if a.resprox or a.logd:
                px3 = torch.as_tensor(proxy_np(torch.cat([A_, A_, B2_]).cpu().numpy(),
                                               torch.cat([C_, B2_, C_]).cpu().numpy(), N, a.G), device=dev)
                p3 = dec_t(p3, px3)
            dac, dab, dbc = p3.chunk(3)
            loss = loss + a.tri * F.relu(dac - dab - dbc).mean()
        if a.bounds > 0:                                           # hinge into [proxy, greedy-upper] on far pairs
            bb2 = torch.randint(0, len(BL0t), (a.bellbs,), device=dev)
            pb = model(BL0t[bb2], BLGt[bb2])
            if a.resprox or a.logd:
                pb = dec_t(pb, torch.as_tensor(proxy_np(BL0t[bb2].cpu().numpy(), BLGt[bb2].cpu().numpy(), N, a.G), device=dev))
            loss = loss + a.bounds * (F.relu(BLlot[bb2] - pb) + F.relu(pb - BLhit[bb2])).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if shadow is not None:
            with torch.no_grad():
                for ps, pm in zip(shadow.parameters(), model.parameters()):
                    ps.mul_(a.emam).add_(pm, alpha=1 - a.emam)
        if step % (a.steps // 6) == 0:
            with torch.no_grad():
                bb = torch.randint(0, len(S1t), (2000,), device=dev)
                mae = (dec_t(model(S1t[bb], S2t[bb]), PXtr[bb]) - Dt[bb]).abs().mean().item()
            print(f"step {step} loss {loss.item():.3f} trainMAE {mae:.3f}", flush=True)
    for r in range(a.distill):                                     # stitching: compose trusted legs, double the radius
        trusted = (a.Rtrain or a.Rmax) * (2 ** r)
        Fs = np.array([_rand_state(N, a.G, rng) for _ in range(4000)])
        Fg = np.array([_rand_state(N, a.G, rng) for _ in range(4000)])
        pxf = proxy_np(Fs, Fg, N, a.G); keep = (pxf > trusted * 0.6) & (pxf < trusted * 2.0)
        Fs, Fg = Fs[keep], Fg[keep]
        if not len(Fs): break
        Ms = np.array([_rand_state(N, a.G, rng) for _ in range(96)])
        Fst, Fgt, Mst = (torch.as_tensor(x, device=dev) for x in (Fs, Fg, Ms))
        with torch.no_grad():
            def legs(Aa, Bb):                                      # (P, M) predicted distances, decoded
                P_, M_ = len(Aa), len(Bb); out = torch.zeros(P_, M_, device=dev)
                for i0 in range(0, P_ * M_, 8192):
                    idx = torch.arange(i0, min(i0 + 8192, P_ * M_), device=dev)
                    ai, bi = idx // M_, idx % M_
                    pv = model(Aa[ai], Bb[bi])
                    pv = dec_t(pv, torch.as_tensor(proxy_np(Aa[ai].cpu().numpy(), Bb[bi].cpu().numpy(), N, a.G), device=dev))
                    out.view(-1)[idx] = pv
                return out
            L1 = legs(Fst, Mst)                                    # (P, M): s -> midpoint
            L2 = legs(Mst, Fgt).T                                  # (P, M): midpoint -> g
            tot = L1 + L2
            bad = (L1 > trusted) | (L2 > trusted)                  # only compose trusted legs
            tot = torch.where(bad, torch.full_like(tot, 1e9), tot)
            lab = tot.min(-1).values
            ok = lab < 1e8
        Fst, Fgt, labt = Fst[ok], Fgt[ok], lab[ok]
        pxk = torch.as_tensor(proxy_np(Fst.cpu().numpy(), Fgt.cpu().numpy(), N, a.G), device=dev)
        print(f"distill round {r + 1}: trusted {trusted} pairs {len(Fst)}", flush=True)
        for st in range(a.steps // 4):
            b = torch.randint(0, len(S1t), (a.bs // 2,), device=dev)
            fb = torch.randint(0, len(Fst), (a.bs // 2,), device=dev)
            loss = F.smooth_l1_loss(model(S1t[b], S2t[b]), enc_t(Dt[b], PXtr[b])) + \
                   F.smooth_l1_loss(model(Fst[fb], Fgt[fb]), enc_t(labt[fb], pxk[fb]))
            opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    if a.arch == "integ" and a.Ttest > 0: model.T = a.Ttest        # budget-free eval: long budget + parking

    @torch.no_grad()
    def evaluate(part):
        rp = np.random.default_rng(a.seed + 7 + (0 if part != "train" else 91))
        preds, trues, aucs = [], [], []
        for _ in range(a.nquery):
            dof = int(rp.integers(2, 2 * N + 1))
            if HELDMODE in ("dofhi", "dofhi2"):
                thr = 7 if HELDMODE == "dofhi" else 6; dof = int(rp.integers(thr, 2 * N + 1)) if part == "test" else int(rp.integers(2, thr))
            q = make_state(N, a.G, rp, part=part, dof=dof); mk, lk = int(q[N]), int(q[N + 1])
            dist = bfs_local(q, N, a.G, a.Rmax)
            items = [(np.array(tv), dv) for tv, dv in dist.items() if dv > 0]
            if len(items) > 200: items = [items[j] for j in rp.choice(len(items), 200, replace=False)]
            tg = np.array([t for t, _ in items]); td = np.array([d for _, d in items], float)
            qrep = np.repeat(q[None], len(tg), 0)
            pxb = proxy_np(qrep, tg, N, a.G)
            qbt, tgt_ = torch.as_tensor(qrep, device=dev), torch.as_tensor(tg, device=dev)
            Tper = None
            if a.proxybudget > 0 and a.arch == "integ" and not a.linmix:
                Tper = torch.as_tensor(np.clip(np.ceil(a.proxybudget * pxb), 4, 6 * a.T), device=dev)
            raw = model(qbt, tgt_, Tper=Tper) if Tper is not None else model(qbt, tgt_)
            pr = dec_t(raw, torch.as_tensor(pxb, device=dev)).cpu().numpy()
            if a.waypoint_eval:
                trusted = float(a.Rtrain or a.Rmax)
                if not hasattr(evaluate, "_W"):
                    Wnp = np.array([_rand_state(N, a.G, np.random.default_rng(a.seed + 5)) for _ in range(128)])
                    Wt = torch.as_tensor(Wnp, device=dev)
                    ii = np.repeat(np.arange(128), 128); jj = np.tile(np.arange(128), 128)
                    dwr = model(Wt[ii], Wt[jj])
                    dw = dec_t(dwr, torch.as_tensor(proxy_np(Wnp[ii], Wnp[jj], N, a.G), device=dev)).cpu().numpy().reshape(128, 128)
                    dw[dw > trusted] = 1e9; np.fill_diagonal(dw, 0)
                    for k in range(128): dw = np.minimum(dw, dw[:, k:k + 1] + dw[k:k + 1, :])   # Floyd-Warshall
                    evaluate._W, evaluate._dw = (Wnp, Wt), dw
                (Wnp, Wt), dw = evaluate._W, evaluate._dw
                dqr = model(torch.as_tensor(np.repeat(q[None], 128, 0), device=dev), Wt)
                dq = dec_t(dqr, torch.as_tensor(proxy_np(np.repeat(q[None], 128, 0), Wnp, N, a.G), device=dev)).cpu().numpy()
                dq[dq > trusted] = 1e9
                for k in range(len(tg)):
                    if pxb[k] <= trusted: continue                  # near pairs: keep direct prediction
                    dtr = model(Wt, torch.as_tensor(np.repeat(tg[k][None], 128, 0), device=dev))
                    dt = dec_t(dtr, torch.as_tensor(proxy_np(Wnp, np.repeat(tg[k][None], 128, 0), N, a.G), device=dev)).cpu().numpy()
                    dt[dt > trusted] = 1e9
                    comp = float(np.min(dq[:, None] + dw + dt[None, :]))
                    if comp < 1e8: pr[k] = min(pr[k], comp)
            preds += list(pr); trues += list(td)
            legal = [list(v) for v in neighbours(list(q), N, a.G) if v[N] == mk and v[N + 1] == lk]   # true 1-step moves
            legset = {tuple(v) for v in legal}; ills = []
            for i in range(N):                                                                        # single-agent moves that break coupling/mobility
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    r, c = q[i] // a.G, q[i] % a.G; nr, nc = r + dr, c + dc
                    if 0 <= nr < a.G and 0 <= nc < a.G:
                        s = q.copy(); s[i] = nr * a.G + nc
                        if tuple(s) not in legset: ills.append(list(s))
            if legal and ills:
                ql, qi = np.repeat(q[None], len(legal), 0), np.repeat(q[None], len(ills), 0)
                dl = dec_t(model(torch.as_tensor(ql, device=dev), torch.as_tensor(np.array(legal), device=dev)),
                           torch.as_tensor(proxy_np(ql, np.array(legal), N, a.G), device=dev)).cpu().numpy()
                di = dec_t(model(torch.as_tensor(qi, device=dev), torch.as_tensor(np.array(ills), device=dev)),
                           torch.as_tensor(proxy_np(qi, np.array(ills), N, a.G), device=dev)).cpu().numpy()
                aucs.append(float(np.mean([[1.0 if x < y else 0.0 for y in di] for x in dl])))
        preds, trues = np.array(preds), np.array(trues)
        mae = round(float(np.abs(preds - trues).mean()), 3)
        corr = round(float(np.corrcoef(preds, trues)[0, 1]), 3) if preds.std() > 1e-6 else None
        px = round(float(np.mean(aucs)), 3) if aucs else None
        out = dict(dist_mae=mae, dist_corr=corr, px_auc=px)
        if a.Rtrain and a.Rtrain < a.Rmax:                                            # length-extrapolation breakdown
            far = trues > a.Rtrain
            if far.any():
                out["mae_beyond"] = round(float(np.abs(preds[far] - trues[far]).mean()), 3)
                out["corr_beyond"] = round(float(np.corrcoef(preds[far], trues[far])[0, 1]), 3) if far.sum() > 2 and preds[far].std() > 1e-6 else None
            if (~far).any(): out["mae_within"] = round(float(np.abs(preds[~far] - trues[~far]).mean()), 3)
        return out

    res = dict(nag=N, G=a.G, enc=a.enc, arch=a.arch, globalbase=a.globalbase, inject=a.inject, heldout=a.heldout,
               Rtrain=a.Rtrain, Rmax=a.Rmax, steps=a.steps, seed=a.seed, d=a.d, layers=a.layers, T=a.T,
               variant={k: getattr(a, k) for k in ("resprox", "logd", "softcount", "stepcap", "proxybudget",
                                                   "bellman", "bellwalk", "tri", "bounds", "distill", "ematarget",
                                                   "waypoint_eval", "linmix", "arrive", "Tmin", "Ttest") if getattr(a, k)})
    res["test" if a.heldout else "all"] = evaluate("test" if a.heldout else "all")     # generalization (held-out)
    if a.heldout: res["train"] = evaluate("train")                                     # in-distribution accuracy
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()

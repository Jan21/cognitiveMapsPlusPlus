"""Distance model on the SWITCHYARD environment — readout ablation.

Same model family and training protocol as the coupling-gridworld `integ_distance.py`; only the
environment and its encoder change. The environment itself is untouched: `switchyard_env.Yard` supplies
the transition function and the exact BFS geodesics.

State is (worker_cell, crate_cell, gate_bits) on a per-instance MAP (walls, gate cells, lever cells and
their XOR wiring, plate cell and mask, chute cells and directions). Unlike the old world the map VARIES
between instances, so it has to be encoded rather than assumed - that is exactly what the `map` and
`wire` splits test.

Tokens
  dynamic (these accumulate displacement):  worker, crate, one per gate            -> n = 2 + D
  context (re-injected, never accumulated): one per lever, plate, one per chute
Sequence per step: [ dynamic(state) | dynamic(goal) | dynamic(start) | context ]

--readout matches integ_distance.py: integrate | scalar | scalar_mlp | iqe | mrn_fixed | sym_embed | concat_mlp
--split   map  : test maps' layouts AND wirings never seen in training
          wire : same layouts as training, lever/plate wiring resampled
          none : same maps, held-out state pairs only
--Rtrain R      : cap TRAINING pair distance at R; evaluation keeps the full range (length extrapolation)
"""
import argparse, collections, json, os, sys, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from switchyard_env import Yard

READOUT, POOL = "integrate", "trunk"
GLOBALBASE = 0
MERGEGATES = False   # one pooled gate-state token instead of one token per gate
SEEWALLS = False     # expose the per-map wall openings to the encoder
PROGT = 0            # >0: sample iteration count from [PROGTMIN, PROGT] during training
PROGTMIN = 1
PADSLOTS = 0         # extra dynamic slots carrying no entity, to test MORE slots than factors
RECURRENT = {"integrate", "scalar", "scalar_mlp"}


# ---------------------------------------------------------------------------- data
def make_maps(n, G, ngate, nlever, nchute, seed, wire_seed=None, lad=None):
    """Layout rng and wiring rng are separate so wiring can be resampled on the SAME layouts.
    `lad` carries the complexity-ladder flags (wire1/noplate/nopush/gatesopen)."""
    lad = lad or {}
    return [Yard(G, ngate, nlever, nchute,
                 rng=np.random.default_rng(seed * 1000 + i),
                 wire_rng=np.random.default_rng((wire_seed if wire_seed is not None else seed) * 7717 + i),
                 **lad)
            for i in range(n)]


def bfs_pulls(y, src, maxnodes=60000):
    """Like Yard.bfs, but also carries how many PULL actions an optimal path used.

    PULL is the only action that rewires gates, so its count is exactly the DEPENDENCY complexity of a
    pair: how many times the route has to stop and reconfigure the world. BFS is level-order, so the
    first arrival at a state is on a shortest path and the pull count recorded there belongs to one such
    path. The env file is untouched -- this only reads y.neighbours(..., with_actions=True).
    """
    dist = {src: 0}; pull = {src: 0}; dq = collections.deque([src])
    while dq and len(dist) < maxnodes:
        u = dq.popleft()
        for v, act in y.neighbours(u, with_actions=True):
            if v not in dist:
                dist[v] = dist[u] + 1; pull[v] = pull[u] + (1 if act == "PULL" else 0)
                dq.append(v)
    return dist, pull


def build_pool(maps, nq_per_map, Rmax, rng, per_bucket=40, fixpull=-1):
    """(state, goal, distance) triples, bucket-sampled over distance, from every map.

    With fixpull >= 0 only pairs whose optimal route uses EXACTLY that many pulls are kept, so the
    dependency complexity is held constant and distance varies for one reason only: walking further.
    """
    S1, S2, D, MI = [], [], [], []
    for mi, y in enumerate(maps):
        for _ in range(nq_per_map):
            src = y.rand_state(rng)
            if fixpull >= 0:
                dist, pull = bfs_pulls(y, src, maxnodes=60000)
                dist = {k: v for k, v in dist.items() if pull[k] == fixpull}
            else:
                dist = y.bfs(src, maxnodes=60000)
            byd = collections.defaultdict(list)
            for st, dv in dist.items():
                if 0 < dv <= Rmax: byd[dv].append(st)
            if not byd: continue
            per = max(1, per_bucket // max(1, len(byd)))
            for dv, lst in byd.items():
                idx = rng.choice(len(lst), min(per, len(lst)), replace=False)
                for j in idx:
                    S1.append(src); S2.append(lst[j]); D.append(dv); MI.append(mi)
    return (np.array(S1, np.int64), np.array(S2, np.int64),
            np.array(D, np.float32), np.array(MI, np.int64))


def map_tensors(maps, G):
    """Static per-map descriptors, padded to a common shape, as integer tensors."""
    ng = max(len(y.gates) for y in maps); nl = max(len(y.levers) for y in maps)
    nc = max(len(y.chutes) for y in maps)
    gate_c = np.zeros((len(maps), ng), np.int64); lev_c = np.zeros((len(maps), nl), np.int64)
    lev_w = np.zeros((len(maps), nl), np.int64); plate_c = np.zeros(len(maps), np.int64)
    plate_m = np.zeros(len(maps), np.int64); chute_c = np.zeros((len(maps), nc), np.int64)
    chute_d = np.zeros((len(maps), nc), np.int64)
    # The 13 cells of the wall cross are the only ones that can be wall-or-open, and WHICH of them
    # are open is resampled per map. Only the 3 that become gates were ever encoded, leaving ~2.3
    # passable doorways per map invisible to the model. These are those doorways.
    def _openings(y):
        wr = wc = y.G // 2
        return [r * y.G + c for r in range(y.G) for c in range(y.G)
                if (r == wr or c == wc) and not y.wall[r, c] and (r, c) not in set(y.gates)]
    no = max(1, max(len(_openings(y)) for y in maps))
    open_c = np.zeros((len(maps), no), np.int64); open_m = np.zeros((len(maps), no), np.float32)
    for i, y in enumerate(maps):
        for j, cell in enumerate(_openings(y)): open_c[i, j] = cell; open_m[i, j] = 1.0
    for i, y in enumerate(maps):
        for j, rc in enumerate(y.gates): gate_c[i, j] = rc[0] * G + rc[1]
        for j, rc in enumerate(y.levers): lev_c[i, j] = rc[0] * G + rc[1]; lev_w[i, j] = y.wiring[j]
        plate_c[i] = y.plate[0] * G + y.plate[1]; plate_m[i] = y.platemask
        for j, (rc, d) in enumerate(y.chutes.items()): chute_c[i, j] = rc[0] * G + rc[1]; chute_d[i, j] = d
    return dict(gate_c=gate_c, lev_c=lev_c, lev_w=lev_w, plate_c=plate_c,
                plate_m=plate_m, chute_c=chute_c, chute_d=chute_d,
                open_c=open_c, open_m=open_m, ng=ng, nl=nl, nc=nc, no=no)


# ---------------------------------------------------------------------------- model
class Block(nn.Module):
    def __init__(self, d, heads, layers):
        super().__init__()
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0,
                                     activation="gelu", batch_first=True, norm_first=True)
                                     for _ in range(layers)])

    def forward(self, z):
        for l in self.layers: z = l(z)
        return z


class Enc(nn.Module):
    """worker + crate + one token per gate  (dynamic);  levers, plate, chutes  (static context)."""

    def __init__(self, G, D, MT, d):
        super().__init__()
        self.G, self.D, self.MT, self.d = G, D, MT, d
        # By default every gate gets its own displacement channel (2 + D tokens). With MERGEGATES the
        # D gate tokens are summed into ONE, giving 3 dynamic slots: worker, crate, gate-state. The
        # environment is unchanged -- this only alters how many slots the readout accumulates over.
        self.ndyn = 2 + (1 if MERGEGATES else D) + PADSLOTS
        # PADSLOTS adds dynamic tokens bound to no entity. They accumulate displacement like any other
        # slot, so this asks whether the readout wants MORE channels than the environment has factors.
        # Raising --ngate would change the TASK; this changes only the model.
        if PADSLOTS:
            self.padbase = nn.Parameter(torch.randn(PADSLOTS, d) * 0.02)
        self.cellf = nn.Embedding(G * G, d)                 # a cell, anywhere on the map
        self.kind = nn.Embedding(6, d)                      # worker | crate | gate | lever | plate | chute
        self.bit = nn.Embedding(2, d)                       # gate open / closed
        self.mask = nn.Embedding(1 << D, d)                 # a wiring mask (lever or plate)
        self.dir = nn.Embedding(4, d)                       # chute direction
        if SEEWALLS:                                        # constructed last: flag-off is unchanged
            self.openkind = nn.Parameter(torch.randn(d) * 0.02)
            self.openabs = nn.Parameter(torch.randn(d) * 0.02)   # marker for a padded slot

    def dynamic(self, cellw, cellc, bits, mi):
        """(B,) tensors -> (B, 2+D, d)."""
        B = cellw.shape[0]
        gate_c = self.MT["gate_c"][mi]                                        # (B, D)
        toks = [self.cellf(cellw) + self.kind.weight[0],
                self.cellf(cellc) + self.kind.weight[1]]
        gt = [self.cellf(gate_c[:, g]) + self.kind.weight[2] + self.bit((bits >> g) & 1)
              for g in range(self.D)]
        if MERGEGATES:
            toks.append(torch.stack(gt, 0).sum(0))       # all gates share one slot
        else:
            toks.extend(gt)                              # one slot per gate
        for k in range(PADSLOTS):                        # spare slots, no entity attached
            toks.append(self.padbase[k][None].expand(B, self.d))
        return torch.stack(toks, 1)

    def context(self, mi):
        B = mi.shape[0]
        lev_c, lev_w = self.MT["lev_c"][mi], self.MT["lev_w"][mi]
        toks = [self.cellf(lev_c[:, j]) + self.kind.weight[3] + self.mask(lev_w[:, j])
                for j in range(lev_c.shape[1])]
        toks.append(self.cellf(self.MT["plate_c"][mi]) + self.kind.weight[4]
                    + self.mask(self.MT["plate_m"][mi]))
        ch_c, ch_d = self.MT["chute_c"][mi], self.MT["chute_d"][mi]
        for j in range(ch_c.shape[1]):
            toks.append(self.cellf(ch_c[:, j]) + self.kind.weight[5] + self.dir(ch_d[:, j]))
        if SEEWALLS:
            op_c, op_m = self.MT["open_c"][mi], self.MT["open_m"][mi]
            for j in range(op_c.shape[1]):
                mk = op_m[:, j:j + 1]
                tok = self.cellf(op_c[:, j]) + self.openkind[None]
                toks.append(mk * tok + (1 - mk) * self.openabs[None])
        return torch.stack(toks, 1)


class Model(nn.Module):
    def __init__(self, G, D, MT, d=128, heads=4, layers=4, T=14):
        super().__init__()
        self.enc = Enc(G, D, MT, d); self.n = self.enc.ndyn; self.T = T
        self.fid = nn.Embedding(self.n, d); self.role = nn.Embedding(3, d)
        self.block = Block(d, heads, layers); self.scale = nn.Parameter(torch.zeros(()))
        self.gbase = nn.Parameter(torch.zeros(())) if GLOBALBASE else None
        # ---- readout-specific modules constructed LAST so the shared trunk's RNG draw is variant-invariant
        self.readout, self.pool, self.head, self.proj = READOUT, POOL, None, None
        if READOUT == "scalar":       self.head = nn.Linear(d, 1)
        elif READOUT == "scalar_mlp": self.head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        elif READOUT == "concat_mlp": self.head = nn.Sequential(nn.Linear(2 * self.n * d, d), nn.GELU(),
                                                                nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        elif READOUT in ("iqe", "mrn_fixed"):
            import torchqmet
            self.head = (torchqmet.IQE(input_size=d, dim_per_component=16) if READOUT == "iqe"
                         else torchqmet.MRNFixed(input_size=d, sym_p=1))
        if READOUT in ("iqe", "mrn_fixed", "sym_embed"):
            self.proj = nn.Linear(self.n * d, d)

    def emb(self, s, mi, role):
        z = self.enc.dynamic(s[:, 0], s[:, 1], s[:, 2], mi)
        ids = torch.arange(self.n, device=s.device)
        return z + self.fid(ids)[None] + self.role(torch.tensor(role, device=s.device))

    def latent(self, s, mi):
        z = self.emb(s, mi, 0)
        if self.pool == "trunk": z = self.block(z)
        elif self.pool == "rtrunk":                      # EQUAL COMPUTE: the same T applications the
            for _ in range(self.T): z = self.block(z)    # integrate readout gets, then pool
        return self.proj(z.flatten(1))

    def forward(self, s_, g, mi, ret_arr=False, ret_unit=False):
        s = s_
        n = self.n
        if self.readout in ("iqe", "mrn_fixed"): return self.head(self.latent(s, mi), self.latent(g, mi))
        if self.readout == "sym_embed": return (self.latent(s, mi) - self.latent(g, mi)).abs().sum(-1)
        ctx = self.enc.context(mi)
        if self.readout == "concat_mlp":
            h = torch.cat([self.emb(s, mi, 0).flatten(1), self.emb(g, mi, 1).flatten(1)], -1)
            return F.softplus(self.head(h)).squeeze(-1)
        base = torch.cat([self.emb(g, mi, 1), self.emb(s, mi, 2), ctx], 1)
        tok = torch.cat([self.emb(s, mi, 0), base], 1)
        cost = torch.zeros(s.shape[0], device=s.device)
        acc = self.readout == "integrate"
        # PROGT: during training, sample the number of iterations. Without this the model learns a
        # depth-specific routine and degrades when run longer ("overthinking"); with it, iteration
        # count becomes something that can be turned up at test time.
        per_step = []
        steps = self.T
        if self.training and PROGT > 0:
            steps = int(torch.randint(max(1, PROGTMIN), PROGT + 1, (1,)).item())
        for _ in range(steps):
            z = self.block(tok)
            if acc:
                dstep = (z[:, :n] - tok[:, :n]).norm(dim=-1).sum(-1)
                cost = cost + dstep
                per_step.append(dstep)
            tok = torch.cat([z[:, :n], base], 1)
        if acc:
            out = F.softplus(self.scale) * cost
            out = out + self.gbase if self.gbase is not None else out
        else:
            out = F.softplus(self.head(tok[:, :n].mean(1))).squeeze(-1)
        if ret_unit:
            # UNIT-STEP: every recurrent step should advance the same amount, so the readout becomes
            # a distance FIELD (distance = steps x unit) instead of a saturating quantity. Without
            # this the model stops moving once it has accumulated the training range and emits a
            # constant for anything longer. Penalise the spread of per-step displacement.
            P = torch.stack(per_step, 1) if per_step else None
            unit = P.std(dim=1).mean() / (P.mean() + 1e-6) if P is not None and P.shape[1] > 1 \
                else torch.zeros((), device=s_.device)
            arr = (tok[:, :n] - self.emb(g, mi, 0)).norm(dim=-1).mean(-1)
            return out, arr, unit
        if ret_arr:
            # "park at goal": after the recurrence the dynamic tokens should BE the goal state,
            # encoded in the same role the start was. Together with progT (run for a random number of
            # steps) this forces the recurrence to be an actual traversal that halts on arrival,
            # which is what makes the step count a usable dial for longer routes.
            arr = (tok[:, :n] - self.emb(g, mi, 0)).norm(dim=-1).mean(-1)
            return out, arr
        return out


# ---------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--G", type=int, default=7); ap.add_argument("--ngate", type=int, default=3)
    ap.add_argument("--nlever", type=int, default=2); ap.add_argument("--nchute", type=int, default=1)
    ap.add_argument("--nmaps", type=int, default=24); ap.add_argument("--ntest_maps", type=int, default=8)
    ap.add_argument("--nq", type=int, default=40, help="query states per map")
    ap.add_argument("--Rmax", type=int, default=24); ap.add_argument("--Rtrain", type=int, default=0)
    ap.add_argument("--d", type=int, default=128); ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4); ap.add_argument("--T", type=int, default=14)
    ap.add_argument("--steps", type=int, default=40000); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--readout", default="integrate",
                    choices=["integrate", "scalar", "scalar_mlp", "iqe", "mrn_fixed", "sym_embed", "concat_mlp"])
    ap.add_argument("--pool", default="trunk", choices=["trunk", "proj", "rtrunk"])
    ap.add_argument("--sched", choices=["const", "cosine"], default="const",
                    help="cosine = decay lr to 0 without restarts, the recipe the IQE paper uses")
    ap.add_argument("--fixpull", type=int, default=-1,
                    help="keep only pairs whose optimal route uses exactly this many PULLs, so the\ndependency complexity is constant and only the walking length grows (-1 = off)")
    ap.add_argument("--unitstep", type=float, default=0.0,
                    help="weight on the per-step uniformity penalty: coefficient of variation of the "
                         "displacement contributed by each recurrent step. Pushes the readout toward "
                         "a distance FIELD (distance = steps x unit), which is what lets extra "
                         "iterations cover longer routes. See Horizon Generalization, arXiv:2501.02709.")
    ap.add_argument("--arrive", type=float, default=0.0,
                    help="weight on the arrival term: after the recurrence the dynamic tokens should "
                         "equal the GOAL state. 0 = off. Pairs with --progT; together they train the "
                         "recurrence to traverse and halt, which is what lets extra iterations cover "
                         "longer routes (Jan's --arrive/--Tmin, Deep Thinking arXiv:2202.05826).")
    ap.add_argument("--seewalls", type=int, default=0,
                    help="give the model the wall openings it currently cannot see. The wall cross is "
                         "fixed but WHICH of its 13 cells are open is resampled per map, and only the "
                         "3 that become gates were encoded -- leaving ~2.3 passable doorways per map "
                         "invisible. Each hidden doorway shifts 35%% of distances by ~2.4 moves, so "
                         "exact prediction was information-theoretically impossible. 0 = off.")
    ap.add_argument("--progTmin", type=int, default=1,
                    help="lower bound of the sampled iteration count when --progT is on. Jan's --Tmin. "
                         "Without a floor, T=1 gets sampled and arriving at a distant goal in one step "
                         "is impossible, so the arrival loss fights the rollout (the instability we "
                         "saw). Default 1 = old behaviour.")
    ap.add_argument("--progT", type=int, default=0,
                    help="progressive iteration training (Deep Thinking, arXiv:2202.05826). Each "
                         "training step samples the iteration count uniformly from [1, progT] "
                         "instead of always using --T, so the model cannot learn behaviour tied to "
                         "one specific depth and the count becomes a usable dial at test time. "
                         "0 = off, always --T.")
    ap.add_argument("--gatesopen", type=int, default=-1,
                    help="env flag, overrides --level. 1 = gates pinned open, levers inert")
    ap.add_argument("--nopush", type=int, default=-1,
                    help="env flag, overrides --level. 1 = crate is a static obstacle")
    ap.add_argument("--wire1", type=int, default=-1,
                    help="env flag, overrides --level. 1 = each lever wired to exactly one gate")
    ap.add_argument("--noplate", type=int, default=-1,
                    help="env flag, overrides --level. 1 = pressure plate disabled")
    ap.add_argument("--level", type=int, default=-1,
                    help="complexity ladder rung 0-5 (Jan's L0-L5); overrides the ladder flags below")
    ap.add_argument("--globalbase", type=int, default=0,
                    help="add one learned constant to the prediction (his length-generalisation fix)")
    ap.add_argument("--split", default="map", choices=["map", "wire", "none"])
    ap.add_argument("--save", default="")
    ap.add_argument("--mergegates", action="store_true",
                    help="pool the per-gate tokens into ONE gate-state slot: 3 dynamic slots "
                         "(worker, crate, gates) instead of 2+ngate. The environment is unchanged; "
                         "only the number of slots the readout accumulates over changes.")
    ap.add_argument("--padslots", type=int, default=0,
                    help="add N extra dynamic slots bound to no entity, to test MORE slots than the "
                         "environment has factors (5 + N). Changes the model only, not the task.")
    ap.add_argument("--evalevery", type=int, default=0,
                    help="evaluate train AND held-out MAE every N steps and record a curve. "
                         "0 = off (bit-identical to before). Also enables best-on-held-out "
                         "checkpointing to <save>.best.pt")
    ap.add_argument("--ckptevery", type=int, default=0,
                    help="also dump a rolling checkpoint every N steps to <save>_ckpt/stepNNNNNNN.pt. "
                         "0 = off. Requires --save")
    a = ap.parse_args()
    global READOUT, POOL; READOUT, POOL = a.readout, a.pool
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)

    # L0 = plain maze + static crate ... L5 = full dependency web (matches his published rungs)
    LADDER = {0: dict(gatesopen=True, nopush=True), 1: dict(gatesopen=True),
              2: dict(wire1=True, noplate=True), 3: dict(noplate=True), 4: {}, 5: {}}
    lad = {}
    if a.level >= 0:
        lad = dict(LADDER[a.level])
        if a.level < 5: a.nchute = 0                 # the one-way chute is what L5 adds
    # explicit env flags override the level preset, so single mechanics can be isolated.
    # Yard takes each as a bool kwarg defaulting to False, so passing False is a no-op.
    for _f in ("gatesopen", "nopush", "wire1", "noplate"):
        _v = getattr(a, _f)
        if _v >= 0:
            lad[_f] = bool(_v)
    global GLOBALBASE, MERGEGATES, PADSLOTS, PROGT, PROGTMIN, SEEWALLS
    SEEWALLS = bool(a.seewalls)
    PROGT = a.progT; PROGTMIN = a.progTmin
    GLOBALBASE = a.globalbase; MERGEGATES = a.mergegates; PADSLOTS = a.padslots
    tr_maps = make_maps(a.nmaps, a.G, a.ngate, a.nlever, a.nchute, seed=a.seed, lad=lad)
    if a.split == "map":                       # unseen layouts AND wirings
        te_maps = make_maps(a.ntest_maps, a.G, a.ngate, a.nlever, a.nchute, seed=a.seed + 991, lad=lad)
    elif a.split == "wire":                    # same layouts, wiring resampled
        te_maps = make_maps(a.nmaps, a.G, a.ngate, a.nlever, a.nchute, seed=a.seed, wire_seed=a.seed + 991, lad=lad)
        te_maps = te_maps[:a.ntest_maps]
    else:
        te_maps = tr_maps[:a.ntest_maps]
    all_maps = tr_maps + te_maps
    MTn = map_tensors(all_maps, a.G)
    MT = {k: (torch.as_tensor(v, device=dev) if isinstance(v, np.ndarray) else v) for k, v in MTn.items()}

    Rtr = a.Rtrain or a.Rmax
    S1, S2, D, MI = build_pool(tr_maps, a.nq, Rtr, rng, fixpull=a.fixpull)
    eS1, eS2, eD, eMI = build_pool(te_maps, max(6, a.nq // 3), a.Rmax,
                                   np.random.default_rng(a.seed + 7), fixpull=a.fixpull)
    eMI = eMI + len(tr_maps)
    print(f"switchyard G={a.G} gates={a.ngate} maps={a.nmaps}/{len(te_maps)} split={a.split} "
          f"Rtrain={Rtr} Rmax={a.Rmax} train={len(S1)} test={len(eS1)} maxd={int(D.max())} "
          f"testmaxd={int(eD.max())}", flush=True)

    t = lambda x: torch.as_tensor(x, device=dev)
    S1t, S2t, Dt, MIt = t(S1), t(S2), t(D), t(MI)
    eS1t, eS2t, eDt, eMIt = t(eS1), t(eS2), t(eD), t(eMI)

    model = Model(a.G, a.ngate, MT, a.d, a.heads, a.layers, a.T).to(dev)
    nparam = sum(p.numel() for p in model.parameters())
    print(f"readout={a.readout} pool={a.pool} params={nparam} tokens={model.n}dyn+{MTn['nl']+1+MTn['nc']}ctx",
          flush=True)
    opt = torch.optim.Adam(model.parameters(), a.lr)
    sched = (torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
             if a.sched == "cosine" else None)
    @torch.no_grad()
    def quick_mae(s1, s2, dd, mi, n=3000):
        """Subsampled MAE, used for the during-training curve. Cheap enough to call often."""
        k = min(n, len(s1))
        idx = torch.arange(k, device=dev) if k == len(s1) else torch.randperm(len(s1), device=dev)[:k]
        model.eval()
        pr = torch.cat([model(s1[idx][i:i + 4096], s2[idx][i:i + 4096], mi[idx][i:i + 4096])
                        for i in range(0, k, 4096)])
        model.train()
        return float((pr - dd[idx]).abs().mean().item())

    curve = []                                   # [(step, train_mae, test_mae), ...]
    best = dict(test=float("inf"), step=-1)
    ck_dir = os.path.splitext(a.save)[0] + "_ckpt" if (a.save and a.ckptevery > 0) else ""
    if ck_dir: os.makedirs(ck_dir, exist_ok=True)

    t0 = time.time()
    for step in range(a.steps):
        b = torch.randint(0, len(S1t), (a.bs,), device=dev)
        if a.arrive > 0 or a.unitstep > 0:
            pred, arr, unit = model(S1t[b], S2t[b], MIt[b], ret_unit=True)
            loss = (F.smooth_l1_loss(pred, Dt[b]) + a.arrive * arr.mean()
                    + a.unitstep * unit)
        else:
            loss = F.smooth_l1_loss(model(S1t[b], S2t[b], MIt[b]), Dt[b])
        opt.zero_grad(); loss.backward(); opt.step()
        if sched is not None: sched.step()
        if step % max(1, a.steps // 6) == 0:
            with torch.no_grad():
                bb = torch.randint(0, len(S1t), (min(2000, len(S1t)),), device=dev)
                mae = (model(S1t[bb], S2t[bb], MIt[bb]) - Dt[bb]).abs().mean().item()
            print(f"step {step} loss {loss.item():.3f} trainMAE {mae:.3f}", flush=True)
        # ---- during-training curve: train AND held-out error, so convergence is visible ----
        if a.evalevery > 0 and (step % a.evalevery == 0 or step == a.steps - 1):
            tr_m, te_m = quick_mae(S1t, S2t, Dt, MIt), quick_mae(eS1t, eS2t, eDt, eMIt)
            curve.append((step, round(tr_m, 4), round(te_m, 4)))
            print(f"CURVE {step} train {tr_m:.4f} test {te_m:.4f}", flush=True)
            if te_m < best["test"]:              # keep the best-on-held-out weights, not just the last
                best = dict(test=round(te_m, 4), train=round(tr_m, 4), step=step)
                if a.save:
                    torch.save(dict(state_dict=model.state_dict(), opt=opt.state_dict(), args=vars(a), step=step,
                                    train_mae=tr_m, test_mae=te_m),
                               os.path.splitext(a.save)[0] + ".best.pt")
        if ck_dir and step > 0 and step % a.ckptevery == 0:
            torch.save(dict(state_dict=model.state_dict(), opt=opt.state_dict(), args=vars(a), step=step),
                       f"{ck_dir}/step{step:07d}.pt")
    train_sec = time.time() - t0
    model.eval()

    @torch.no_grad()
    def ev(s1, s2, dd, mi):
        pr = torch.cat([model(s1[i:i + 4096], s2[i:i + 4096], mi[i:i + 4096])
                        for i in range(0, len(s1), 4096)]).float().cpu().numpy()
        tv = dd.cpu().numpy()
        out = dict(dist_mae=round(float(np.abs(pr - tv).mean()), 3),
                   dist_corr=round(float(np.corrcoef(pr, tv)[0, 1]), 3) if pr.std() > 1e-6 else None)
        if a.Rtrain and a.Rtrain < a.Rmax:
            far = tv > a.Rtrain
            if far.any():
                out["mae_beyond"] = round(float(np.abs(pr[far] - tv[far]).mean()), 3)
                out["corr_beyond"] = (round(float(np.corrcoef(pr[far], tv[far])[0, 1]), 3)
                                      if far.sum() > 2 and pr[far].std() > 1e-6 else None)
            if (~far).any(): out["mae_within"] = round(float(np.abs(pr[~far] - tv[~far]).mean()), 3)
        rung = {}
        for dv in sorted(set(int(x) for x in tv)):
            sel = tv == dv
            if sel.sum() >= 5: rung[int(dv)] = round(float(np.abs(pr[sel] - dv).mean()), 4)
        out["rungs"] = rung
        return out

    res = dict(env="switchyard", G=a.G, ngate=a.ngate, nmaps=a.nmaps, split=a.split, Rtrain=a.Rtrain,
               sched=a.sched, lr=a.lr, evalevery=a.evalevery, mergegates=a.mergegates, padslots=a.padslots, progT=a.progT, progTmin=a.progTmin, seewalls=a.seewalls, arrive=a.arrive, unitstep=a.unitstep,
               level=a.level, globalbase=a.globalbase,
               fixpull=a.fixpull,
               Rmax=a.Rmax, steps=a.steps, seed=a.seed, d=a.d, layers=a.layers, T=a.T,
               readout=a.readout, pool=a.pool, params=nparam)
    res["test"] = ev(eS1t, eS2t, eDt, eMIt)
    res["train"] = ev(S1t[:20000], S2t[:20000], Dt[:20000], MIt[:20000])
    if curve: res["curve"] = curve; res["best"] = best
    print("RESULT " + json.dumps(res), flush=True)
    print("META " + json.dumps(dict(readout=a.readout, params=nparam, train_sec=round(train_sec, 1))), flush=True)
    if a.save:
        torch.save(dict(state_dict=model.state_dict(), opt=opt.state_dict(), args=vars(a), result=res,
                        maps=MTn), a.save)


if __name__ == "__main__":
    main()

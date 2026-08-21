"""Switchyard: a game-like discrete gridworld benchmark with INTERDEPENDENT factors.

A worker pushes a crate around a walled yard whose gates are toggled by levers and held by a pressure
plate. Every element is borrowed from a published GCRL benchmark, but here they are wired into one
dependency web so that the geodesic distance is a joint function of all factors:

  element                  borrowed from                        dependency it creates
  gates (open/closed bits) MiniGrid DoorKey / MAD KeyDoor       gate bits gate BOTH worker and crate moves
  levers (toggle wiring)   OGBench Puzzle / Lights-Out          one pull flips SEVERAL gates (XOR wiring)
  pushable crate           DeepNorm `push` domain / Sokoban     crate moves only via worker contact; pushes
                                                                can be irreversible (directed graph)
  pressure plate           MiniGrid ObstructedMaze family       plate-wired gates are open IFF crate sits on
                                                                the plate, overriding lever bits
  one-way chutes           PQE/IQE one-way-doors gridworld      cells passable in one direction (worker only)

State = [worker_cell, crate_cell, gate_bits]  (factored; gate_bits dynamic).
Config = wall layout + gate positions + lever wiring + plate wiring + chute directions (static per map).
Distance = BFS on the joint deterministic transition graph (directed; exact).

Modes:  --probe   state-space / coupling statistics (factorization-gap analysis)
        --train   smoke-train the recall-flow integrator (+ scalar-head control) on distance pairs
"""
import argparse, collections, itertools, json, numpy as np

# ---------------------------------------------------------------- environment
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))                        # N S W E; action 4 = pull lever

class Yard:
    """One map instance: static layout + wiring config. States are (worker, crate, bits) tuples."""
    def __init__(self, G=7, ngate=3, nlever=2, nchute=1, rng=None, wiring=None, wire_rng=None,
                 wire1=False, noplate=False, nopush=False, gatesopen=False):
        self.G = G; self.D = ngate; rng = rng or np.random.default_rng(0)
        self.nopush = nopush; self.gatesopen = gatesopen
        wrng = wire_rng or rng                                    # wiring rng separable for the wire-only split
        self.wall = np.zeros((G, G), bool)
        wc, wr = G // 2, G // 2                                   # four rooms: one vertical + one horizontal wall
        self.wall[:, wc] = True; self.wall[wr, :] = True
        gaps = [(rng.integers(0, wr), wc), (rng.integers(wr + 1, G), wc),
                (wr, rng.integers(0, wc)), (wr, rng.integers(wc + 1, G))]
        for arm in range(4):                                      # extra always-open gap per arm w.p. 0.5 (connectivity)
            if rng.random() < 0.5:
                r, c = [(rng.integers(0, wr), wc), (rng.integers(wr + 1, G), wc),
                        (wr, rng.integers(0, wc)), (wr, rng.integers(wc + 1, G))][arm]
                if (r, c) not in gaps: self.wall[r, c] = False
        for r, c in gaps: self.wall[r, c] = False
        gi = rng.permutation(4)[:ngate]
        self.gates = [gaps[i] for i in gi]                        # gate g sits in gap cell; open iff bit g (or plate)
        free = [(r, c) for r in range(G) for c in range(G) if not self.wall[r, c] and (r, c) not in gaps]
        pick = rng.permutation(len(free))
        self.levers = [free[i] for i in pick[:nlever]]
        self.plate = free[pick[nlever]]
        self.chutes = {}                                          # cell -> sole allowed entry direction index
        for i in range(nchute):
            self.chutes[free[pick[nlever + 1 + i]]] = int(rng.integers(4))
        if wiring is None:                                        # lever wiring: nonzero masks over gates
            if wire1:                                             # complexity ladder: one distinct gate per lever
                wiring = [(1 << (l % ngate)) if ngate else 0 for l in range(nlever)]
            else:
                wiring = [(1 + int(wrng.integers((1 << ngate) - 1))) if ngate else 0 for _ in range(nlever)]
        self.wiring = list(wiring)
        self.platemask = 0 if (noplate or not ngate) else 1 + int(wrng.integers((1 << ngate) - 1))
        self.cells = [(r, c) for r in range(G) for c in range(G) if not self.wall[r, c] or (r, c) in gaps]
        self.cid = {rc: i for i, rc in enumerate(self.cells)}

    def cfg_key(self):
        return tuple(self.wiring) + (self.platemask,) + tuple(sorted(self.chutes.items()))

    def open_gates(self, bits, crate):
        if self.gatesopen: return (1 << self.D) - 1              # ladder L0/L1: gate dynamics disabled
        eff = bits
        if crate == self.plate: eff |= self.platemask            # plate held down forces its gates open
        return eff

    def passable(self, rc, bits, crate, came_dir=None, is_crate=False):
        r, c = rc
        if not (0 <= r < self.G and 0 <= c < self.G): return False
        if self.wall[r, c] and rc not in [g for g in self.gates]:
            return rc in [g for g in self.gates]
        if rc in self.gates:
            g = self.gates.index(rc)
            if not (self.open_gates(bits, crate) >> g) & 1: return False
        if not is_crate and rc in self.chutes and came_dir is not None and came_dir != self.chutes[rc]:
            return False                                          # chute: worker may enter only along its direction
        return True

    def neighbours(self, s):
        (wr_, cr_, bits) = s; out = []
        w = self.cells[wr_]; b = self.cells[cr_]
        for d, (dr, dc) in enumerate(DIRS):
            nw = (w[0] + dr, w[1] + dc)
            if nw == b:                                           # push attempt
                if self.nopush: continue                          # ladder L0: crate is a static obstacle
                nb = (b[0] + dr, b[1] + dc)
                if self.passable(nb, bits, b, None, True) and nb not in self.chutes and nb != nw:
                    nbits = bits                                  # crate leaves/enters plate -> gates re-evaluated lazily
                    if self.passable(nw, nbits, nb, d):
                        out.append((self.cid[nw], self.cid[nb], nbits))
            elif self.passable(nw, bits, b, d):
                out.append((self.cid[nw], cr_, bits))
        if w in self.levers and not self.gatesopen:               # pull: XOR the lever's wiring into the gate bits
            out.append((wr_, cr_, bits ^ self.wiring[self.levers.index(w)]))
        return out

    def bfs(self, src, maxnodes=200000):
        dist = {src: 0}; dq = collections.deque([src])
        while dq and len(dist) < maxnodes:
            u = dq.popleft()
            for v in self.neighbours(u):
                if v not in dist: dist[v] = dist[u] + 1; dq.append(v)
        return dist

    def rand_state(self, rng):
        while True:
            wr_, cr_ = int(rng.integers(len(self.cells))), int(rng.integers(len(self.cells)))
            if wr_ == cr_: continue
            if self.cells[cr_] in self.chutes: continue
            return (wr_, cr_, 0 if self.gatesopen else int(rng.integers(1 << self.D)))

    def vec(self, s):                                             # factored vector for the model
        return np.array([s[0], s[1], s[2]], np.int64)

# ------------------------------------------------------- factorized proxy (coupling probe)
def proxy_dist(yard, s, g):
    """Best factorized approximation: independent worker walk + crate pushes + minimal lever pulls,
    each computed in a DECOUPLED world (all gates open, no plate, other factor ghosted)."""
    G = yard.G
    def walk(a, b, crate_graph=False):
        if a == b: return 0
        dist = {a: 0}; dq = collections.deque([a])
        while dq:
            u = dq.popleft()
            for dr, dc in DIRS:
                v = (u[0] + dr, u[1] + dc)
                if not (0 <= v[0] < G and 0 <= v[1] < G): continue
                if yard.wall[v[0], v[1]] and v not in yard.gates: continue
                if v not in dist:
                    dist[v] = dist[u] + 1
                    if v == b: return dist[v]
                    dq.append(v)
        return None
    dw = walk(yard.cells[s[0]], yard.cells[g[0]])
    db = walk(yard.cells[s[1]], yard.cells[g[1]])
    need = s[2] ^ g[2]                                            # min pulls reaching target bits (BFS over 2^D)
    dist = {s[2]: 0}; dq = collections.deque([s[2]])
    dbit = None
    while dq:
        u = dq.popleft()
        if u == g[2]: dbit = dist[u]; break
        for wmask in yard.wiring:
            v = u ^ wmask
            if v not in dist: dist[v] = dist[u] + 1; dq.append(v)
    if None in (dw, db) or dbit is None: return None
    return dw + db + dbit

# ---------------------------------------------------------------- probe mode
def probe(a):
    rng = np.random.default_rng(a.seed)
    gaps_stats, diam, reach, sizes = [], [], [], []
    couple = {"true": [], "proxy": []}
    cfg_sense = []
    for m in range(a.nmaps):
        yard = Yard(a.G, a.ngate, a.nlever, a.nchute, np.random.default_rng(a.seed + m))
        src = yard.rand_state(rng)
        dist = yard.bfs(src)
        nstates = len(yard.cells) * (len(yard.cells) - 1) * (1 << yard.D)
        sizes.append(nstates); reach.append(len(dist) / nstates); diam.append(max(dist.values()))
        items = list(dist.items())
        for tv, dv in [items[i] for i in rng.permutation(len(items))[:a.npairs]]:
            if dv == 0: continue
            p = proxy_dist(yard, src, tv)
            if p is None: continue
            couple["true"].append(dv); couple["proxy"].append(p)
        base = yard.rand_state(rng); goal = yard.rand_state(rng)   # same endpoints under different wirings
        ds = []
        for w in range(a.nwire):
            y2 = Yard(a.G, a.ngate, a.nlever, a.nchute, np.random.default_rng(a.seed + m))
            y2.wiring = [(1 + int(rng.integers((1 << a.ngate) - 1))) if a.ngate else 0 for _ in range(a.nlever)]
            y2.platemask = (1 + int(rng.integers((1 << a.ngate) - 1))) if a.ngate else 0
            d2 = y2.bfs(base).get(goal)
            if d2 is not None: ds.append(d2)
        if len(ds) > 1: cfg_sense.append(float(np.std(ds)))
    t, p = np.array(couple["true"], float), np.array(couple["proxy"], float)
    gap = t - p
    print(json.dumps(dict(
        nmaps=a.nmaps, statespace=int(np.mean(sizes)), reachable_frac=round(float(np.mean(reach)), 3),
        diameter=round(float(np.mean(diam)), 1), npairs=len(t),
        proxy_corr=round(float(np.corrcoef(t, p)[0, 1]), 3),
        proxy_mae=round(float(np.abs(gap).mean()), 2),
        mean_true=round(float(t.mean()), 2),
        gap_ge3_frac=round(float((gap >= 3).mean()), 3),
        gap_ge6_frac=round(float((gap >= 6).mean()), 3),
        cfg_dist_std=round(float(np.mean(cfg_sense)), 2) if cfg_sense else None)))

# ---------------------------------------------------------------- train mode
def make_yards(a, wire1=None, noplate=None):
    """Returns (yards, train_ids, test_ids). split=map: held-out layouts+wirings. split=wire: SAME
    layouts in train and test, wiring resampled for test (the combo-split analogue)."""
    w1 = a.wire1 if wire1 is None else wire1; npl = a.noplate if noplate is None else noplate
    mk = lambda m, w: Yard(a.G, a.ngate, a.nlever, a.nchute, np.random.default_rng(a.seed + m),
                           wire_rng=np.random.default_rng(50000 + a.seed + w),
                           wire1=w1, noplate=npl, nopush=a.nopush, gatesopen=a.gatesopen)
    if a.split == "wire":
        yards = [mk(m, m) for m in range(a.nmaps)] + [mk(m, 90000 + m) for m in range(a.nmaps)]
        return yards, list(range(a.nmaps)), list(range(a.nmaps, 2 * a.nmaps))
    yards = [mk(m, m) for m in range(a.nmaps)]
    return yards, [m for m in range(a.nmaps) if m % 4], [m for m in range(a.nmaps) if not m % 4]

def build_pool(a, rng, yards, mapids, Rcap):
    S1, S2, D, C = [], [], [], []
    for _ in range(a.poolq):
        m = int(mapids[rng.integers(len(mapids))]); yard = yards[m]
        src = yard.rand_state(rng); dist = yard.bfs(src)
        byd = collections.defaultdict(list)
        for tv, dv in dist.items():
            if 0 < dv <= Rcap: byd[dv].append(tv)
        per = max(1, 40 // max(1, len(byd)))
        for dv, lst in byd.items():
            for j in rng.choice(len(lst), min(per, len(lst)), replace=False):
                S1.append(yard.vec(src)); S2.append(yard.vec(lst[j])); D.append(dv); C.append(m)
    return np.array(S1), np.array(S2), np.array(D, np.float32), np.array(C)

def train(a):
    import torch, torch.nn as nn, torch.nn.functional as F
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    yards, tr_ids, te_ids = make_yards(a)
    ncell = a.G * a.G; NW = max(len([1 for r in range(a.G) for c in range(a.G) if y.wall[r, c]]) for y in yards)
    cell = lambda y, rc: rc[0] * a.G + rc[1]
    # per-map structural index tensors (walls padded to NW with a count for mean-pooling)
    WALL = torch.zeros(len(yards), NW, dtype=torch.long); WN = torch.zeros(len(yards), 1)
    GC = torch.zeros(len(yards), a.ngate, dtype=torch.long); LC = torch.zeros(len(yards), a.nlever, dtype=torch.long)
    LW = torch.zeros(len(yards), a.nlever, a.ngate); PC = torch.zeros(len(yards), dtype=torch.long)
    PW = torch.zeros(len(yards), a.ngate); CC = torch.zeros(len(yards), dtype=torch.long); CD = torch.zeros(len(yards), dtype=torch.long)
    CELL = torch.zeros(len(yards), ncell, dtype=torch.long)      # map-local cell id -> absolute grid cell
    WALLIMG = torch.zeros(len(yards), ncell)                     # per-map binary wall image (for --enc image render)
    NOPEN = (4 - a.ngate) + 4                                     # --seewalls: max open NON-gate cross cells (arm gaps + extras)
    OPENC = torch.zeros(len(yards), NOPEN, dtype=torch.long); OPENM = torch.zeros(len(yards), NOPEN)
    WIREPATH = torch.zeros(len(yards), a.nlever, ncell)          # --wirepath render: 0.5 along lever->gate L-paths
    for i, y in enumerate(yards):
        wl = [r * a.G + c for r in range(a.G) for c in range(a.G) if y.wall[r, c] and (r, c) not in y.gates]
        WALL[i, :len(wl)] = torch.tensor(wl); WN[i] = max(1, len(wl))
        for cidx in wl: WALLIMG[i, cidx] = 1.0
        wr_ = wc_ = a.G // 2
        cross = [(r, wc_) for r in range(a.G)] + [(wr_, c) for c in range(a.G) if c != wc_]
        opn = [r * a.G + c for (r, c) in cross if not y.wall[r, c] and (r, c) not in y.gates]   # passable, NOT a gate
        OPENC[i, :len(opn)] = torch.tensor(opn[:NOPEN]); OPENM[i, :len(opn)] = 1.0
        for l, (lr_, lc_) in enumerate(y.levers):                 # --wirepath: L-shaped path (row then col) lever -> wired gates
            for g, (gr_, gc_) in enumerate(y.gates):
                if (y.wiring[l] >> g) & 1:
                    for c_ in range(min(lc_, gc_), max(lc_, gc_) + 1): WIREPATH[i, l, lr_ * a.G + c_] = 0.5
                    for r_ in range(min(lr_, gr_), max(lr_, gr_) + 1): WIREPATH[i, l, r_ * a.G + gc_] = 0.5
        GC[i] = torch.tensor([cell(y, g) for g in y.gates]); LC[i] = torch.tensor([cell(y, l) for l in y.levers])
        for li, wm in enumerate(y.wiring): LW[i, li] = torch.tensor([(wm >> g) & 1 for g in range(a.ngate)], dtype=torch.float)
        PC[i] = cell(y, y.plate); PW[i] = torch.tensor([(y.platemask >> g) & 1 for g in range(a.ngate)], dtype=torch.float)
        ch = list(y.chutes.items())[0] if y.chutes else (((0, 0)), 0)
        CC[i] = cell(y, ch[0]); CD[i] = ch[1]
        CELL[i, :len(y.cells)] = torch.tensor([cell(y, rc) for rc in y.cells])
    WALL, WN, GC, LC, LW, PC, PW, CC, CD, CELL, WALLIMG, OPENC, OPENM, WIREPATH = (t.to(dev) for t in (WALL, WN, GC, LC, LW, PC, PW, CC, CD, CELL, WALLIMG, OPENC, OPENM, WIREPATH))
    NTOK = 3 + a.ngate + a.nlever + 3 + (NOPEN if a.seewalls else 0)   # worker crate bits | gates levers plate chute wallpool [| open doorways]
    IMGC = 8                                                      # --enc image render channels: wall worker crate gate gateopen lever plate chute
    PIMGC = 5 + a.nlever + 1 + 4                                  # --enc pureimage: wall worker crate gate gateopen | lever_l+its gates | plate+its gates | chute dir x4
    NENT = 2 + a.ngate + a.nlever + (0 if a.noplate else 1) + a.nchute   # entities that can light a pixel
    if a.enc == "pureimage":                                      # NO symbolic tokens: slots (K queries) | all pixels | foreground pixels
        NTOK = a.slots if a.readout in ("xattn", "slotattn") else (NENT if a.readout == "fgpix" else ncell)

    class Enc(nn.Module):
        """Structural, compositional encoding: unseen maps/wirings are new combinations of known pieces."""
        def __init__(s, d):
            super().__init__()
            s.pos = nn.Embedding(ncell, d); s.bitv = nn.Embedding(2 * a.ngate, d)
            s.gid = nn.Embedding(a.ngate, d); s.gwire = nn.Embedding(a.ngate, d); s.pwire = nn.Embedding(a.ngate, d)
            s.cdir = nn.Embedding(4, d); s.fid = nn.Embedding(NTOK, d)
            if a.enc != "factored":                                # image: worker+crate rendered on a GxG canvas
                s.cellemb = nn.Embedding(2, d)                     # per moving-agent channel (worker, crate)
                s.cellbase = nn.Parameter(torch.randn(d) * 0.02)
                s.cpe = nn.Embedding(ncell, d)                     # per-cell positional code
                if a.enc == "marker":                             # learn to bind agent i from the shared canvas
                    s.cquery = nn.Parameter(torch.randn(2, d) * 0.02)
                    s.cslot = nn.Embedding(2, d)
                    s.cattn = nn.MultiheadAttention(d, a.markerheads, batch_first=True)
                    s.bindhead = nn.Linear(d, ncell) if a.markeraux > 0 else None   # aux: decode bound -> true cell
            if a.enc == "image":                                  # CNN over the FULL rendered scene + cross-attn read
                convs = [nn.Conv2d(IMGC, a.cnnw, 3, padding=1), nn.ReLU()]
                for _ in range(max(0, a.cnndepth - 2)):           # extra hidden conv layers
                    convs += [nn.Conv2d(a.cnnw, a.cnnw, 3, padding=1), nn.ReLU()]
                convs += [nn.Conv2d(a.cnnw, d, 3, padding=1), nn.ReLU()]
                s.cnn = nn.Sequential(*convs)
                if a.readout == "xattn":                          # worker/crate query tokens cross-attend the feat map
                    s.imgpos = nn.Parameter(torch.randn(ncell, d) * 0.02)
                    s.iquery = nn.Parameter(torch.randn(2, d) * 0.02)
                    s.islot = nn.Embedding(2, d)
                    s.iattn = nn.MultiheadAttention(d, a.markerheads, batch_first=True)
                    s.bindhead = nn.Linear(d, ncell) if a.markeraux > 0 else None   # PROBE: supervised binding aux on image
                    if a.recon != "none": s.rdec = nn.Linear(d, d)   # reconstruction decoder: slot -> query over imgpos
                    if a.hardattn:                                    # own single-head attention (for straight-through)
                        s.hq = nn.Linear(d, d); s.hk = nn.Linear(d, d); s.hv = nn.Linear(d, d)
                elif a.readout == "convspatial":                  # per-agent conv logits + spatial softmax pool
                    s.ihead = nn.Conv2d(d, 2, 1)
                else:                                             # convpool: global pool -> project to 2 tokens
                    s.ipool = nn.Linear(d, 2 * d)
            if a.enc == "pureimage":                              # everything in pixels -> CNN -> K slots (or pixel tokens)
                kk, pd_ = a.cnnk, a.cnnk // 2                          # --cnnk 1 = per-pixel MLP (no spatial mixing)
                k2, p2 = (3, 1) if a.cnnmix else (kk, pd_)              # --cnnmix: 1x1 first (keep identity), 3x3 after (context)
                convs = [nn.Conv2d(PIMGC + (2 if a.coordconv else 0) + (1 if a.objch else 0), a.cnnw, kk, padding=pd_), nn.ReLU()]
                if a.readout == "fgpix": s.emptytok = nn.Parameter(torch.randn(d) * 0.02)   # pad token for unlit slots
                for _ in range(max(0, a.cnndepth - 2)):
                    convs += [nn.Conv2d(a.cnnw, a.cnnw, k2, padding=p2), nn.ReLU()]
                convs += [nn.Conv2d(a.cnnw, d, k2, padding=p2), nn.ReLU()]
                if a.slotln: s.slotln = nn.LayerNorm(d)
                s.pcnn = nn.Sequential(*convs)
                s.pimgpos = nn.Parameter(torch.randn(ncell, d) * 0.02)
                if a.readout == "slotattn":                       # COMPETITIVE ITERATIVE SLOT ATTENTION (Locatello et al.)
                    s.smu = nn.Parameter(torch.randn(a.slots, d) * 0.02)          # learned slot init (per-slot identity)
                    s.slogsig = nn.Parameter(torch.zeros(a.slots, d) - 2.0)       # log-sigma for optional init noise
                    s.sln_in = nn.LayerNorm(d); s.sln_s = nn.LayerNorm(d); s.sln_m = nn.LayerNorm(d)
                    s.sq = nn.Linear(d, d, bias=False); s.sk = nn.Linear(d, d, bias=False); s.sv = nn.Linear(d, d, bias=False)
                    s.sgru = nn.GRUCell(d, d)
                    s.smlp = nn.Sequential(nn.Linear(d, 2 * d), nn.ReLU(), nn.Linear(2 * d, d))
                    if a.recon == "slot":
                        s.prdec = nn.Linear(d, d); s.prch = nn.Linear(d, PIMGC - 1)
                if a.readout == "xattn":                          # K generic learned slot queries (nothing named)
                    s.squery = nn.Parameter(torch.randn(a.slots, d) * 0.02)
                    s.sattn = nn.MultiheadAttention(d, a.markerheads, batch_first=True)
                    if a.fgmask == 2: s.fgbias = nn.Parameter(torch.tensor(2.0))   # learned objectness bias on attention logits
                    if a.recon == "slot":                          # UNSUPERVISED: K slots explain ALL entity channels exclusively
                        s.prdec = nn.Linear(d, d)                  # slot -> spatial query over pimgpos
                        s.prch = nn.Linear(d, PIMGC - 1)           # slot -> which entity channel (walls excluded)
                    if a.supbind:                                  # CEILING probe: Hungarian-matched supervised binding
                        s.sbdec = nn.Linear(d, d)
            s._aux = None                                         # set to 0 by training loop to collect binding aux
            if a.seewalls:                                        # open non-gate doorway tokens: pos(cell)+openkind | absent
                s.openkind = nn.Parameter(torch.randn(d) * 0.02)
                s.absent = nn.Parameter(torch.randn(d) * 0.02)
        def _render(s, x, m):
            """render the FULL state to an (B, IMGC, G, G) image (walls/worker/crate/gate/gateopen/lever/plate/chute)."""
            B, dev = x.shape[0], x.device
            wc = CELL[m].gather(1, x[:, 0:2])                      # (B,2) worker,crate cells
            bits = torch.stack([(x[:, 2] >> g) & 1 for g in range(a.ngate)], 1).float()   # (B,ngate)
            img = torch.zeros(B, IMGC, ncell, device=dev)
            img[:, 0] = WALLIMG[m]                                 # walls
            img[:, 1].scatter_(1, wc[:, 0:1], 1.0)                 # worker
            img[:, 2].scatter_(1, wc[:, 1:2], 1.0)                 # crate
            img[:, 3].scatter_(1, GC[m], 1.0)                     # gate cells
            img[:, 4].scatter_(1, GC[m], (bits > 0).float())      # gate-open indicator
            img[:, 5].scatter_(1, LC[m], 1.0)                     # levers
            img[:, 6].scatter_(1, PC[m][:, None], 1.0)            # plate
            img[:, 7].scatter_(1, CC[m][:, None], 1.0)            # chute
            return img.view(B, IMGC, a.G, a.G)

        def _image_read(s, x, m):
            """CNN over the rendered scene, then read worker/crate tokens (no position labels used)."""
            feat = s.cnn(s._render(x, m))                          # (B,d,G,G)
            B = x.shape[0]
            if a.readout == "xattn":
                img = s._render(x, m)                                            # keep for reconstruction targets
                fmap = feat.flatten(2).transpose(1, 2) + s.imgpos[None]         # (B,ncell,d)
                q = (s.iquery + s.islot(torch.arange(2, device=x.device)))[None].expand(B, -1, -1)
                if a.hardattn:                                                   # straight-through top-1 read
                    sc = torch.einsum("bqd,bkd->bqk", s.hq(q), s.hk(fmap)) / fmap.shape[-1] ** 0.5
                    soft = sc.softmax(-1)
                    hard = torch.zeros_like(soft).scatter_(-1, soft.argmax(-1, keepdim=True), 1.0)
                    w = hard + soft - soft.detach()
                    bound = torch.einsum("bqk,bkd->bqd", w, s.hv(fmap))
                else:
                    bound, _ = s.iattn(q, fmap, fmap)
                if a.recon != "none" and s._aux is not None:                     # UNSUPERVISED: reconstruct own input channels
                    d_ = fmap.shape[-1]
                    logits = torch.einsum("bqd,kd->bqk", s.rdec(bound), s.imgpos) / d_ ** 0.5   # (B,2,ncell)
                    ent = img[:, 1:3].flatten(2)                                 # (B,2,ncell) worker / crate one-hot channels
                    if a.recon == "tied":                                        # slot i explains entity channel i
                        s._aux = s._aux + F.cross_entropy(logits.reshape(-1, logits.shape[-1]), ent.argmax(-1).reshape(-1))
                    else:                                                        # slot: explain the UNION mask, no identity, exclusive
                        p = logits.softmax(-1)                                   # (B,2,ncell)
                        mix = 0.5 * (p[:, 0] + p[:, 1])
                        mask = ent.sum(1) / 2.0                                  # two entity cells, weight 1/2 each
                        s._aux = s._aux - (mask * (mix + 1e-9).log()).sum(-1).mean() \
                                        + 2.0 * (p[:, 0] * p[:, 1]).sum(-1).mean()   # overlap penalty
                        s._slotstat = (p[:, 0].argmax(-1) == ent[:, 0].argmax(-1)).float().mean()   # slot0==worker consistency
                if getattr(s, "bindhead", None) is not None and s._aux is not None:   # supervised binding aux (upper-bound probe)
                    wc = CELL[m].gather(1, x[:, 0:2])                          # true worker/crate cells
                    s._aux = s._aux + F.cross_entropy(s.bindhead(bound[:, 0]), wc[:, 0]) \
                                    + F.cross_entropy(s.bindhead(bound[:, 1]), wc[:, 1])
                return bound[:, 0], bound[:, 1]
            if a.readout == "convspatial":
                att = s.ihead(feat).flatten(2).softmax(-1)                       # (B,2,ncell)
                z = torch.einsum("bnh,bdh->bnd", att, feat.flatten(2))          # (B,2,d)
                return z[:, 0], z[:, 1]
            z = s.ipool(feat.mean((2, 3))).view(B, 2, feat.shape[1])            # convpool
            return z[:, 0], z[:, 1]

        def _wc(s, x, m):
            """absolute-cell worker/crate tokens; factored=lookup, bmask=lossless canvas, marker/image=learned read."""
            if a.enc == "image":
                return s._image_read(x, m)
            wc = CELL[m].gather(1, x[:, 0:2])                      # (B,2) absolute grid cells
            if a.enc == "factored" or a.enc == "bmask":           # bmask canvas is lossless -> same cell tokens
                return s.pos(wc[:, 0]), s.pos(wc[:, 1])
            B, dev = x.shape[0], x.device                         # marker: additive canvas + query attention
            canvas = s.cellbase[None, None].expand(B, ncell, s.pos.weight.shape[1]).clone()
            for i in range(2):
                hit = torch.zeros(B, ncell, device=dev).scatter_(1, wc[:, i:i + 1], 1.0)
                canvas = canvas + hit[..., None] * s.cellemb.weight[i]
            canvas = canvas + s.cpe(torch.arange(ncell, device=dev))[None]
            dd = s.pos.weight.shape[1]
            if a.bindmode == "gather":                            # UNSUPERVISED: match each agent's own cellemb
                allpos = s.pos(torch.arange(ncell, device=dev))   # (ncell,d) -> soft one-hot over positions
                out = []
                for i in range(2):
                    w = ((canvas * s.cellemb.weight[i]).sum(-1) / dd ** 0.5).softmax(-1)   # (B,ncell)
                    out.append(w @ allpos)
                return out[0], out[1]
            if a.bindmode == "slot":                              # UNSUPERVISED: slots COMPETE for cells
                slots = (s.cquery + s.cslot(torch.arange(2, device=dev)))[None].expand(B, -1, -1)  # (B,2,d)
                for _ in range(3):
                    att = (canvas @ slots.transpose(1, 2) / dd ** 0.5).softmax(-1)   # softmax over SLOTS -> (B,ncell,2)
                    att = att / (att.sum(1, keepdim=True) + 1e-8)                    # normalize over cells
                    slots = att.transpose(1, 2) @ canvas                            # (B,2,d) weighted mean
                return slots[:, 0], slots[:, 1]
            q = (s.cquery + s.cslot(torch.arange(2, device=dev)))[None].expand(B, -1, -1)   # attn (learned query)
            bound, _ = s.cattn(q, canvas, canvas)
            if s.bindhead is not None and s._aux is not None:     # optional binding SUPERVISION (aux loss; --bindmode attn)
                s._aux = s._aux + F.cross_entropy(s.bindhead(bound[:, 0]), wc[:, 0]) \
                                + F.cross_entropy(s.bindhead(bound[:, 1]), wc[:, 1])
            return bound[:, 0], bound[:, 1]
        def _render_pure(s, x, m):
            """EVERYTHING as pixels: (B, PIMGC, G, G). wall | worker | crate | gate cells | gate-open bits |
            per lever: its cell + the gate cells it toggles | plate cell + the gate cells it holds | chute cell in one of 4 direction channels."""
            B, dev = x.shape[0], x.device
            wc = CELL[m].gather(1, x[:, 0:2])
            bits = torch.stack([(x[:, 2] >> g) & 1 for g in range(a.ngate)], 1).float()
            img = torch.zeros(B, PIMGC, ncell, device=dev)
            img[:, 0] = WALLIMG[m]
            img[:, 1].scatter_(1, wc[:, 0:1], 1.0); img[:, 2].scatter_(1, wc[:, 1:2], 1.0)
            img[:, 3].scatter_(1, GC[m], 1.0); img[:, 4].scatter_(1, GC[m], (bits > 0).float())
            for l in range(a.nlever):                              # wiring drawn as shared "colour": lever l + gates it flips
                if a.wirepath: img[:, 5 + l] = img[:, 5 + l] + WIREPATH[m, l]   # 0.5 along an L-path lever -> each wired gate
                img[:, 5 + l].scatter_(1, LC[m][:, l:l + 1], 1.0)
                img[:, 5 + l].scatter_(1, GC[m], LW[m][:, l, :])
            pc = 5 + a.nlever
            img[:, pc].scatter_(1, PC[m][:, None], 1.0); img[:, pc].scatter_(1, GC[m], PW[m])
            for dd_ in range(4):                                   # chute direction as channel choice
                img[:, pc + 1 + dd_].scatter_(1, CC[m][:, None], (CD[m] == dd_).float()[:, None])
            return img.view(B, PIMGC, a.G, a.G)
        def _pure(s, x, m):
            B = x.shape[0]
            img = s._render_pure(x, m)                             # (B,PIMGC,G,G)
            cin = img
            if a.coordconv:                                        # explicit x/y coordinate channels
                yy, xx = torch.meshgrid(torch.linspace(-1, 1, a.G, device=x.device), torch.linspace(-1, 1, a.G, device=x.device), indexing="ij")
                cin = torch.cat([cin, yy[None, None].expand(B, 1, -1, -1), xx[None, None].expand(B, 1, -1, -1)], 1)
            if a.objch:                                            # objectness FEATURE channel: 1 where any entity channel is lit
                cin = torch.cat([cin, (img[:, 1:] > 0.75).any(1, keepdim=True).float()], 1)
            feat = s.pcnn(cin)                                     # (B,d,G,G)
            fmap = feat.flatten(2).transpose(1, 2) + s.pimgpos[None]   # (B,ncell,d)
            if a.readout == "fgpix":                               # FOREGROUND pixels as tokens: cells lit in any entity channel
                fg = (img.flatten(2)[:, 1:] > 0.75).any(1).float()  # (B,ncell) non-wall, non-wirepath
                order = torch.argsort(fg + 1e-3 * torch.rand_like(fg), dim=1, descending=True)[:, :NTOK]   # lit cells first
                tok = fmap.gather(1, order[..., None].expand(-1, -1, fmap.shape[-1]))
                lit = fg.gather(1, order)[..., None]
                tok = lit * tok + (1 - lit) * s.emptytok            # unlit slots -> learned empty token
                return tok + s.fid(torch.arange(NTOK, device=x.device))[None]
            if a.readout == "slotattn":                            # competitive iterative slot attention
                d_ = fmap.shape[-1]; K = a.slots
                inp = s.sln_in(fmap); k_ = s.sk(inp); v_ = s.sv(inp)                    # (B,N,d)
                slots = s.smu[None].expand(B, -1, -1)
                if a.slotnoise and s.training: slots = slots + s.slogsig.exp()[None] * torch.randn_like(slots)
                for _ in range(a.slotiters):
                    prev = slots
                    q_ = s.sq(s.sln_s(slots))                                             # (B,K,d)
                    logits = torch.einsum("bkd,bnd->bkn", q_, k_) / d_ ** 0.5           # (B,K,N)
                    attn = logits.softmax(1)                                              # COMPETITION: softmax over SLOTS per pixel
                    attn = attn / (attn.sum(-1, keepdim=True) + 1e-8)                     # weighted mean over pixels
                    upd = torch.einsum("bkn,bnd->bkd", attn, v_)
                    slots = s.sgru(upd.reshape(-1, d_), prev.reshape(-1, d_)).view(B, K, d_)
                    slots = slots + s.smlp(s.sln_m(slots))
                tok = slots
                s._lastattn = attn.detach(); s._lastimg = img.detach()
                if a.recon == "slot" and s._aux is not None:                              # same unsupervised recon aux
                    imgf = img.flatten(2)[:, 1:]; tgt = (imgf > 0.75).float()
                    p = (torch.einsum("bkd,nd->bkn", s.prdec(tok), s.pimgpos) / d_ ** 0.5).softmax(-1)
                    ch = s.prch(tok).softmax(-1); pred = torch.einsum("bkc,bkn->bcn", ch, p)
                    nent = tgt.sum((1, 2)).clamp(min=1)
                    s._aux = s._aux - ((tgt * (pred + 1e-9).log()).sum((1, 2)) / nent).mean() \
                                    + 2.0 * (torch.einsum("bkn,bjn->bkj", p, p).triu(1).sum((1, 2))).mean()
                return tok + s.fid(torch.arange(NTOK, device=x.device))[None]
            if a.readout == "xattn":                               # K slots
                q = s.squery[None].expand(B, -1, -1)
                if a.fgmask:                                       # objectness hint: which pixels are objects (lit in any entity channel)
                    fgb = (img.flatten(2)[:, 1:] > 0.75).any(1)    # (B,ncell) bool
                    if a.fgmask == 1:                              # hard: slots may only attend to object cells
                        tok, attw = s.sattn(q, fmap, fmap, key_padding_mask=~fgb, need_weights=True, average_attn_weights=True)
                    else:                                          # soft: learned additive bias on object cells' logits
                        am = (fgb.float() * s.fgbias)[:, None, :].expand(-1, a.slots, -1)               # (B,K,ncell)
                        am = am.repeat_interleave(a.markerheads, 0)                                       # (B*heads,K,ncell)
                        tok, attw = s.sattn(q, fmap, fmap, attn_mask=am, need_weights=True, average_attn_weights=True)
                else:
                    tok, attw = s.sattn(q, fmap, fmap, need_weights=True, average_attn_weights=True)   # (B,K,d), (B,K,ncell)
                s._lastattn = attw.detach(); s._lastimg = img.detach()
                if a.attnent > 0 and s._aux is not None:           # sharpen: penalise attention entropy (soft 'crisp read')
                    s._aux = s._aux + a.attnent * (-(attw * (attw + 1e-9).log()).sum(-1)).mean()
                if a.attnovl > 0 and s._aux is not None:           # slot DIVERSITY: penalise overlapping attention maps
                    s._aux = s._aux + a.attnovl * torch.einsum("bkn,bjn->bkj", attw, attw).triu(1).sum((1, 2)).mean()
                if a.slotln: tok = s.slotln(tok)
                if a.recon == "slot" and s._aux is not None:       # slots reconstruct the entity channels of their OWN input
                    img = s._render_pure(x, m).flatten(2)[:, 1:]   # (B,PIMGC-1,ncell) all channels except walls
                    tgt = (img > 0.75).float()                     # entity cells only (wire paths at 0.5 excluded)
                    d_ = fmap.shape[-1]
                    p = (torch.einsum("bkd,nd->bkn", s.prdec(tok), s.pimgpos) / d_ ** 0.5).softmax(-1)   # (B,K,ncell)
                    ch = s.prch(tok).softmax(-1)                   # (B,K,C)  which channel each slot explains
                    pred = torch.einsum("bkc,bkn->bcn", ch, p)     # (B,C,ncell) mixture over slots
                    nent = tgt.sum((1, 2)).clamp(min=1)            # entities per example
                    s._aux = s._aux - ((tgt * (pred + 1e-9).log()).sum((1, 2)) / nent).mean() \
                                    + 2.0 * (torch.einsum("bkn,bjn->bkj", p, p).triu(1).sum((1, 2))).mean()   # slot overlap penalty
                if a.supbind and s._aux is not None:               # CEILING: optimal slot<->entity assignment, CE on matches
                    from scipy.optimize import linear_sum_assignment
                    d_ = fmap.shape[-1]
                    logp = (torch.einsum("bkd,nd->bkn", s.sbdec(tok), s.pimgpos) / d_ ** 0.5).log_softmax(-1)   # (B,K,ncell)
                    wc = CELL[m].gather(1, x[:, 0:2])
                    ents = torch.cat([wc, GC[m], LC[m], PC[m][:, None], CC[m][:, None]], 1)   # (B,E) true entity cells
                    E = ents.shape[1]; K = logp.shape[1]
                    cost = -logp.gather(2, ents[:, None, :].expand(-1, K, -1))               # (B,K,E)
                    ce = torch.zeros((), device=x.device)
                    cn = cost.detach().cpu().numpy()
                    for b_ in range(x.shape[0]):
                        rows, cols = linear_sum_assignment(cn[b_])
                        ce = ce + cost[b_, rows, cols].sum() / len(rows)
                    s._aux = s._aux + ce / x.shape[0]
            else:                                                  # pixels: every cell is a token
                tok = fmap
            return tok + s.fid(torch.arange(NTOK, device=x.device))[None]
        def forward(s, x, m):
            if a.enc == "pureimage": return s._pure(x, m)
            B, d = x.shape[0], s.pos.weight.shape[1]
            w, c = s._wc(x, m)
            bits = torch.stack([(x[:, 2] >> g) & 1 for g in range(a.ngate)], 1)
            bt = s.bitv(torch.arange(a.ngate, device=x.device)[None] * 2 + bits).sum(1)
            gts = s.pos(GC[m]) + s.gid.weight[None, :, :]
            lvs = s.pos(LC[m]) + torch.einsum("blg,gd->bld", LW[m], s.gwire.weight)
            plt = s.pos(PC[m]) + torch.einsum("bg,gd->bd", PW[m], s.pwire.weight)
            cht = s.pos(CC[m]) + s.cdir(CD[m])
            wal = s.pos(WALL[m]).sum(1) / WN[m]
            toks = [w[:, None], c[:, None], bt[:, None], gts, lvs, plt[:, None], cht[:, None], wal[:, None]]
            if a.seewalls:
                om = OPENM[m][..., None]                           # (B,NOPEN,1)
                toks.append(om * (s.pos(OPENC[m]) + s.openkind) + (1 - om) * s.absent)
            tok = torch.cat(toks, 1)
            return tok + s.fid(torch.arange(NTOK, device=x.device))[None]

    class Block(nn.Module):
        def __init__(s, d, h, l):
            super().__init__()
            s.ls = nn.ModuleList([nn.TransformerEncoderLayer(d, h, 2 * d, dropout=0.0, activation="gelu",
                                  batch_first=True, norm_first=True) for _ in range(max(0, l))])   # l=0 -> identity
        def forward(s, z):
            for l in s.ls: z = l(z)
            return z

    class Integ(nn.Module):
        def __init__(s, d=a.d, h=a.heads, l=a.layers, T=a.T):
            super().__init__()
            s.T = T; s.enc = Enc(d); s.role = nn.Embedding(3, d)
            s.block = Block(d, h, l); s.scale = nn.Parameter(torch.zeros(()))
            s.gbase = nn.Parameter(torch.zeros(())) if a.globalbase else None
            s.head = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, 1)) if a.decodehead else None
        def forward(s, x, g, m, Trun=None, ret_arr=False):
            zs = s.enc(x, m) + s.role.weight[0]; zg = s.enc(g, m) + s.role.weight[1]
            base = torch.cat([zg, s.enc(x, m) + s.role.weight[2]], 1)
            tok = torch.cat([zs, base], 1); cost = torch.zeros(x.shape[0], device=x.device)
            for _ in range(Trun or s.T):
                z = s.block(tok); cost = cost + (z[:, :NTOK] - tok[:, :NTOK]).norm(dim=-1).sum(-1)
                tok = z if a.norecall else torch.cat([z[:, :NTOK], base], 1)
            if s.head is not None:                                 # decode-head ablation: same recurrence, no accumulation
                out = F.softplus(s.head(tok[:, :NTOK].mean(1)).squeeze(-1))   # softplus matches reference scalar_mlp
            else:
                out = F.softplus(s.scale) * cost
                out = out + F.softplus(s.gbase) if s.gbase is not None else out
            if ret_arr:
                return out, (tok[:, :NTOK] - (s.enc(g, m) + s.role.weight[0])).norm(dim=-1).mean(-1)
            return out

    BL = a.baselayers if a.baselayers >= 0 else a.layers           # baseline mix-block depth (tunable, fair; 0 = no transformer)

    class Pool(nn.Module):
        """baseline state pooling: mean over tokens (default) or FLATTEN all tokens -> Linear (keeps which-token-where)."""
        def __init__(s, d):
            super().__init__()
            s.flat = nn.Sequential(nn.Linear(NTOK * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)) if a.basepool == "flat" else None
        def forward(s, z):
            return s.flat(z.flatten(1)) if s.flat is not None else z.mean(1)

    class Sym(nn.Module):
        """Symmetric-embedding baseline on the same structural encoder: D = ||f(s) - f(g)||_1."""
        def __init__(s, d=a.d):
            super().__init__()
            s.enc = Enc(d); s.mix = Block(d, a.heads, BL); s.pool = Pool(d); s.proj = nn.Linear(d, d)
            s.ln = nn.LayerNorm(d) if a.latentnorm else None
        def emb1(s, x, m):
            e = s.proj(s.pool(s.mix(s.enc(x, m))))
            return s.ln(e) if s.ln is not None else e
        def forward(s, x, g, m):
            return (s.emb1(x, m) - s.emb1(g, m)).abs().sum(-1)

    class Qmet(nn.Module):
        """torchqmet baselines on the structural encoder."""
        def __init__(s, kind, d=a.d):
            super().__init__()
            import torchqmet
            s.enc = Enc(d); s.mix = Block(d, a.heads, BL); s.pool = Pool(d); s.proj = nn.Linear(d, d)
            s.ln = nn.LayerNorm(d) if a.latentnorm else None       # standard pre-quasimetric norm; fixes MRN nan
            s.head = torchqmet.IQE(d, dim_per_component=16) if kind == "iqe" else torchqmet.MRNFixed(d)
        def emb1(s, x, m):
            e = s.proj(s.pool(s.mix(s.enc(x, m))))
            return s.ln(e) if s.ln is not None else e
        def forward(s, x, g, m):
            return s.head(s.emb1(x, m), s.emb1(g, m))

    class Scalar(nn.Module):
        def __init__(s, d=a.d):
            super().__init__()
            s.enc = Enc(d); s.mix = Block(d, a.heads, BL); s.pool = Pool(d)
            s.ln = nn.LayerNorm(d) if a.latentnorm else None
            s.head = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, 1))
        def forward(s, x, g, m):
            hs = s.pool(s.mix(s.enc(x, m))); hg = s.pool(s.mix(s.enc(g, m)))
            if s.ln is not None: hs, hg = s.ln(hs), s.ln(hg)
            out = s.head(torch.cat([hs, hg], 1)).squeeze(-1)
            return F.softplus(out) if a.scalarsp else out          # softplus matches reference concat_mlp

    def proxy_pairs(SA, SB, CM):
        out = np.zeros(len(SA), np.float32)
        for i in range(len(SA)):
            y = yards[int(CM[i])]
            p = proxy_dist(y, tuple(int(v) for v in SA[i]), tuple(int(v) for v in SB[i]))
            if p is None:
                wa, wb = y.cells[SA[i][0]], y.cells[SB[i][0]]; ca, cb = y.cells[SA[i][1]], y.cells[SB[i][1]]
                p = abs(wa[0] - wb[0]) + abs(wa[1] - wb[1]) + abs(ca[0] - cb[0]) + abs(ca[1] - cb[1]) + bin(SA[i][2] ^ SB[i][2]).count("1")
            out[i] = p
        return out

    Rcap = a.Rtrain if a.Rtrain else a.Rmax
    S1, S2, D, C = build_pool(a, rng, yards, tr_ids, Rcap)
    phases = [(len(S1) and 1.0, (S1, S2, D, C))]                  # single phase by default
    if a.lencurr and a.Rtrain > 8:                                # length curriculum: growing training range
        caps = sorted({8, min(12, a.Rtrain), a.Rtrain})
        pools = [build_pool(a, np.random.default_rng(a.seed + 40 + i), yards, tr_ids, cap)
                 for i, cap in enumerate(caps[:-1])] + [(S1, S2, D, C)]
        fr = [0.25] * (len(pools) - 1)
        phases = list(zip(fr + [1.0 - sum(fr)], pools))
    if a.curriculum:                                              # easy wiring -> no plate -> full (same encoder)
        ye, _, _ = make_yards(a, wire1=True, noplate=True); ym, _, _ = make_yards(a, wire1=False, noplate=True)
        pe = build_pool(a, np.random.default_rng(a.seed + 7), ye, tr_ids, Rcap)
        pm = build_pool(a, np.random.default_rng(a.seed + 8), ym, tr_ids, Rcap)
        phases = [(0.25, pe), (0.25, pm), (0.5, (S1, S2, D, C))]
    S1t, S2t, Dt, Ct = (torch.as_tensor(x, device=dev) for x in (S1, S2, D, C))
    E1, E2, ED, EC = build_pool(a, np.random.default_rng(a.seed + 99), yards, te_ids, a.Rmax)
    E1t, E2t, EDt, ECt = (torch.as_tensor(x, device=dev) for x in (E1, E2, ED, EC))
    PXtr = torch.as_tensor(proxy_pairs(S1, S2, C), device=dev) if a.resprox else None
    PXte = torch.as_tensor(proxy_pairs(E1, E2, EC), device=dev) if a.resprox else None
    def enc_t(dd, px): return (dd - px) if a.resprox else dd
    def dec_t(pp, px): return (pp + px) if a.resprox else pp
    print(f"pool train={len(S1)} test={len(E1)} split={a.split} Rcap={Rcap} maxd_tr={int(D.max())} maxd_te={int(ED.max())}", flush=True)
    out = {}
    if a.symonly:
        models = [("sym", Sym().to(dev))]
    elif a.iqeonly:
        models = [("iqe", Qmet("iqe").to(dev))]
    elif a.mrnonly:
        models = [("mrn", Qmet("mrn").to(dev))]
    elif a.scalaronly:
        models = [("scalar", Scalar().to(dev))]
    else:
        models = [("integ", Integ().to(dev))] + ([] if a.nobaseline else [("scalar", Scalar().to(dev))])
        if a.symbaseline: models.append(("sym", Sym().to(dev)))
    if a.bellman > 0:                                              # unlabeled pairs + neighbor sets, per train map
        U0, UG, UN, UM, UC = [], [], [], [], []
        nmax = 0; tmp = []
        brng = np.random.default_rng(a.seed + 31)
        for _ in range(a.bellpairs):
            m = int(tr_ids[brng.integers(len(tr_ids))]); y = yards[m]
            s = y.rand_state(brng); g = y.rand_state(brng)
            nb = y.neighbours(s)
            if not nb or s == g: continue
            tmp.append((m, s, g, nb)); nmax = max(nmax, len(nb))
        for m, s, g, nb in tmp:
            U0.append(y.vec(s) if False else np.array(s)); UG.append(np.array(g)); UC.append(m)
            UN.append(np.stack([np.array(x) for x in nb] + [np.array(nb[0])] * (nmax - len(nb))))
            UM.append([1.0] * len(nb) + [0.0] * (nmax - len(nb)))
        U0t = torch.as_tensor(np.array(U0), device=dev); UGt = torch.as_tensor(np.array(UG), device=dev)
        UNt = torch.as_tensor(np.array(UN), device=dev); UMt = torch.as_tensor(np.array(UM), device=dev)
        UCt = torch.as_tensor(np.array(UC), device=dev)
        print(f"bellman pool {len(U0)} nmax {nmax}", flush=True)
    for name, model in models:
        NPARAM = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{name} params {NPARAM}", flush=True)
        opt = torch.optim.Adam(model.parameters(), a.lr)
        parking = name == "integ" and (a.arrive > 0 or a.Tmin >= 0)
        shadow = None
        if a.bellman > 0 and a.ematarget and name == "integ":
            import copy
            shadow = copy.deepcopy(model)
            for p in shadow.parameters(): p.requires_grad_(False)
        step = 0; nskip = 0; best_c = float("-inf"); best_m = float("inf")
        for frac, (P1, P2, PD, PCm) in phases:
            P1t, P2t, PDt, PCt = (torch.as_tensor(x, device=dev) for x in (P1, P2, PD, PCm))
            for _ in range(int(a.steps * frac)):
                b = torch.randint(0, len(P1t), (a.bs,), device=dev)
                tgt = enc_t(PDt[b], PXtr[b]) if (a.resprox and not a.curriculum) else PDt[b]
                aux_on = hasattr(model, "enc") and ((a.markeraux > 0 and getattr(model.enc, "bindhead", None) is not None)
                                                    or (a.recon != "none" and a.enc == "image")
                                                    or ((a.recon == "slot" or a.supbind or a.attnent > 0 or a.attnovl > 0) and a.enc == "pureimage" and a.readout in ("xattn", "slotattn")))
                if aux_on:
                    model.enc._aux = torch.zeros((), device=dev)   # collect binding aux over this forward's enc calls
                if parking:
                    Trun = int(torch.randint(max(1, a.Tmin), a.T + 1, (1,)).item()) if a.Tmin >= 0 else None
                    cost, arr = model(P1t[b], P2t[b], PCt[b], Trun=Trun, ret_arr=True)
                    loss = F.smooth_l1_loss(cost, tgt) + a.arrive * arr.mean()
                else:
                    loss = F.smooth_l1_loss(model(P1t[b], P2t[b], PCt[b]), tgt)
                if a.bellman > 0 and name == "integ" and step >= a.bellstart * a.steps:
                    bw = a.bellman * (min(1.0, step / max(1, int(a.steps * 0.4))) if a.bellwarm else 1.0)
                    tm = shadow if shadow is not None else model
                    ub = torch.randint(0, len(U0t), (a.bellbs,), device=dev)
                    with torch.no_grad():
                        nb_ = UNt[ub]; B_, K_, _ = nb_.shape
                        dn = tm(nb_.reshape(B_ * K_, -1), UGt[ub].repeat_interleave(K_, 0),
                                UCt[ub].repeat_interleave(K_, 0)).reshape(B_, K_)
                        dn = torch.where(UMt[ub] > 0, dn, torch.full_like(dn, 1e9))
                        btgt = 1 + dn.min(-1).values
                    loss = loss + bw * F.smooth_l1_loss(model(U0t[ub], UGt[ub], UCt[ub]), btgt)
                if aux_on and isinstance(model.enc._aux, torch.Tensor):
                    loss = loss + (a.markeraux if a.markeraux > 0 else a.supbind if a.supbind > 0 else a.reconw) * model.enc._aux
                    model.enc._aux = None                          # off during any eval / bellman-only forwards
                if a.warmup > 0 or a.cosine:                       # linear lr warmup (+ optional cosine decay to 5%)
                    import math
                    wu = min(1.0, (step + 1) / a.warmup) if a.warmup > 0 else 1.0
                    cd = (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / max(1, a.steps))))) if a.cosine else 1.0
                    for gparam in opt.param_groups: gparam["lr"] = a.lr * wu * cd
                if not torch.isfinite(loss):                       # skip non-finite spikes (MRN); don't poison weights
                    opt.zero_grad(set_to_none=True); nskip += 1; step += 1; continue
                opt.zero_grad(); loss.backward()
                if a.gradclip > 0: torch.nn.utils.clip_grad_norm_(model.parameters(), a.gradclip)
                opt.step()
                if shadow is not None:
                    with torch.no_grad():
                        for ps, pm in zip(shadow.parameters(), model.parameters()):
                            ps.mul_(a.emam).add_(pm, alpha=1 - a.emam)
                if step % max(1, a.steps // 4) == 0:
                    print(f"{name} step {step} loss {loss.item():.3f}", flush=True)
                if a.evalevery > 0 and step > 0 and step % a.evalevery == 0:   # held-out eval curve (best-checkpoint diagnostic)
                    with torch.no_grad():
                        prv = torch.cat([model(E1t[i:i + 4000], E2t[i:i + 4000], ECt[i:i + 4000]) for i in range(0, len(E1t), 4000)])
                        if a.resprox: prv = dec_t(prv, PXte)
                        vm = float((prv - EDt).abs().mean().item()); vc = float(np.corrcoef(prv.cpu(), ED)[0, 1])
                    if vm < best_m: best_m = vm
                    if vc > best_c: best_c = vc
                    print(f"{name} step {step} evalmae {vm:.3f} evalcorr {vc:.3f}", flush=True)
                step += 1
        if name == "integ" and a.Ttest > 0: model.T = a.Ttest
        model.eval()
        with torch.no_grad():
            pr_tr = model(S1t[:4000], S2t[:4000], Ct[:4000])
            pr = torch.cat([model(E1t[i:i + 4000], E2t[i:i + 4000], ECt[i:i + 4000]) for i in range(0, len(E1t), 4000)])
            if a.resprox:
                pr_tr = dec_t(pr_tr, PXtr[:4000]); pr = dec_t(pr, PXte)
        if a.recon == "slot" and a.enc == "image" and hasattr(model, "enc"):     # unsupervised slot identity consistency
            with torch.no_grad():
                model.enc._aux = torch.zeros((), device=dev); model(E1t[:2000], E2t[:2000], ECt[:2000])
                slot0w = round(float(getattr(model.enc, "_slotstat", float("nan"))), 3); model.enc._aux = None
        else:
            slot0w = None
        r = dict(train_mae=round((pr_tr - Dt[:4000]).abs().mean().item(), 3), slot0_is_worker=slot0w, params=NPARAM,
                 test_mae=round((pr - EDt).abs().mean().item(), 3),
                 test_corr=round(float(np.corrcoef(pr.cpu(), ED)[0, 1]), 3), nskip=nskip)
        if a.evalevery > 0:                                        # best point on the held-out eval curve (incl. final eval)
            r["best_mae"] = round(min(best_m, r["test_mae"]), 3)
            r["best_corr"] = round(max(best_c, r["test_corr"]), 3)
        if a.enc == "pureimage" and a.readout in ("xattn", "slotattn") and name == "integ":     # SLOT DIAGNOSTIC on test maps
            with torch.no_grad():
                model(E1t[:1024], E2t[:1024], ECt[:1024])
                att = model.enc._lastattn; im = model.enc._lastimg.flatten(2)     # (B,K,ncell), (B,C,ncell)  (last enc call = goal batch)
                ent = -(att * (att + 1e-9).log()).sum(-1).mean(0)                 # (K,) mean attention entropy (max = ln 49 = 3.89)
                top = att.argmax(-1)                                              # (B,K) argmax cell per slot
                names = ["wall", "worker", "crate", "gate", "gateopen"] + [f"lever{l}" for l in range(a.nlever)] + ["plate", "chuteN", "chuteS", "chuteW", "chuteE"]
                hist = {}
                for k in range(att.shape[1]):
                    ch = im.gather(2, top[:, k][:, None, None].expand(-1, im.shape[1], -1)).squeeze(-1)   # (B,C) channels lit at slot k's argmax cell
                    lit = (ch > 0.75).float().mean(0)                              # fraction of examples where argmax cell has channel c lit
                    empty = float(((ch[:, 1:] > 0.75).sum(1) == 0).float().mean())
                    hist[f"slot{k}"] = {"H": round(float(ent[k]), 2), "empty": round(empty, 2),
                                        **{names[c]: round(float(lit[c]), 2) for c in range(im.shape[1]) if float(lit[c]) >= 0.05}}
                r["slotdiag"] = hist
        if a.Rtrain and a.Rtrain < a.Rmax:
            far = EDt > a.Rtrain
            r["mae_within"] = round((pr[~far] - EDt[~far]).abs().mean().item(), 3)
            if far.any():
                r["mae_beyond"] = round((pr[far] - EDt[far]).abs().mean().item(), 3)
                r["corr_beyond"] = round(float(np.corrcoef(pr[far].cpu(), EDt[far].cpu())[0, 1]), 3) if far.sum() > 2 else None
        out[name] = r
        if a.save:      # weights + full args: reload via the same class after rebuilding with these args
            torch.save(dict(state_dict=model.state_dict(), args=vars(a), model_name=name, result=r),
                       f"{a.save}_{name}.pt")
        if a.dumppred:  # held-out pool: true BFS distance + prediction per pair (calibration plots)
            np.savez(f"{a.dumppred}_{name}.npz", d_true=ED.astype(np.float32),
                     d_pred=pr.cpu().numpy().astype(np.float32))
    print("RESULT " + json.dumps(dict(G=a.G, ngate=a.ngate, nlever=a.nlever, nmaps=a.nmaps, split=a.split,
                                      Rtrain=a.Rtrain, globalbase=a.globalbase, enc=a.enc, markeraux=a.markeraux,
                                      markerheads=a.markerheads, bindmode=a.bindmode, readout=a.readout, cnnw=a.cnnw, cnndepth=a.cnndepth, steps=a.steps, seed=a.seed,
                                      d=a.d, layers=a.layers, baselayers=(a.baselayers if a.baselayers >= 0 else a.layers),
                                      heads=a.heads, T=a.T, lr=a.lr, latentnorm=a.latentnorm, gradclip=a.gradclip, seewalls=a.seewalls,
                                      recon=a.recon, reconw=a.reconw, hardattn=a.hardattn, slots=a.slots, norecall=a.norecall, wirepath=a.wirepath, supbind=a.supbind, coordconv=a.coordconv, cnnk=a.cnnk, slotiters=a.slotiters, slotnoise=a.slotnoise, warmup=a.warmup, attnent=a.attnent, attnovl=a.attnovl, slotln=a.slotln, cnnmix=a.cnnmix, cosine=a.cosine, bs=a.bs, curriculum=int(a.curriculum), basepool=a.basepool, fgmask=a.fgmask, objch=a.objch,
                                      gatesopen=int(a.gatesopen), nopush=int(a.nopush), wire1=int(a.wire1), noplate=int(a.noplate),
                                      tag=a.tag, poolq=a.poolq, evalevery=a.evalevery, **out)), flush=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--G", type=int, default=7); ap.add_argument("--ngate", type=int, default=3)
    ap.add_argument("--nlever", type=int, default=2); ap.add_argument("--nchute", type=int, default=1)
    ap.add_argument("--nmaps", type=int, default=12); ap.add_argument("--npairs", type=int, default=400)
    ap.add_argument("--nwire", type=int, default=6); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--probe", action="store_true"); ap.add_argument("--train", action="store_true")
    ap.add_argument("--poolq", type=int, default=300); ap.add_argument("--Rmax", type=int, default=24)
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3); ap.add_argument("--d", type=int, default=64)
    ap.add_argument("--layers", type=int, default=3); ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--split", choices=["map", "wire"], default="map")
    ap.add_argument("--enc", choices=["factored", "bmask", "marker", "image", "pureimage"], default="factored",
                    help="worker/crate input: factored index | bmask lossless canvas | marker canvas-binding | image CNN+readout")
    ap.add_argument("--slots", type=int, default=8, help="--enc pureimage: number of generic learned slot queries")
    ap.add_argument("--supbind", type=float, default=0.0, help="--enc pureimage: CEILING probe, Hungarian-matched supervised binding aux weight")
    ap.add_argument("--wirepath", type=int, default=0, help="--enc pureimage: draw lever->gate wiring as an L-shaped 0.5 path in the lever channel")
    ap.add_argument("--cnnk", type=int, default=3, help="--enc pureimage: conv kernel size (1 = per-pixel MLP, no spatial mixing)")
    ap.add_argument("--coordconv", type=int, default=0, help="--enc pureimage: add x/y coordinate channels to the CNN input")
    ap.add_argument("--objch", type=int, default=0, help="--enc pureimage: add an object-vs-background input channel (feature, not a mask)")
    ap.add_argument("--fgmask", type=int, default=0, help="--enc pureimage xattn: objectness hint for slots. 1 = attend only to lit (object) cells; 2 = learned additive bias on object cells")
    ap.add_argument("--attnovl", type=float, default=0.0, help="--enc pureimage xattn: slot attention-overlap penalty (diversity)")
    ap.add_argument("--slotln", type=int, default=0, help="--enc pureimage xattn: LayerNorm on slot tokens")
    ap.add_argument("--cnnmix", type=int, default=0, help="--enc pureimage: 1x1 first conv, 3x3 afterwards")
    ap.add_argument("--cosine", type=int, default=0, help="cosine lr decay to 5% over training (after warmup)")
    ap.add_argument("--attnent", type=float, default=0.0, help="--enc pureimage xattn: attention-entropy penalty weight (sharpen slots)")
    ap.add_argument("--warmup", type=int, default=0, help="linear lr warmup steps (0 = off)")
    ap.add_argument("--slotiters", type=int, default=3, help="--readout slotattn: number of competitive attention iterations")
    ap.add_argument("--slotnoise", type=int, default=0, help="--readout slotattn: sample slot init around learned mu (train only)")
    ap.add_argument("--readout", choices=["xattn", "convspatial", "convpool", "pixels", "fgpix", "slotattn"], default="xattn",
                    help="--enc image: how factor tokens read the CNN feature map (xattn=cross-attention)")
    ap.add_argument("--cnnw", type=int, default=32, help="--enc image: CNN hidden width")
    ap.add_argument("--cnndepth", type=int, default=2, help="--enc image: number of conv layers")
    ap.add_argument("--markeraux", type=float, default=0.0, help="marker: aux loss decoding bound token -> true cell (direct binding signal)")
    ap.add_argument("--markerheads", type=int, default=4, help="marker: attention heads for the binding")
    ap.add_argument("--bindmode", choices=["attn", "gather", "slot"], default="attn",
                    help="marker binding: attn(+optional aux) | gather(unsup value-match) | slot(unsup slot-attn)")
    ap.add_argument("--Rtrain", type=int, default=0, help="cap TRAINING pair distance (0 = Rmax); eval uses Rmax")
    ap.add_argument("--globalbase", type=int, default=0, help="add learned base once to the output (length-gen fix)")
    ap.add_argument("--nobaseline", action="store_true", help="skip the scalar-head baseline")
    ap.add_argument("--symbaseline", action="store_true", help="also train the symmetric-embedding baseline")
    ap.add_argument("--symonly", action="store_true", help="train ONLY the symmetric-embedding baseline")
    ap.add_argument("--wire1", action="store_true", help="ladder: each lever wired to exactly one gate")
    ap.add_argument("--noplate", action="store_true", help="ladder: disable the pressure plate")
    ap.add_argument("--nopush", action="store_true", help="ladder: crate is a static obstacle")
    ap.add_argument("--gatesopen", action="store_true", help="ladder: gates always open, levers inert, bits pinned 0")
    ap.add_argument("--curriculum", type=int, default=0, help="phases: easy wiring -> no plate -> full")
    ap.add_argument("--arrive", type=float, default=0.0, help="integ: arrival-loss weight (park at goal)")
    ap.add_argument("--Tmin", type=int, default=-1, help="integ: anytime training budget ~ U[Tmin, T]")
    ap.add_argument("--Ttest", type=int, default=-1, help="integ: eval loop count (-1 = T)")
    ap.add_argument("--resprox", type=int, default=0, help="predict d - factorized proxy")
    ap.add_argument("--bellman", type=float, default=0.0, help="Bellman self-consistency on unlabeled pairs")
    ap.add_argument("--bellpairs", type=int, default=12000); ap.add_argument("--bellbs", type=int, default=48)
    ap.add_argument("--ematarget", type=int, default=0); ap.add_argument("--emam", type=float, default=0.995)
    ap.add_argument("--bellwarm", type=int, default=0, help="ramp bellman weight 0->full over the first 40% of training")
    ap.add_argument("--bellstart", type=float, default=0.0, help="activate bellman only after this fraction of training")
    ap.add_argument("--decodehead", type=int, default=0, help="ablation: decode final state instead of accumulating")
    ap.add_argument("--norecall", type=int, default=0, help="ablation: no re-injection of goal/start")
    ap.add_argument("--iqeonly", action="store_true", help="train ONLY the torchqmet IQE baseline")
    ap.add_argument("--mrnonly", action="store_true", help="train ONLY the torchqmet MRNFixed baseline")
    ap.add_argument("--scalaronly", action="store_true", help="train ONLY the scalar-head baseline")
    ap.add_argument("--scalarsp", type=int, default=0, help="softplus on the scalar-head output (matches reference concat_mlp)")
    ap.add_argument("--evalevery", type=int, default=0, help="eval on the held-out pool every N steps; RESULT gains best_mae/best_corr")
    ap.add_argument("--lencurr", type=int, default=0, help="length curriculum: train range grows 8 -> 12 -> Rtrain")
    ap.add_argument("--heads", type=int, default=4, help="attention heads for every transformer block (d must be divisible)")
    ap.add_argument("--baselayers", type=int, default=-1, help="mix-block depth for sym/qmet/scalar baselines (-1 = use --layers; 0 = no transformer)")
    ap.add_argument("--basepool", choices=["mean", "flat"], default="mean", help="baseline pooling: mean over tokens | flatten all tokens -> MLP")
    ap.add_argument("--latentnorm", type=int, default=0, help="LayerNorm on baseline latents before the metric head (fixes MRN nan; fair, swept)")
    ap.add_argument("--gradclip", type=float, default=0.0, help="clip grad norm before opt.step (0=off; general nan safety, applied to all models)")
    ap.add_argument("--recon", choices=["none", "tied", "slot"], default="none",
                    help="image xattn: UNSUPERVISED reconstruction aux from the model's OWN input channels. tied: slot i decodes "
                         "its own entity channel; slot: slots jointly+exclusively explain the entity mask (no identity given)")
    ap.add_argument("--reconw", type=float, default=1.0, help="reconstruction aux weight")
    ap.add_argument("--hardattn", type=int, default=0, help="image xattn: straight-through hard (top-1 cell) attention read")
    ap.add_argument("--seewalls", type=int, default=0, help="encode open NON-gate cross doorways as tokens (else only via the pooled wall token)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--save", default="", help="prefix: save final weights+args to <prefix>_<model>.pt")
    ap.add_argument("--dumppred", default="", help="prefix: save held-out (d_true, d_pred) to <prefix>_<model>.npz")
    a = ap.parse_args()
    if a.probe: probe(a)
    if a.train: train(a)

if __name__ == "__main__":
    main()

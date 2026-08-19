"""Switchyard: a game-like discrete gridworld whose distance function is shaped by INTERDEPENDENT factors.

A worker pushes a crate around a four-room yard. Gates (open/closed bits) block both of them; wall levers
each XOR-toggle a wired subset of gates; a floor pressure plate forces its wired gates open while the crate
sits on it; some cells are one-way chutes for the worker. State = (worker_cell, crate_cell, gate_bits);
a task is to reach a full target configuration in the fewest moves. Exact geodesics via BFS.

Every mechanic is borrowed from a published benchmark (see paper/switchyard.md for provenance); the novelty
is only that they are wired into one dependency web, so distances do not factorize over the state components.

This module is self-contained (numpy only): environment, BFS solver with action-labeled optimal paths,
and an instance generator/exporter used by reports/switchyard_playable.html.

Usage:
    python switchyard_env.py --export instances.json --n 10     # 10 curated instances + optimal solutions
"""
import argparse, collections, json, numpy as np

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))                        # N S W E; extra action: PULL (on a lever)
DIRNAMES = ("N", "S", "W", "E")


class Yard:
    """One map instance: static layout + wiring config. States are (worker_cell_id, crate_cell_id, bits)."""

    def __init__(self, G=7, ngate=3, nlever=2, nchute=1, rng=None, wire_rng=None,
                 wire1=False, noplate=False, nopush=False, gatesopen=False):
        """The four keyword flags are the COMPLEXITY LADDER, ported to match Jan's switchyard.py
        line for line so our rungs are his rungs:
            gatesopen  gate dynamics disabled, levers inert, bits pinned open   (L0, L1)
            nopush     the crate is a static obstacle                           (L0)
            wire1      each lever wired to exactly one gate, no XOR overlap     (L2)
            noplate    the pressure plate is disabled                           (L2, L3)
        L0 = gatesopen+nopush, L1 = gatesopen, L2 = wire1+noplate+nchute0,
        L3 = noplate+nchute0, L4 = nchute0, L5 = full.
        """
        self.G = G; self.D = ngate; rng = rng or np.random.default_rng(0)
        self.wire1, self.noplate, self.nopush, self.gatesopen = wire1, noplate, nopush, gatesopen
        wrng = wire_rng or rng
        self.wall = np.zeros((G, G), bool)
        wc, wr = G // 2, G // 2                                   # four rooms: one vertical + one horizontal wall
        self.wall[:, wc] = True; self.wall[wr, :] = True
        gaps = [(int(rng.integers(0, wr)), wc), (int(rng.integers(wr + 1, G)), wc),
                (wr, int(rng.integers(0, wc))), (wr, int(rng.integers(wc + 1, G)))]
        for arm in range(4):                                      # extra always-open gap per arm w.p. 0.5 (connectivity)
            if rng.random() < 0.5:
                r, c = [(int(rng.integers(0, wr)), wc), (int(rng.integers(wr + 1, G)), wc),
                        (wr, int(rng.integers(0, wc))), (wr, int(rng.integers(wc + 1, G)))][arm]
                if (r, c) not in gaps: self.wall[r, c] = False
        for r, c in gaps: self.wall[r, c] = False
        gi = rng.permutation(4)[:ngate]
        self.gates = [gaps[i] for i in gi]                        # gate g sits in a gap cell; open iff bit g (or plate)
        free = [(r, c) for r in range(G) for c in range(G) if not self.wall[r, c] and (r, c) not in gaps]
        pick = rng.permutation(len(free))
        self.levers = [free[int(i)] for i in pick[:nlever]]
        self.plate = free[int(pick[nlever])]
        self.chutes = {}                                          # cell -> sole allowed worker entry direction index
        for i in range(nchute):
            self.chutes[free[int(pick[nlever + 1 + i])]] = int(rng.integers(4))
        if wire1:                                                 # ladder L2: one distinct gate per lever
            self.wiring = [(1 << (l % ngate)) if ngate else 0 for l in range(nlever)]
        else:
            self.wiring = [(1 + int(wrng.integers((1 << ngate) - 1))) if ngate else 0 for _ in range(nlever)]
        self.platemask = 0 if (noplate or not ngate) else (1 + int(wrng.integers((1 << ngate) - 1)))
        self.cells = [(r, c) for r in range(G) for c in range(G) if not self.wall[r, c] or (r, c) in gaps]
        self.cid = {rc: i for i, rc in enumerate(self.cells)}

    # ---------------------------------------------------------------- dynamics
    def open_gates(self, bits, crate_cell):
        if self.gatesopen: return (1 << self.D) - 1               # ladder L0/L1: gate dynamics disabled
        eff = bits
        if crate_cell == self.plate: eff |= self.platemask        # plate held down forces its gates open
        return eff

    def passable(self, rc, bits, crate_cell, came_dir=None, is_crate=False):
        r, c = rc
        if not (0 <= r < self.G and 0 <= c < self.G): return False
        if self.wall[r, c] and rc not in self.gates: return False
        if rc in self.gates:
            g = self.gates.index(rc)
            if not (self.open_gates(bits, crate_cell) >> g) & 1: return False
        if not is_crate and rc in self.chutes and came_dir is not None and came_dir != self.chutes[rc]:
            return False                                          # chute: worker may enter only along its direction
        return True

    def neighbours(self, s, with_actions=False):
        """Successors of state s = (worker_id, crate_id, bits). Actions: N/S/W/E moves (auto-push) + PULL."""
        (wr_, cr_, bits) = s; out = []
        w = self.cells[wr_]; b = self.cells[cr_]
        for d, (dr, dc) in enumerate(DIRS):
            nw = (w[0] + dr, w[1] + dc)
            if nw == b:                                           # push attempt
                if self.nopush: continue                          # ladder L0: crate is a static obstacle
                nb = (b[0] + dr, b[1] + dc)
                if self.passable(nb, bits, b, None, True) and nb not in self.chutes and nb != nw:
                    if self.passable(nw, bits, nb, d):
                        out.append(((self.cid[nw], self.cid[nb], bits), DIRNAMES[d]))
            elif self.passable(nw, bits, b, d):
                out.append(((self.cid[nw], cr_, bits), DIRNAMES[d]))
        if w in self.levers and not self.gatesopen:               # pull: XOR the lever's wiring into the gate bits
            out.append(((wr_, cr_, bits ^ self.wiring[self.levers.index(w)]), "PULL"))
        return out if with_actions else [x for x, _ in out]

    # ---------------------------------------------------------------- solver
    def bfs(self, src, maxnodes=300000):
        dist = {src: 0}; dq = collections.deque([src])
        while dq and len(dist) < maxnodes:
            u = dq.popleft()
            for v in self.neighbours(u):
                if v not in dist: dist[v] = dist[u] + 1; dq.append(v)
        return dist

    def solve(self, src, goal, maxnodes=300000):
        """Optimal action sequence src -> goal, or None. Returns list of (action, state)."""
        parent = {src: None}; dq = collections.deque([src])
        while dq:
            u = dq.popleft()
            if u == goal: break
            for v, a in self.neighbours(u, with_actions=True):
                if v not in parent:
                    parent[v] = (u, a); dq.append(v)
                    if len(parent) > maxnodes: return None
        if goal not in parent: return None
        path = []
        cur = goal
        while parent[cur] is not None:
            prev, act = parent[cur]; path.append((act, cur)); cur = prev
        return path[::-1]

    def rand_state(self, rng):
        while True:
            wr_, cr_ = int(rng.integers(len(self.cells))), int(rng.integers(len(self.cells)))
            if wr_ == cr_: continue
            if self.cells[cr_] in self.chutes: continue
            return (wr_, cr_, int(rng.integers(1 << self.D)))

    # ---------------------------------------------------------------- export
    def spec(self):
        return dict(G=self.G,
                    walls=[[r, c] for r in range(self.G) for c in range(self.G)
                           if self.wall[r, c] and (r, c) not in self.gates],
                    gates=[[r, c] for r, c in self.gates],
                    levers=[[r, c] for r, c in self.levers], wiring=self.wiring,
                    plate=[*self.plate], platemask=self.platemask,
                    chutes=[[r, c, d] for (r, c), d in self.chutes.items()],
                    cells=[[r, c] for r, c in self.cells])


def curate_instances(n=10, seed=0, dmin=7, dmax=15):
    """n instances whose optimal solution uses the coupling (a lever pull or a crate push), with solutions."""
    out = []; rng = np.random.default_rng(seed); m = 0
    while len(out) < n:
        yard = Yard(rng=np.random.default_rng(1000 + m), wire_rng=np.random.default_rng(5000 + m)); m += 1
        src = yard.rand_state(rng)
        dist = yard.bfs(src)
        cands = [s for s, d in dist.items() if dmin <= d <= dmax]
        if not cands: continue
        goal = cands[int(rng.integers(len(cands)))]
        sol = yard.solve(src, goal)
        if sol is None: continue
        acts = [a for a, _ in sol]
        pushes = sum(1 for i, (a, st) in enumerate(sol) if st[1] != (sol[i - 1][1][1] if i else src[1]))
        if "PULL" not in acts and pushes == 0: continue           # keep only instances that exercise the coupling
        out.append(dict(map=yard.spec(), start=list(src), goal=list(goal), optimal=len(sol),
                        solution=[[a, list(s)] for a, s in sol]))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", default="instances.json"); ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    inst = curate_instances(a.n, a.seed)
    json.dump(inst, open(a.export, "w"))
    print(f"wrote {len(inst)} instances to {a.export}; optimal lengths: {[i['optimal'] for i in inst]}")

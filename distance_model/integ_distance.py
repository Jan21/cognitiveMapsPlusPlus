"""Integration-based distance model on a coupling gridworld.

Trains a recall-flow integrator to predict the BFS geodesic distance between two states, and reports DISTANCE
ACCURACY (MAE, correlation) and GENERALIZATION to held-out constraint configurations. Input is either factored
(--enc factored) or an IMAGE (--enc bmask|marker: agents rendered onto a GxG canvas, constraints in a key token).

State = [positions(N), mobility_key (base-4 over gated agents 1..N-1), link_key (bitmask over agent pairs)]. Linked
agents move as a rigid group, so coupling couples factors and reduces the reachable set. --heldout {combo,links2,dofhi,
dofhi2} holds out constraint configurations to test generalization. No latent-space / DOF analysis -- distance only.
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

def bfs_local(src, N, G, Rmax, maxnodes=8000):
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
    def emb(self, s, role):
        ids = torch.arange(self.n, device=s.device)
        return self.enc(s) + self.fid(ids)[None] + self.role(torch.tensor(role, device=s.device))
    def forward(self, s, g):
        n = self.n; zs = self.emb(s, 0); zg = self.emb(g, 1); base = torch.cat([zg, self.emb(s, 2)], 1)
        tok = torch.cat([zs, base], 1); cost = torch.zeros(s.shape[0], device=s.device)
        for _ in range(self.T):
            z = self.block(tok); cost = cost + (z[:, :n] - tok[:, :n]).norm(dim=-1).sum(-1)
            tok = torch.cat([z[:, :n], base], 1)                   # re-inject goal + start (recall)
        return F.softplus(self.scale) * cost


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nag", type=int, default=4); ap.add_argument("--G", type=int, default=6)
    ap.add_argument("--Rmax", type=int, default=12); ap.add_argument("--T", type=int, default=14)
    ap.add_argument("--d", type=int, default=128); ap.add_argument("--layers", type=int, default=4); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=40000); ap.add_argument("--bs", type=int, default=128); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--poolq", type=int, default=2000); ap.add_argument("--nquery", type=int, default=80); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--enc", choices=["factored", "bmask", "marker"], default="factored")
    ap.add_argument("--inject", type=int, default=0, help="per-agent constraint injection into position tokens")
    ap.add_argument("--heldout", choices=["", "combo", "links2", "dofhi", "dofhi2"], default="")
    a = ap.parse_args()
    global ENC, INJECT, HELDMODE, _HELD; ENC = a.enc; INJECT = bool(a.inject); HELDMODE = a.heldout; N = a.nag
    if a.heldout == "combo":
        _HELD = {((0, 0, 0), 1), ((1, 1, 1), 3), ((3, 0, 0), 2), ((0, 2, 0), 4), ((2, 2, 2), 7), ((1, 0, 2), 5)}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    S1, S2, D = build_pool(N, a.G, a.Rmax, a.poolq, rng)
    S1t, S2t, Dt = (torch.as_tensor(x, device=dev) for x in (S1, S2, D))
    print(f"nag={N} G={a.G} enc={a.enc} heldout={a.heldout} pool={len(S1)} maxd={int(D.max())}", flush=True)
    model = Integrator(N, a.G, a.d, a.heads, a.layers, a.T).to(dev)
    opt = torch.optim.Adam(model.parameters(), a.lr)
    for step in range(a.steps):
        b = torch.randint(0, len(S1t), (a.bs,), device=dev)
        loss = F.smooth_l1_loss(model(S1t[b], S2t[b]), Dt[b])
        opt.zero_grad(); loss.backward(); opt.step()
        if step % (a.steps // 6) == 0:
            with torch.no_grad():
                bb = torch.randint(0, len(S1t), (2000,), device=dev); mae = (model(S1t[bb], S2t[bb]) - Dt[bb]).abs().mean().item()
            print(f"step {step} loss {loss.item():.3f} trainMAE {mae:.3f}", flush=True)
    model.eval()

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
            pr = model(torch.as_tensor(np.repeat(q[None], len(tg), 0), device=dev), torch.as_tensor(tg, device=dev)).cpu().numpy()
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
                dl = model(torch.as_tensor(np.repeat(q[None], len(legal), 0), device=dev), torch.as_tensor(np.array(legal), device=dev)).cpu().numpy()
                di = model(torch.as_tensor(np.repeat(q[None], len(ills), 0), device=dev), torch.as_tensor(np.array(ills), device=dev)).cpu().numpy()
                aucs.append(float(np.mean([[1.0 if x < y else 0.0 for y in di] for x in dl])))
        preds, trues = np.array(preds), np.array(trues)
        mae = round(float(np.abs(preds - trues).mean()), 3)
        corr = round(float(np.corrcoef(preds, trues)[0, 1]), 3) if preds.std() > 1e-6 else None
        px = round(float(np.mean(aucs)), 3) if aucs else None
        return dict(dist_mae=mae, dist_corr=corr, px_auc=px)

    res = dict(nag=N, G=a.G, enc=a.enc, inject=a.inject, heldout=a.heldout, steps=a.steps, seed=a.seed,
               d=a.d, layers=a.layers, T=a.T)
    res["test" if a.heldout else "all"] = evaluate("test" if a.heldout else "all")     # generalization (held-out)
    if a.heldout: res["train"] = evaluate("train")                                     # in-distribution accuracy
    print("RESULT " + json.dumps(res), flush=True)


if __name__ == "__main__":
    main()

"""Back to the ORIGINAL training signal (local only), now with the ATTENTION distance head, then
read local dimension with VGT.

No geodesic supervision. We only teach:
  legal 1-step neighbour  -> distance 1     (isometry on the move graph)
  illegal 1-step / random -> distance >= margin  (repel non-neighbours)
This is the cheap, scalable signal (no all-pairs BFS). The question: does a GATED attention distance
over factored components turn that local signal into a metric whose local correlation dimension (VGT)
recovers the true number of degrees of freedom -- where a plain Euclidean factored embedding does not?

Env (factored, guarded): NAG agents each on a cycle(G) (a 1-D component) + a shared key in path(Kk).
Agent i is movable iff key > i, so the number of movable agents = key, giving a clean DOF ladder in
one model. The key moves freely (cost 1); moving a frozen agent is illegal (repelled).

Distance head (gated-L1 over components):
    d(x,y) = sum_i w_i(key_x,key_y) * ||agent_emb_i(x) - agent_emb_i(y)|| + ||key_emb(x)-key_emb(y)||
    w_i = softplus(MLP(key pair))   -- the gate lets a FROZEN agent's move be pushed far (large w),
                                       a FREE agent's move cost ~1 (small w). Euclidean has no gate.

VGT: jitter the agents around a probe, compute head-distance to the probe, take the k nearest, read
the correlation-dimension slope. Frozen agents' jitters land far (repelled) and drop out of the
nearest set, so the dimension = number of free agents = key.
"""
import argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

NAG, G, Kk = 3, 12, 4                      # 3 agents on 12-cycles, key in 0..3 ; free_i = key>i
MARGIN = 15.0

def free_mask(key):                        # (B,NAG) bool: agent i free iff key>i
    return key[:, None] > np.arange(NAG)[None]

def rand_states(n, rng):
    pos = rng.integers(0, G, (n, NAG)); key = rng.integers(0, Kk, (n,))
    return np.concatenate([pos, key[:, None]], 1)

def legal_neighbour(s, rng):
    n = s.shape[0]; out = s.copy(); key = s[:, NAG]
    fr = free_mask(key)
    for b in range(n):
        moves = []
        if key[b] - 1 >= 0: moves.append(("k", -1))
        if key[b] + 1 <= Kk - 1: moves.append(("k", +1))
        for i in range(NAG):
            if fr[b, i]: moves += [("a", i, -1), ("a", i, +1)]
        m = moves[rng.integers(0, len(moves))]
        if m[0] == "k": out[b, NAG] = key[b] + m[1]
        else: out[b, m[1]] = (s[b, m[1]] + m[2]) % G
    return out

def illegal_neighbour(s, rng):             # move a FROZEN agent one step (an illegal transition)
    n = s.shape[0]; out = s.copy(); key = s[:, NAG]; fr = free_mask(key); ok = np.zeros(n, bool)
    for b in range(n):
        frozen = [i for i in range(NAG) if not fr[b, i]]
        if frozen:
            i = frozen[rng.integers(0, len(frozen))]; out[b, i] = (s[b, i] + (1 if rng.random() < .5 else -1)) % G; ok[b] = True
    return out, ok

# ---------- heads ----------
class AttnDist(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.ag = nn.ModuleList([nn.Embedding(G, d) for _ in range(NAG)]); self.key = nn.Embedding(Kk, d)
        self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, NAG))
    def forward(self, x, y):
        kx, ky = self.key(x[:, NAG]), self.key(y[:, NAG])
        w = F.softplus(self.gate(torch.cat([kx + ky, (kx - ky).abs()], -1)))          # (B,NAG) symmetric
        dag = torch.stack([torch.norm(self.ag[i](x[:, i]) - self.ag[i](y[:, i]), dim=-1) for i in range(NAG)], -1)
        return (w * dag).sum(-1) + torch.norm(kx - ky, dim=-1)

class Euclid(nn.Module):
    def __init__(self, d=32):
        super().__init__()
        self.ag = nn.ModuleList([nn.Embedding(G, d) for _ in range(NAG)]); self.key = nn.Embedding(Kk, d)
    def emb(self, x):
        return torch.cat([self.ag[i](x[:, i]) for i in range(NAG)] + [self.key(x[:, NAG])], -1)
    def forward(self, x, y): return torch.norm(self.emb(x) - self.emb(y), dim=-1)

def train(head, steps, rng, lr=3e-3, bs=256):
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    for _ in range(steps):
        s = rand_states(bs, rng); nb = legal_neighbour(s, rng); il, ok = illegal_neighbour(s, rng); rd = rand_states(bs, rng)
        st, nt, it, rt = (torch.tensor(a) for a in (s, nb, il, rd))
        d_nb = head(st, nt); loss = ((d_nb - 1.0) ** 2).mean()                          # neighbour -> 1
        d_rd = head(st, rt); loss = loss + F.softplus(MARGIN - d_rd).mean()             # random -> far
        okm = torch.tensor(ok)
        if okm.any():
            d_il = head(st, it)[okm]; loss = loss + F.softplus(MARGIN - d_il).mean()    # illegal move -> far
        opt.zero_grad(); loss.backward(); opt.step()
    return head

def vgt(dist, lo=0.05, hi=0.6, mlo=6):
    d = np.sort(dist[dist > 1e-9]); N = d.size
    if N < 12: return np.nan
    a, b = max(mlo, int(lo * N)), int(hi * N)
    if b - a < 5 or d[b - 1] - d[a] < 1e-6: return np.nan
    return float(np.polyfit(np.log(d[a:b]), np.log(np.arange(1, N + 1, dtype=float)[a:b]), 1)[0])

@torch.no_grad()
def local_dim(head, probe, W, M, rng):
    """Jitter ALL agents; the learned distance ranks free-agent variation near and frozen far, so the
    sorted distances are bimodal (free cluster, then a big jump to frozen). Cut the ball at that gap
    -- the metric excludes the frozen coordinates for us -- then read the correlation dim of the free
    cluster. No graph, no knowledge of which agents are free."""
    pool = np.tile(probe, (M, 1))
    for i in range(NAG): pool[:, i] = (pool[:, i] + rng.integers(-W, W + 1, M)) % G
    d = np.sort(head(torch.tensor(pool), torch.tensor(probe[None]).repeat(M, 1)).numpy())
    d = d[d > 1e-6]
    if d.size < 40: return np.nan
    ld = np.log(d); gap = ld[1:] - ld[:-1]                                               # multiplicative gaps
    lo, hi = 20, int(0.9 * d.size)
    cand = np.where(gap[lo:hi] > 0.8)[0]                                                  # FIRST big jump = free/frozen
    if cand.size: d = d[:lo + int(cand[0]) + 1]                                           # keep free cluster only
    return vgt(d)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=5000); ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(); rng = np.random.default_rng(args.seed)
    probes = [(np.array([4, 4, 4, m]), m) for m in (1, 2, 3)]                            # key=m -> m free agents
    print(f"NAG={NAG} G={G} Kk={Kk} ; probe key=m has m free agents -> true local dim = m")
    print(f"{'head':<9}" + "".join(f"  key={m}(dim{m})" for _, m in probes))
    heads = {}
    for name, mk in [("attn", AttnDist), ("euclid", Euclid)]:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        h = train(mk(), args.steps, rng); heads[name] = h
        row = "".join(f"{local_dim(h, p, W=5, M=60000, rng=rng):>11.2f}" for p, _ in probes)
        print(f"{name:<9}{row}")

    # ---- diagnostics on the attention head: gate weights + per-agent 1-step distance ----
    h = heads["attn"]
    print("\n[diag] attn: per-agent gate weight w_i(key) and 1-step move distance at each probe")
    with torch.no_grad():
        for p, m in probes:
            pt = torch.tensor(p[None])
            kx = h.key(pt[:, NAG]); w = F.softplus(h.gate(torch.cat([kx + kx, (kx - kx).abs()], -1)))[0]
            steps = []
            for i in range(NAG):
                q = p.copy(); q[i] = (q[i] + 1) % G
                steps.append(float(h(pt, torch.tensor(q[None]))[0]))
            qk = p.copy(); qk[NAG] = min(Kk - 1, p[NAG] + 1)
            dkey = float(h(pt, torch.tensor(qk[None]))[0])
            fr = [("free" if m > i else "froz") for i in range(NAG)]
            print(f"  key={m}: w={[round(float(x),2) for x in w]}  "
                  f"step_d={[f'{steps[i]:.2f}({fr[i]})' for i in range(NAG)]}  key_step={dkey:.2f}")

if __name__ == "__main__":
    main()

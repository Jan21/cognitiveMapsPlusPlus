"""Can a learned distance reproduce a DETOUR geodesic? (the user's example)

State = (pos in 0..9, knob in 0..5). The knob is a free dial: (p,k)->(p,k +-1) always costs 1.
The agent can move ONLY when knob==5: (p,k)->(p +-1,k) costs 1 iff k==5. So to move the agent you
must dial the knob up to 5, step, then dial back. Hence:
    d((0,0),(5,0)) = (0->5 dial) + (0->5 move) + (5->0 dial) = 5 + 5 + 5 = 15.
This detour (a fixed additive cost, active only when the position changes) is what we want the
embedding distance to reproduce, at least approximately.

Only 60 states -> compute the EXACT all-pairs geodesic by BFS and regress it with several heads:
  mds_norm    d = ||Ex - Ey||_2 ,  E = free per-state embedding      (norm readout / self_norm)
  mds_l1      d = ||Ex - Ey||_1 , free embedding                     (L1 norm readout)
  factored    d = wp(kx,ky)*||dpos|| + ||dknob||  (faithful factors, gated additive)
  attn_scalar attention over factor-diff tokens -> a SCALAR distance  (the 'attention' idea)
  mlp_pair    d = softplus(MLP([Ex+Ey, |Ex-Ey|]))  general symmetric  (expressiveness ceiling)

We report RMSE over all pairs and the predicted distance on a handful of diagnostic pairs whose
true geodesic exercises the detour.
"""
import argparse, collections, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

P, K = 10, 6; N = P * K
def dec(i): return i // K, i % K
def enc(p, k): return p * K + k

def build_geo():
    adj = [[] for _ in range(N)]
    for i in range(N):
        p, k = dec(i)
        if k + 1 <= 5: adj[i].append(enc(p, k + 1))
        if k - 1 >= 0: adj[i].append(enc(p, k - 1))
        if k == 5:
            if p + 1 <= 9: adj[i].append(enc(p + 1, k))
            if p - 1 >= 0: adj[i].append(enc(p - 1, k))
    INF = 10 ** 9; geo = np.full((N, N), INF, np.int64)
    for s in range(N):
        dq = collections.deque([s]); geo[s, s] = 0
        while dq:
            u = dq.popleft()
            for v in adj[u]:
                if geo[s, v] == INF: geo[s, v] = geo[s, u] + 1; dq.append(v)
    return geo

# ---------- heads ----------
class MDS(nn.Module):
    def __init__(self, D=16, norm="l2"):
        super().__init__(); self.E = nn.Embedding(N, D); self.norm = norm
    def forward(self, i, j):
        d = self.E(i) - self.E(j)
        return torch.norm(d, p=(2 if self.norm == "l2" else 1), dim=-1)

class Factored(nn.Module):
    def __init__(self, d=24):
        super().__init__()
        self.pos = nn.Embedding(P, d); self.knob = nn.Embedding(K, d)
        self.wp = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
    def forward(self, i, j):
        pi, ki = i // K, i % K; pj, kj = j // K, j % K
        dp = torch.norm(self.pos(pi) - self.pos(pj), dim=-1)
        dk = torch.norm(self.knob(ki) - self.knob(kj), dim=-1)
        w = F.softplus(self.wp(torch.cat([self.knob(ki), self.knob(kj)], -1))).squeeze(-1)
        return w * dp + dk

class AttnSoftmax(nn.Module):
    """BROKEN reference: attention with softmax -> weighted AVERAGE of factors, cannot accumulate."""
    def __init__(self, d=24):
        super().__init__()
        self.pos = nn.Embedding(P, d); self.knob = nn.Embedding(K, d); self.typ = nn.Embedding(2, d)
        self.q = nn.Linear(2 * d, d); self.k = nn.Linear(2 * d, d); self.v = nn.Linear(2 * d, d)
        self.out = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1)); self.d = d
    def _dist(self, i, j):
        pi, ki = i // K, i % K; pj, kj = j // K, j % K
        dpos = self.pos(pi) - self.pos(pj); dknob = self.knob(ki) - self.knob(kj)
        t0 = torch.cat([dpos, self.typ(torch.zeros_like(pi))], -1)
        t1 = torch.cat([dknob, self.typ(torch.ones_like(pi))], -1)
        tok = torch.stack([t0, t1], 1)
        q = self.q(torch.cat([self.knob(ki), self.knob(kj)], -1))[:, None]
        a = torch.softmax((q * self.k(tok)).sum(-1) / self.d ** 0.5, -1)   # <-- softmax = average
        z = (a[..., None] * self.v(tok)).sum(1)
        return F.softplus(self.out(z)).squeeze(-1)
    def forward(self, i, j):
        return 0.5 * (self._dist(i, j) + self._dist(j, i))


class AttnSum(nn.Module):
    """FIX: attention as a SUM of independently-gated per-token distances. sigmoid gates (not
    softmax) so terms accumulate. Tokens: pos-diff, knob-diff, and a DETOUR token whose value is a
    learned function of BOTH knobs and whose gate can fire on position change (query sees dpos)."""
    def __init__(self, d=24):
        super().__init__()
        self.pos = nn.Embedding(P, d); self.knob = nn.Embedding(K, d); self.typ = nn.Embedding(3, d)
        self.q = nn.Linear(2 * d + 2, d)                                   # query sees both knobs + dpos,dknob
        self.k = nn.Linear(2 * d, d)
        self.val = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1)); self.d = d
    def _dist(self, i, j):
        pi, ki = i // K, i % K; pj, kj = j // K, j % K
        ei, ej = self.knob(ki), self.knob(kj)
        dpos = torch.norm(self.pos(pi) - self.pos(pj), dim=-1, keepdim=True)
        dknob = torch.norm(ei - ej, dim=-1, keepdim=True)
        tpos = torch.cat([self.pos(pi) - self.pos(pj), self.typ(torch.zeros_like(pi))], -1)
        tknob = torch.cat([ei - ej, self.typ(torch.ones_like(pi))], -1)
        tdet = torch.cat([ei + ej, self.typ(2 * torch.ones_like(pi))], -1)  # detour: symmetric in knobs
        tok = torch.stack([tpos, tknob, tdet], 1)                          # (B,3,2d)
        q = self.q(torch.cat([ei, ej, dpos, dknob], -1))[:, None]
        gate = torch.sigmoid((q * self.k(tok)).sum(-1) / self.d ** 0.5)    # (B,3) INDEPENDENT gates
        v = F.softplus(self.val(tok)).squeeze(-1)                          # (B,3) >=0 per-token distance
        return (gate * v).sum(-1)                                          # SUM, not average
    def forward(self, i, j):
        return 0.5 * (self._dist(i, j) + self._dist(j, i))


class AttnAdd(nn.Module):
    """Neural-additive form of the same idea: T terms, each an independent sigmoid gate times a
    non-negative value, both computed from the pair context (both knobs + raw factor distances).
    d = sum_t gate_t * value_t. Most transparent 'sum of gated distances'."""
    def __init__(self, d=24, T=6):
        super().__init__()
        self.pos = nn.Embedding(P, d); self.knob = nn.Embedding(K, d)
        self.gate = nn.Sequential(nn.Linear(2 * d + 2, d), nn.GELU(), nn.Linear(d, T))
        self.val = nn.Sequential(nn.Linear(2 * d + 2, d), nn.GELU(), nn.Linear(d, T))
    def _dist(self, i, j):
        pi, ki = i // K, i % K; pj, kj = j // K, j % K
        dpos = torch.norm(self.pos(pi) - self.pos(pj), dim=-1, keepdim=True)
        dknob = torch.norm(self.knob(ki) - self.knob(kj), dim=-1, keepdim=True)
        ctx = torch.cat([self.knob(ki), self.knob(kj), dpos, dknob], -1)
        return (torch.sigmoid(self.gate(ctx)) * F.softplus(self.val(ctx))).sum(-1)
    def forward(self, i, j):
        return 0.5 * (self._dist(i, j) + self._dist(j, i))

class MLPpair(nn.Module):
    def __init__(self, D=24):
        super().__init__()
        self.E = nn.Embedding(N, D)
        self.net = nn.Sequential(nn.Linear(2 * D, 64), nn.GELU(), nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, i, j):
        ex, ey = self.E(i), self.E(j)
        return F.softplus(self.net(torch.cat([ex + ey, (ex - ey).abs()], -1))).squeeze(-1)

def train(head, geo, steps, lr=3e-3):
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    I, J = np.meshgrid(np.arange(N), np.arange(N)); I = I.ravel(); J = J.ravel()
    it = torch.tensor(I); jt = torch.tensor(J); gt = torch.tensor(geo[I, J], dtype=torch.float32)
    for _ in range(steps):
        d = head(it, jt); loss = ((d - gt) ** 2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return head

@torch.no_grad()
def rmse(head, geo):
    I, J = np.meshgrid(np.arange(N), np.arange(N)); I = I.ravel(); J = J.ravel()
    d = head(torch.tensor(I), torch.tensor(J)).numpy()
    return float(np.sqrt(((d - geo[I, J]) ** 2).mean()))

@torch.no_grad()
def pred(head, a, b):
    return float(head(torch.tensor([a]), torch.tensor([b]))[0])

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0); args = ap.parse_args()
    geo = build_geo()
    probes = [("(0,0)->(5,0)", enc(0, 0), enc(5, 0)),      # detour: 15
              ("(0,5)->(5,5)", enc(0, 5), enc(5, 5)),      # both enabled: 5
              ("(0,0)->(0,3)", enc(0, 0), enc(0, 3)),      # dial only: 3
              ("(0,0)->(9,0)", enc(0, 0), enc(9, 0)),      # 5+9+5 = 19
              ("(0,2)->(3,4)", enc(0, 2), enc(3, 4)),      # 3+3+1 = 7
              ("(0,0)->(0,5)", enc(0, 0), enc(0, 5))]      # dial up: 5
    print("true geodesics:", {n: int(geo[a, b]) for n, a, b in probes})
    hdr = "".join(f"{n.split('->')[0]+'>'+n.split('->')[1].split(',')[0][1:]:>10}" for n, _, _ in probes)
    print(f"{'head':<12}{'RMSE':>7}" + "".join(f"{n[1:6]+n[8:11]:>11}" for n, _, _ in probes))
    heads = {
        "mds_l1":       lambda: MDS(norm="l1"),
        "factored":     lambda: Factored(),
        "attn_softmax": lambda: AttnSoftmax(),
        "attn_sum":     lambda: AttnSum(),
        "attn_add":     lambda: AttnAdd(),
    }
    for name, mk in heads.items():
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        h = train(mk(), geo, args.steps)
        row = "".join(f"{pred(h,a,b):>11.1f}" for _, a, b in probes)
        print(f"{name:<12}{rmse(h,geo):>7.2f}{row}")
    print("\n(targets:       " + "".join(f"{int(geo[a,b]):>11d}" for _, a, b in probes) + " )")

if __name__ == "__main__":
    main()

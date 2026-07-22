"""Isolate IMAGE INPUT from 2-D-agent measurement difficulty: N agents as distinct markers on a 1-D
STRIP of G cells + a key token. Agent 0 always free; agent j free iff key>=j; DOF = 1 + #free -> clean
ladder 1..N (steps of 1, easy to measure -- like the factored 1-D case that worked cleanly). Same
recipe: marker encoder (per-component query + marker-id), gated L1 distance (optional key-gate),
multi-scale isometry + repel, gate-guided VGT. If this reads a clean ladder, image input works cleanly
and the 2-D agents were the limiter.
"""
import argparse, json, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=5); ap.add_argument("--G", type=int, default=48)
    ap.add_argument("--d", type=int, default=64); ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20000); ap.add_argument("--K", type=int, default=6)
    ap.add_argument("--lam_iso", type=float, default=3.0); ap.add_argument("--margin", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--keygate", action="store_true")
    ap.add_argument("--factored", action="store_true", help="control: clean per-agent pos embeddings instead of shared image")
    ap.add_argument("--W", type=int, default=16); ap.add_argument("--L", type=int, default=1500)
    ap.add_argument("--M", type=int, default=60000); ap.add_argument("--qlo", type=float, default=0.05)
    ap.add_argument("--qhi", type=float, default=0.6)
    args = ap.parse_args()
    N, G, Kk = args.N, args.G, args.N
    NP = G; VAL_VOCAB = (N + 1) + Kk; MARGIN = args.margin
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def render(s):
        B = s.shape[0]; img = torch.zeros(B, NP + 1, dtype=torch.long, device=dev); ar = torch.arange(B, device=dev)
        for i in range(N): img[ar, s[:, i]] = i + 1
        img[:, NP] = (N + 1) + s[:, N]
        return img
    def rand_states(n, rng): return np.concatenate([rng.integers(0, G, (n, N)), rng.integers(0, Kk, (n, 1))], 1)
    def move_multiscale(s, rng, K):
        out = s.copy(); dist = np.zeros(len(s), np.float32)
        for b in range(len(s)):
            key = s[b, N]
            for j in range(N):
                if j == 0 or key >= j:
                    st = int(rng.integers(-K, K + 1)); out[b, j] = (s[b, j] + st) % G; dist[b] += abs(st)
        return out, dist
    def illegal_neighbour(s, rng):
        out = s.copy(); ok = np.zeros(len(s), bool)
        for b in range(len(s)):
            key = s[b, N]; opts = [("a", j) for j in range(1, N) if key < j]
            if s[b, 0] != 0: opts.append(("k", 1 if key < Kk - 1 else -1))
            if opts:
                m = opts[rng.integers(0, len(opts))]; ok[b] = True
                if m[0] == "a": out[b, m[1]] = (s[b, m[1]] + 1) % G
                else: out[b, N] = key + m[1]
        return out, ok

    class Enc(nn.Module):
        def __init__(self):
            super().__init__()
            self.val = nn.Embedding(VAL_VOCAB, args.d); self.pe = nn.Embedding(NP + 1, args.d)
            self.query = nn.Parameter(torch.randn(N + 1, args.d) * 0.02); self.mk = nn.Embedding(N + 1, args.d)
            self.attn = nn.MultiheadAttention(args.d, args.heads, batch_first=True)
            self.fpos = nn.Embedding(G, args.d); self.fkey = nn.Embedding(Kk, args.d)  # --factored control
            self.gate = nn.Sequential(nn.Linear(2 * args.d, args.d), nn.GELU(), nn.Linear(args.d, 1))
            self.wk = nn.Parameter(torch.zeros(()))
        def components(self, s):
            if args.factored:  # clean per-agent factors: NO shared canvas, NO binding problem
                ac = [self.fpos(s[:, i]) + self.mk(torch.tensor(i, device=dev)) for i in range(N)]
                return torch.stack(ac + [self.fkey(s[:, N])], 1)
            img = render(s); B = img.shape[0]
            tok = self.val(img) + self.pe(torch.arange(NP + 1, device=dev))[None]
            ids = torch.arange(N + 1, device=dev)
            comp, _ = self.attn((self.query + self.mk(ids))[None].expand(B, -1, -1), tok, tok)
            return comp
        def forward(self, x, y):
            cx, cy = self.components(x), self.components(y)
            ax = [cx[:, i] for i in range(N)]; kx = cx[:, N]; ay = [cy[:, i] for i in range(N)]; ky = cy[:, N]
            d = F.softplus(self.wk) * torch.norm(kx - ky, dim=-1)
            if args.keygate:
                kctx = kx + ky; B = x.shape[0]
                for i in range(N):
                    aid = self.mk(torch.tensor(i, device=dev))[None].expand(B, -1)
                    w = F.softplus(self.gate(torch.cat([aid, kctx], -1))).squeeze(-1)
                    d = d + w * torch.norm(ax[i] - ay[i], dim=-1)
            else:
                ctx = torch.stack([a + b for a, b in zip(ax + [kx], ay + [ky])], 0).mean(0)
                for i in range(N):
                    w = F.softplus(self.gate(torch.cat([ax[i] + ay[i], ctx], -1))).squeeze(-1)
                    d = d + w * torch.norm(ax[i] - ay[i], dim=-1)
            return d

    rng = np.random.default_rng(args.seed); torch.manual_seed(args.seed); np.random.seed(args.seed)
    head = Enc().to(dev); opt = torch.optim.Adam(head.parameters(), lr=3e-3)
    for _ in range(args.steps):
        s = rand_states(256, rng); mv, md = move_multiscale(s, rng, args.K); il, ok = illegal_neighbour(s, rng); rd = rand_states(256, rng)
        st, mt, it, rt = (torch.as_tensor(a, device=dev) for a in (s, mv, il, rd)); mdist = torch.as_tensor(md, device=dev)
        loss = args.lam_iso * ((head(st, mt) - mdist) ** 2).mean() + F.softplus(MARGIN - head(st, rt)).mean()
        okm = torch.as_tensor(ok, device=dev)
        if okm.any(): loss = loss + F.softplus(MARGIN - head(st, it)[okm]).mean()
        opt.zero_grad(); loss.backward(); opt.step()

    def vgt(dist, lo=args.qlo, hi=args.qhi, mlo=6):
        dd = np.sort(dist[dist > 1e-9]); Nn = dd.size
        if Nn < 12: return np.nan
        a, b = max(mlo, int(lo * Nn)), int(hi * Nn)
        if b - a < 5 or dd[b - 1] - dd[a] < 1e-6: return np.nan
        return float(np.polyfit(np.log(dd[a:b]), np.log(np.arange(1, Nn + 1, dtype=float)[a:b]), 1)[0])

    @torch.no_grad()
    def measure(probe, W=args.W, M=args.M, L=args.L):
        pt = torch.as_tensor(probe[None], device=dev); steps = []
        for j in range(N):
            q = probe.copy(); q[j] = (q[j] + 1) % G; steps.append(float(head(pt, torch.as_tensor(q[None], device=dev))[0]))
        thr = 5 * min(steps); free = [j for j in range(N) if steps[j] <= thr]
        if not free: return np.nan
        pool = np.tile(probe, (M, 1))
        for j in free: pool[:, j] = (pool[:, j] + rng.integers(-W, W + 1, M)) % G
        out = []
        for k in range(0, M, 20000):
            b = torch.as_tensor(pool[k:k + 20000], device=dev); out.append(head(pt.expand(len(b), -1), b).cpu().numpy())
        dd = np.concatenate(out); order = np.argsort(dd)
        return vgt(np.sort(dd[order[1:L + 1]]))

    head.eval(); rj = np.random.default_rng(args.seed + 1); ladder = []
    for m in range(N):
        probes = [np.concatenate([rj.integers(1, G, N), [m]]) for _ in range(8)]
        vg = [measure(p, rng=rj) if False else measure(p) for p in probes]; vg = [x for x in vg if np.isfinite(x) and 0 < x < 30]
        ladder.append(round(float(np.mean(vg)), 2) if vg else None)
    print("RESULT " + json.dumps(dict(N=N, G=G, d=args.d, steps=args.steps, keygate=args.keygate, factored=args.factored, seed=args.seed, W=args.W, L=args.L, M=args.M, qlo=args.qlo, qhi=args.qhi, ladder=ladder, dof="1+key")), flush=True)

if __name__ == "__main__":
    main()

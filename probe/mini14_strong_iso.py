"""Strengthen the isometry by removing the gate's ability to CHEAT.

Diagnosis: the mini12 gate took the displacement magnitude |Delta e| as input, so it shrank the weight
w for larger moves to keep d = w*||Delta e|| flat -> the embedding never had to stretch (ed(k=6)/ed(k=1)
stayed ~1.1 even with heavy isometry loss). Fix (AttnDist1D_iso): the gate reads ONLY position-sum
features and a pooled position-sum context (which carries the key), never any displacement. Then w
depends on WHICH agent and the key, not on HOW FAR it moved, so isometry must be met by the embedding.

Reports local isometry quality, ring monotonicity, and the VGT ladder; saves plots.
"""
import argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import mini11_estimators as m11
from mini12_gate_guided import train, measure

class AttnDist1D_iso(nn.Module):
    def __init__(self, d=48):
        super().__init__()
        self.pos = nn.ModuleList([nn.Embedding(m11.G, d) for _ in range(m11.N)])
        self.aid = nn.Embedding(m11.N, d); self.key = nn.Embedding(m11.Kk, d)
        self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))     # input: [agent pos-sum, context pos-sum]
        self.wk = nn.Parameter(torch.zeros(()))
    def _embs(self, s):
        ea = [self.pos[i](s[:, i]) + self.aid(torch.full_like(s[:, i], i)) for i in range(m11.N)]
        return ea, self.key(s[:, m11.N])
    def forward(self, x, y):
        ax, kx = self._embs(x); ay, ky = self._embs(y)
        cx, cy = ax + [kx], ay + [ky]
        ctx = torch.stack([a + b for a, b in zip(cx, cy)], 0).mean(0)                   # ONLY sum-pool: no displacement
        d = F.softplus(self.wk) * torch.norm(kx - ky, dim=-1)
        for i in range(m11.N):
            w = F.softplus(self.gate(torch.cat([ax[i] + ay[i], ctx], -1))).squeeze(-1)  # gate sees position+key, not |Delta|
            d = d + w * torch.norm(ax[i] - ay[i], dim=-1)
        return d

def ringness(W):
    p = PCA(n_components=min(W.shape)).fit(W); ev = p.explained_variance_ratio_
    Y = p.transform(W)[:, :2]; c = Y.mean(0)
    ang = np.arctan2(Y[:, 1] - c[1], Y[:, 0] - c[0]); dif = np.diff(np.unwrap(ang))
    return Y, ev, max((dif > 0).mean(), (dif < 0).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=12000); ap.add_argument("--lam_iso", type=float, default=3.0)
    ap.add_argument("--margin", type=float, default=60.0); ap.add_argument("--d", type=int, default=48); args = ap.parse_args()
    N, G = 5, 48; m11.N, m11.G, m11.Kk = N, G, N
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  gate=iso(no-displacement)  lam_iso={args.lam_iso} margin={args.margin} d={args.d}", flush=True)
    rng = np.random.default_rng(0); torch.manual_seed(0); np.random.seed(0)
    head = train(AttnDist1D_iso(d=args.d), args.steps, rng, device, K=6, lam_iso=args.lam_iso, margin=args.margin)

    gd = np.array([min(k, G - k) for k in range(G)])
    print("\n[isometry] corr(embed,graph) k<=8 ; ed(k=1) ; ed(k=6) ; ratio ; ring", flush=True)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    for i in range(N):
        W = head.pos[i].weight.detach().cpu().numpy(); ed = np.linalg.norm(W - W[0], axis=1)
        loc = gd <= 8; corr = np.corrcoef(gd[loc], ed[loc])[0, 1]
        e1, e6 = ed[gd == 1].mean(), ed[gd == 6].mean(); Y, ev, mono = ringness(W)
        print(f"  agent {i}: corr={corr:+.2f}  ed1={e1:.2f} ed6={e6:.2f} ratio={e6/max(e1,1e-6):.2f}  ring={mono:.2f}", flush=True)
        ax = axes.flat[i]; loop = np.vstack([Y, Y[0]])
        ax.plot(loop[:, 0], loop[:, 1], '-', color="#0e8f8a", alpha=.45, lw=1)
        ax.scatter(Y[:, 0], Y[:, 1], c=np.arange(G), cmap="twilight", s=36, zorder=3)
        ax.set_title(f"agent {i}  PC1+2={ev[:2].sum():.2f} ring={mono:.2f}", fontsize=11); ax.set_aspect("equal")
    Wk = head.key.weight.detach().cpu().numpy(); Yk, evk, _ = ringness(Wk)
    ax = axes.flat[5]; ax.plot(Yk[:, 0], Yk[:, 1], '-o', color="#d9622f")
    for k in range(len(Yk)): ax.annotate(str(k), Yk[k], fontsize=9)
    ax.set_aspect("equal"); ax.set_title(f"key PC1+2={evk[:2].sum():.2f}", fontsize=11)
    plt.suptitle(f"Isometry gate (no displacement), lam={args.lam_iso} margin={args.margin}", fontsize=12); plt.tight_layout()
    plt.savefig("/home/jan/projects/CIIRC/colabs/Alma/cognitiveMapsPlusPlus/factored_vis/mini14_component_pca.png", dpi=110, bbox_inches="tight")

    W0 = head.pos[0].weight.detach().cpu().numpy(); ed0 = np.linalg.norm(W0 - W0[0], axis=1)
    fig2, ax2 = plt.subplots(figsize=(6, 5)); ax2.scatter(gd, ed0, s=24, color="#0e8f8a")
    ax2.set_xlabel("cyclic graph distance"); ax2.set_ylabel("embedding distance"); ax2.set_title("agent 0 isometry (no-cheat gate)")
    plt.savefig("/home/jan/projects/CIIRC/colabs/Alma/cognitiveMapsPlusPlus/factored_vis/mini14_isometry_check.png", dpi=110, bbox_inches="tight")

    head.to(device); rj = np.random.default_rng(1)
    print("\n[VGT ladder]", flush=True)
    for m in range(N):
        probes = [np.concatenate([rj.integers(1, G, N), [m]]) for _ in range(6)]
        vg = [measure(head, p, rj, device, W=16, M=150000, L=2000)[0] for p in probes]
        vg = [x for x in vg if np.isfinite(x) and 0 < x < 40]
        print(f"  key={m} DOF={1 + m}:  VGT={np.mean(vg):.2f}" if vg else f"  key={m} DOF={1 + m}: nan", flush=True)

if __name__ == "__main__":
    main()

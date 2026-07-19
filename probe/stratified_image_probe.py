"""
Clean strata from IMAGE input.

Every stratified-embedding picture so far came from a factored-token model. This trains the
same stratified space from RAW PIXELS: agents + knobs rendered as an image, read by an
attention encoder. Goal: show the embedding organizes into clean strata of different local
dimension, from pixels alone.

Env: 2 agents on a GxG torus, each with a movement knob {all=2D,horiz=1D,vert=1D,none=0D};
local DOF = sum in {0..4}. Image = GxG grid (agent i -> pixel value i+1) + one knob-cell per
agent (value = knob state). Trained on adjacency only (contrastive + multi-scale L2 isometry,
no actions). Then: VGT local dimension vs DOF, and a PCA/UMAP strata visualization.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

NAGENTS = 2
G = 15
NPOS = G * G
NKNOB = 4
P = NPOS + NAGENTS                      # pixels: grid + one knob-cell per agent
VOCAB = max(NAGENTS + 1, NKNOB)         # values: 0 bg, 1..NAGENTS agents ; knob cells 0..3
KDOF = np.array([2, 1, 1, 0])
OUT = os.path.join(os.path.dirname(__file__), "..", "factored_vis")


def rand_states(n, device):
    return (torch.randint(0, NPOS, (n, NAGENTS), device=device),
            torch.randint(0, NKNOB, (n, NAGENTS), device=device))


def render(pos, knob):
    """(B,NAGENTS) pos + knob -> (B,P) pixel image in {0..VOCAB-1}."""
    B, dev = pos.shape[0], pos.device
    grid = torch.zeros(B, P, dtype=torch.long, device=dev)
    ar = torch.arange(B, device=dev)
    for i in range(NAGENTS):
        grid[ar, pos[:, i]] = i + 1                      # later agents overwrite on collision
    for i in range(NAGENTS):
        grid[:, NPOS + i] = knob[:, i]
    return grid


def sample_neighbor(pos, knob):
    n, dev = pos.shape[0], pos.device
    ar = torch.arange(n, device=dev)
    a = torch.randint(0, NAGENTS, (n,), device=dev)
    act = torch.randint(0, 3, (n,), device=dev)
    ka = knob[ar, a]
    row_ok = (ka == 0) | (ka == 2); col_ok = (ka == 0) | (ka == 1)
    valid = (act == 2) | ((act == 0) & row_ok) | ((act == 1) & col_ok)
    act = torch.where(valid, act, torch.full_like(act, 2))
    pa = pos[ar, a]; r, c = pa // G, pa % G
    sign = torch.randint(0, 2, (n,), device=dev) * 2 - 1
    new = torch.where(act == 0, ((r + sign) % G) * G + c,
                      torch.where(act == 1, r * G + (c + sign) % G, pa))
    nk = torch.where(act == 2, (ka + torch.randint(1, NKNOB, (n,), device=dev)) % NKNOB, ka)
    pos = pos.clone(); knob = knob.clone()
    pos[ar, a] = new; knob[ar, a] = nk
    return pos, knob


def sample_move_pair(pos, knob, K):
    n, dev = pos.shape[0], pos.device
    dsq = torch.zeros(n, device=dev); tgt = pos.clone()
    for i in range(NAGENTS):
        ki = knob[:, i]; r, c = pos[:, i] // G, pos[:, i] % G
        row_ok = ((ki == 0) | (ki == 2)).long(); col_ok = ((ki == 0) | (ki == 1)).long()
        drow = torch.randint(-K, K + 1, (n,), device=dev) * row_ok
        dcol = torch.randint(-K, K + 1, (n,), device=dev) * col_ok
        tgt[:, i] = ((r + drow) % G) * G + ((c + dcol) % G)
        dsq = dsq + drow.float() ** 2 + dcol.float() ** 2
    return tgt, dsq.sqrt()


def knob_change(pos, knob):
    n, dev = knob.shape[0], knob.device
    ar = torch.arange(n, device=dev); a = torch.randint(0, NAGENTS, (n,), device=dev)
    knob = knob.clone()
    knob[ar, a] = (knob[ar, a] + torch.randint(1, NKNOB, (n,), device=dev)) % NKNOB
    return pos, knob


class ImageEnc(nn.Module):
    def __init__(self, d_model=48, D=32, n_slots=6, n_heads=4):
        super().__init__()
        self.val_emb = nn.Embedding(VOCAB, d_model)
        self.pos_emb = nn.Embedding(P, d_model)
        self.slots = nn.Parameter(torch.randn(n_slots, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(n_slots * d_model, 128), nn.GELU(), nn.Linear(128, D))

    def embed(self, pos, knob):
        grid = render(pos, knob)                          # (B,P)
        B = grid.shape[0]
        pix = self.val_emb(grid) + self.pos_emb(torch.arange(P, device=grid.device)).unsqueeze(0)
        z, _ = self.attn(self.slots.unsqueeze(0).expand(B, -1, -1), pix, pix)
        return self.proj(z.reshape(B, -1))


def train(model, steps, batch, lr, device, K=2, rep_offset=10.0, eval_every=1000):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for s in range(steps):
        pos, knob = rand_states(batch, device)
        eu = model.embed(pos, knob)
        tgt, dist = sample_move_pair(pos, knob, K)
        loss_iso = (torch.norm(eu - model.embed(tgt, knob), dim=-1) - dist).square().mean()
        _, nk = knob_change(pos, knob)
        loss_knob = (torch.norm(eu - model.embed(pos, nk), dim=-1) - 1.0).square().mean()
        rp, rk = rand_states(batch, device)
        loss_rep = F.softplus(rep_offset - torch.norm(eu - model.embed(rp, rk), dim=-1)).mean()
        loss = loss_iso + loss_knob + loss_rep
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % eval_every == 0 or s == 0:
            print(f"step {s+1}/{steps} loss {loss.item():.3f} (iso {loss_iso.item():.3f} "
                  f"knob {loss_knob.item():.3f} rep {loss_rep.item():.3f})")
    return model


# ---------- dimension via local-ball VGT ----------
def vgt_slope(dist, window=3, rmax_frac=0.55, min_count=6, num_radii=30):
    d = np.sort(dist[dist > 1e-9])
    if d.size < 12:
        return np.nan
    rmax = d[int(len(d) * rmax_frac)]
    if rmax <= d[0]:
        return np.nan
    radii = np.logspace(np.log10(d[0]), np.log10(rmax), num_radii)
    counts = np.searchsorted(d, radii, side="left").astype(float)
    valid = counts >= min_count
    if valid.sum() < window + 1:
        return np.nan
    lr, lc = np.log(radii[valid]), np.log(counts[valid])
    sl = [np.polyfit(lr[i:i + window], lc[i:i + window], 1)[0] for i in range(len(lr) - window)]
    return float(np.median(sl)) if sl else np.nan


def local_ball(pos, knob, R, N, rng):
    cols = []
    for i in range(NAGENTS):
        k = knob[i]
        dr = rng.integers(-R, R + 1, N) if k in (0, 2) else np.zeros(N, int)
        dc = rng.integers(-R, R + 1, N) if k in (0, 1) else np.zeros(N, int)
        cols += [dr, dc]
    off = np.unique(np.stack(cols, 1), axis=0)
    npos = np.zeros((off.shape[0], NAGENTS), np.int64)
    for i in range(NAGENTS):
        r, c = pos[i] // G, pos[i] % G
        npos[:, i] = ((r + off[:, 2 * i]) % G) * G + ((c + off[:, 2 * i + 1]) % G)
    return npos


@torch.no_grad()
def measure_vgt(model, device, R, N, per_dof, seed):
    rng = np.random.default_rng(seed); rows = []
    for dof in range(0, 2 * NAGENTS + 1):
        for _ in range(per_dof):
            knob = None
            for _ in range(2000):
                cand = rng.integers(0, NKNOB, NAGENTS)
                if int(KDOF[cand].sum()) == dof:
                    knob = cand; break
            if knob is None:
                continue
            pos = rng.integers(0, NPOS, NAGENTS)
            ball = local_ball(pos, knob, R, N, rng); M = ball.shape[0]
            E = model.embed(torch.tensor(ball, device=device),
                            torch.tensor(np.tile(knob, (M, 1)), device=device)).cpu().numpy()
            q = model.embed(torch.tensor(pos[None], device=device),
                            torch.tensor(knob[None], device=device)).cpu().numpy()[0]
            rows.append((dof, vgt_slope(np.linalg.norm(E - q, axis=1))))
    return np.array(rows, float)


# ---------- strata visualization ----------
@torch.no_grad()
def visualize(model, device, per_dof, seed):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    import umap
    rng = np.random.default_rng(seed)
    P_, K_, D_ = [], [], []
    for dof in range(0, 2 * NAGENTS + 1):
        got = 0
        while got < per_dof:
            cand = rng.integers(0, NKNOB, NAGENTS)
            if int(KDOF[cand].sum()) != dof:
                continue
            pos = rng.integers(0, NPOS, NAGENTS)
            P_.append(pos); K_.append(cand); D_.append(dof); got += 1
    pos = torch.tensor(np.array(P_), device=device); knob = torch.tensor(np.array(K_), device=device)
    dof = np.array(D_)
    E = []
    for i in range(0, len(dof), 4096):
        E.append(model.embed(pos[i:i + 4096], knob[i:i + 4096]).cpu().numpy())
    E = np.concatenate(E)
    p2 = PCA(2).fit_transform(E); p3 = PCA(3).fit_transform(E)
    u2 = umap.UMAP(n_neighbors=20, min_dist=0.12, random_state=1).fit_transform(E)
    fig = plt.figure(figsize=(15, 4.4))
    ax = fig.add_subplot(1, 3, 1)
    s = ax.scatter(p2[:, 0], p2[:, 1], c=dof, cmap="viridis", s=9); ax.set_title("PCA | true dimension (DOF)")
    plt.colorbar(s, ax=ax, fraction=.046); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(1, 3, 2, projection="3d")
    ax.scatter(p3[:, 0], p3[:, 1], p3[:, 2], c=dof, cmap="viridis", s=8)
    ax.set_title("PCA 3D | DOF"); ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax = fig.add_subplot(1, 3, 3)
    s = ax.scatter(u2[:, 0], u2[:, 1], c=dof, cmap="viridis", s=9); ax.set_title("UMAP | DOF")
    plt.colorbar(s, ax=ax, fraction=.046); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Stratified embedding from IMAGE input (2 agents, dims 0-4)", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT, "stratified_image.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.relpath(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--R", type=int, default=6)
    ap.add_argument("--N", type=int, default=6000)
    ap.add_argument("--per_dof", type=int, default=30)
    ap.add_argument("--vis_per_dof", type=int, default=700)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(OUT, exist_ok=True)
    print(f"device={device} IMAGE input, {NAGENTS} agents G={G} P={P} vocab={VOCAB} (DOF 0..4)")

    model = ImageEnc().to(device)
    train(model, args.steps, args.batch, args.lr, device, K=args.K, eval_every=args.eval_every)

    res = measure_vgt(model, device, args.R, args.N, args.per_dof, args.seed)
    dof, vgt = res[:, 0], res[:, 1]
    m = ~np.isnan(vgt)
    corr = spearmanr(vgt[m], dof[m]).statistic
    ladder = {int(d): round(float(vgt[m & (dof == d)].mean()), 2) for d in sorted(set(dof.astype(int)))
              if (m & (dof == d)).sum() >= 2}
    print("\n================ IMAGE-INPUT STRATIFICATION ================")
    print(f"VGT local dimension vs true DOF:  corr={corr:+.3f}   ladder={ladder}")
    print("============================================================")
    visualize(model, device, args.vis_per_dof, args.seed)


if __name__ == "__main__":
    main()

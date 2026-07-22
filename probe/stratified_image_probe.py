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


def train(model, steps, batch, lr, device, K=2, knob_dist=6.0, rep_offset=15.0, eval_every=1000):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for s in range(steps):
        pos, knob = rand_states(batch, device)
        eu = model.embed(pos, knob)
        tgt, dist = sample_move_pair(pos, knob, K)
        loss_iso = (torch.norm(eu - model.embed(tgt, knob), dim=-1) - dist).square().mean()
        _, nk = knob_change(pos, knob)
        # knob change anchored to knob_dist (default 1 = unit cost, same as a position move)
        loss_knob = (torch.norm(eu - model.embed(pos, nk), dim=-1) - knob_dist).square().mean()
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
            E = []
            for i in range(0, M, 8192):                    # chunk to avoid OOM on large balls
                b = ball[i:i + 8192]
                E.append(model.embed(torch.tensor(b, device=device),
                                     torch.tensor(np.tile(knob, (len(b), 1)), device=device)).cpu().numpy())
            E = np.concatenate(E)
            q = model.embed(torch.tensor(pos[None], device=device),
                            torch.tensor(knob[None], device=device)).cpu().numpy()[0]
            rows.append((dof, vgt_slope(np.linalg.norm(E - q, axis=1))))
    return np.array(rows, float)


# ---------- per-point local dimension (participation ratio of move-neighbors) ----------
def _move_nbr_offsets(k):
    dirs = []
    if k in (0, 2): dirs += [(-1, 0), (1, 0)]     # vertical (row) moves
    if k in (0, 1): dirs += [(0, -1), (0, 1)]     # horizontal (col) moves
    return dirs


@torch.no_grad()
def per_point_localdim(model, device, POS, KNOB):
    """for each state, participation-ratio dimension of its move-neighbors' embedding diffs."""
    out = np.zeros(len(POS))
    for j in range(len(POS)):
        pos, knob = POS[j], KNOB[j]
        nb_pos = [pos]
        for i in range(NAGENTS):
            r, c = pos[i] // G, pos[i] % G
            for dr, dc in _move_nbr_offsets(knob[i]):
                p2 = pos.copy(); p2[i] = ((r + dr) % G) * G + ((c + dc) % G); nb_pos.append(p2)
        if len(nb_pos) == 1:
            out[j] = 0.0; continue
        nb = np.stack(nb_pos)
        E = model.embed(torch.tensor(nb, device=device),
                        torch.tensor(np.tile(knob, (len(nb), 1)), device=device)).cpu().numpy()
        diffs = E[1:] - E[0]
        ev = np.clip(np.linalg.eigvalsh(diffs.T @ diffs), 0, None)
        s1 = ev.sum()
        out[j] = (s1 ** 2) / (ev ** 2).sum() if s1 > 1e-9 else 0.0
    return out


# ---------- strata visualization ----------
@torch.no_grad()
def visualize(model, device, n_vis, seed, tag=""):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import umap
    rng = np.random.default_rng(seed)
    MAXD = 2 * NAGENTS
    # sample states stratified by DOF (so every level 0..MAXD is covered)
    per = max(1, n_vis // (MAXD + 1))
    P_, K_, D_ = [], [], []
    for dof_t in range(MAXD + 1):
        got = 0
        while got < per:
            cand = rng.integers(0, NKNOB, NAGENTS)
            if int(KDOF[cand].sum()) != dof_t:
                continue
            P_.append(rng.integers(0, NPOS, NAGENTS)); K_.append(cand); D_.append(dof_t); got += 1
    Pa, Ka = np.array(P_), np.array(K_); dof = np.array(D_)
    pos = torch.tensor(Pa, device=device); knob = torch.tensor(Ka, device=device)
    E = []
    for i in range(0, len(dof), 4096):
        E.append(model.embed(pos[i:i + 4096], knob[i:i + 4096]).cpu().numpy())
    E = np.concatenate(E)
    ld = per_point_localdim(model, device, Pa, Ka)                       # measured local dim per point
    u2 = umap.UMAP(n_neighbors=25, min_dist=0.1, random_state=1).fit_transform(E)

    fig = plt.figure(figsize=(15, 4.6))
    ax = fig.add_subplot(1, 3, 1)
    s = ax.scatter(u2[:, 0], u2[:, 1], c=dof, cmap="viridis", s=7); ax.set_title(f"UMAP · TRUE dimension (DOF 0–{MAXD})")
    plt.colorbar(s, ax=ax, fraction=.046); ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(1, 3, 2)
    s = ax.scatter(u2[:, 0], u2[:, 1], c=ld, cmap="plasma", s=7)
    ax.set_title("UMAP · MEASURED local dimension"); plt.colorbar(s, ax=ax, fraction=.046)
    ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(1, 3, 3)
    ax.scatter(dof + np.random.default_rng(0).normal(0, .06, len(dof)), ld, c=dof, cmap="viridis", s=6, alpha=.5)
    ax.set_title("measured local dim vs true DOF"); ax.set_xlabel("true DOF"); ax.set_ylabel("measured")
    ax.plot([0, MAXD], [0, MAXD], color="0.6", ls="--", lw=1)
    fig.suptitle(f"Stratified embedding from IMAGE input (unit-cost) — {NAGENTS} agents · dims 0–{MAXD}", fontsize=12)
    fig.tight_layout()
    path = os.path.join(OUT, f"stratified_image{('_' + tag) if tag else ''}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight"); plt.close(fig)
    print("wrote", os.path.relpath(path))


def main():
    global NAGENTS, G, NPOS, P, VOCAB
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--K", type=int, default=2)
    ap.add_argument("--knob_dist", type=float, default=1.0)   # knob change = unit cost, like a move
    ap.add_argument("--R", type=int, default=6)
    ap.add_argument("--N", type=int, default=6000)
    ap.add_argument("--per_dof", type=int, default=30)
    ap.add_argument("--n_vis", type=int, default=3000)
    ap.add_argument("--nagents", type=int, default=NAGENTS)
    ap.add_argument("--G", type=int, default=G)
    ap.add_argument("--eval_every", type=int, default=2000)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--D", type=int, default=48)
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    NAGENTS, G = args.nagents, args.G
    NPOS = G * G; P = NPOS + NAGENTS; VOCAB = max(NAGENTS + 1, NKNOB)

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(OUT, exist_ok=True)
    print(f"device={device} IMAGE input tag={args.tag} {NAGENTS} agents G={G} P={P} vocab={VOCAB} "
          f"knob_dist={args.knob_dist} d_model={args.d_model} D={args.D} slots={args.slots}")

    model = ImageEnc(d_model=args.d_model, D=args.D, n_slots=args.slots).to(device)
    train(model, args.steps, args.batch, args.lr, device, K=args.K, knob_dist=args.knob_dist,
          rep_offset=args.knob_dist + 9.0, eval_every=args.eval_every)

    res = measure_vgt(model, device, args.R, args.N, args.per_dof, args.seed)
    dof, vgt = res[:, 0], res[:, 1]
    m = ~np.isnan(vgt)
    corr = spearmanr(vgt[m], dof[m]).statistic
    ladder = {int(d): round(float(vgt[m & (dof == d)].mean()), 2) for d in sorted(set(dof.astype(int)))
              if (m & (dof == d)).sum() >= 2}
    print(f"\n[{args.tag}] VGT corr={corr:+.3f}  ladder={ladder}", flush=True)
    visualize(model, device, args.n_vis, args.seed, tag=args.tag)


if __name__ == "__main__":
    main()

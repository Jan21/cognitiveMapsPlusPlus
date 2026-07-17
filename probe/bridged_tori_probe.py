"""
Probe: factored latent + attention distance head on a two-torus bridged graph.

Tests whether a learned cross-attention distance D(A,B), trained ONLY on one-step
transitions (local supervision), recovers the true graph geodesic on a graph where
crossing between two 15x15 tori is only possible through a single bridge node.

Compares against a plain-embedding + L-p distance baseline.

Standalone: no Hydra, no path-JSON. Builds graph + transitions in memory.
Runs on CPU (tiny) or CUDA if available.

Kill criterion (set before running):
  PASS if BOTH
    1. Spearman(D, geodesic) for the factored+attention model beats the baseline
       by >= SPEARMAN_MARGIN.
    2. Detour signature: D((p,x),(p,y)) rank-correlates with torus_geo(p, 150)
       (Spearman >= DETOUR_MIN), i.e. same-position-different-torus distance grows
       with distance-to-bridge instead of being constant.
"""

import argparse
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

# ---- graph constants ----
SIDE = 15                 # 15x15 torus
N_POS = SIDE * SIDE       # 225 positions per torus
N_STATES = 2 * N_POS      # 450 states
BRIDGE_POS = 150          # only crossing point
N_ACTIONS = 6             # up, down, left, right, next, prev
UP, DOWN, LEFT, RIGHT, NEXT, PREV = range(N_ACTIONS)

# ---- kill-criterion thresholds ----
SPEARMAN_MARGIN = 0.05    # factored must beat baseline Spearman by this
DETOUR_MIN = 0.5          # min Spearman for detour signature


# ---------- graph + transitions ----------
def gid(pos, torus):
    """global state id in [0, 450)."""
    return torus * N_POS + pos


def move(pos, action):
    """grid move on a single 15x15 wraparound torus. returns new pos."""
    r, c = divmod(pos, SIDE)
    if action == UP:
        r = (r - 1) % SIDE
    elif action == DOWN:
        r = (r + 1) % SIDE
    elif action == LEFT:
        c = (c - 1) % SIDE
    elif action == RIGHT:
        c = (c + 1) % SIDE
    return r * SIDE + c


def step(pos, torus, action):
    """true environment dynamics. returns (next_pos, next_torus)."""
    if action in (UP, DOWN, LEFT, RIGHT):
        return move(pos, action), torus
    if action == NEXT:
        if pos == BRIDGE_POS and torus == 0:
            return pos, 1
        return pos, torus            # self-loop everywhere else
    if action == PREV:
        if pos == BRIDGE_POS and torus == 1:
            return pos, 0
        return pos, torus            # self-loop everywhere else
    raise ValueError(action)


def build_graph():
    """450-node bridged-tori graph (undirected) for ground-truth geodesics."""
    G = nx.Graph()
    G.add_nodes_from(range(N_STATES))
    for torus in (0, 1):
        for pos in range(N_POS):
            u = gid(pos, torus)
            for a in (UP, DOWN, LEFT, RIGHT):
                v = gid(move(pos, a), torus)
                G.add_edge(u, v)
    # bridge
    G.add_edge(gid(BRIDGE_POS, 0), gid(BRIDGE_POS, 1))
    return G


def build_transitions():
    """all (state, action, next_state) triples, incl. self-loops. shape (N,3) long."""
    rows = []
    for torus in (0, 1):
        for pos in range(N_POS):
            for a in range(N_ACTIONS):
                npos, nt = step(pos, torus, a)
                rows.append((gid(pos, torus), a, gid(npos, nt)))
    return torch.tensor(rows, dtype=torch.long)


def all_pairs_geodesic(G):
    """dense (450,450) geodesic matrix via BFS."""
    D = np.zeros((N_STATES, N_STATES), dtype=np.float32)
    for src, lengths in nx.all_pairs_shortest_path_length(G):
        for dst, d in lengths.items():
            D[src, dst] = d
    return D


def torus_geo_to_bridge(G):
    """geodesic within a single torus from each pos to BRIDGE_POS (torus 0)."""
    # restrict to torus-0 nodes (no bridge use since we stay on torus 0)
    lengths = nx.shortest_path_length(G, source=gid(BRIDGE_POS, 0))
    return np.array([lengths[gid(p, 0)] for p in range(N_POS)], dtype=np.float32)


# ---------- factored + attention model ----------
class FactoredAttentionModel(nn.Module):
    def __init__(self, d_model=32, n_layers=2, n_heads=4, dyn_hidden=128):
        super().__init__()
        self.d_model = d_model
        # factored embedding: shared position table across BOTH tori + torus id table
        self.E_pos = nn.Embedding(N_POS, d_model)
        self.E_id = nn.Embedding(2, d_model)
        # distance-head structural embeddings
        self.type_emb = nn.Embedding(2, d_model)   # 0=pos token, 1=id token
        self.role_emb = nn.Embedding(2, d_model)   # 0=state u, 1=state v
        self.dist_token = nn.Parameter(torch.randn(d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.dist_out = nn.Linear(d_model, 1)
        # dynamics net: (t_pos, t_id, action) -> (t_pos', t_id')  residual
        self.act_emb = nn.Embedding(N_ACTIONS, d_model)
        self.dyn = nn.Sequential(
            nn.Linear(3 * d_model, dyn_hidden), nn.GELU(),
            nn.Linear(dyn_hidden, dyn_hidden), nn.GELU(),
            nn.Linear(dyn_hidden, 2 * d_model),
        )

    def state_tokens(self, pos, torus):
        """(B,) pos and torus -> (B, 2, d) tokens [t_pos, t_id]."""
        return torch.stack([self.E_pos(pos), self.E_id(torus)], dim=1)

    def _raw_distance(self, tok_u, tok_v):
        """directional head over tokens. tok_*: (B,2,d) -> (B,) >=0."""
        B = tok_u.shape[0]
        typ = self.type_emb(torch.tensor([0, 1], device=tok_u.device))   # (2,d)
        ru = self.role_emb(torch.tensor(0, device=tok_u.device))
        rv = self.role_emb(torch.tensor(1, device=tok_u.device))
        u = tok_u + typ + ru                                             # (B,2,d)
        v = tok_v + typ + rv
        dist = self.dist_token.expand(B, 1, self.d_model)                # (B,1,d)
        seq = torch.cat([dist, u, v], dim=1)                             # (B,5,d)
        out = self.encoder(seq)
        return F.softplus(self.dist_out(out[:, 0]).squeeze(-1))

    def distance(self, tok_u, tok_v):
        """symmetrized distance."""
        return 0.5 * (self._raw_distance(tok_u, tok_v) + self._raw_distance(tok_v, tok_u))

    def dynamics(self, tok, action):
        """(B,2,d) tokens + (B,) action -> predicted next tokens (B,2,d)."""
        t_pos, t_id = tok[:, 0], tok[:, 1]
        inp = torch.cat([t_pos, t_id, self.act_emb(action)], dim=-1)
        delta = self.dyn(inp).view(-1, 2, self.d_model)
        return tok + delta

    def pos_torus(self, gids):
        """global ids -> (pos, torus)."""
        return gids % N_POS, gids // N_POS


# ---------- plain baseline ----------
class PlainEmbeddingModel(nn.Module):
    """one vector per state, fixed L-p distance. no factoring, no attention, no actions."""
    def __init__(self, latent=16, p=1.5):
        super().__init__()
        self.emb = nn.Embedding(N_STATES, latent)
        self.p = p

    def distance_ids(self, a_ids, b_ids):
        za, zb = self.emb(a_ids), self.emb(b_ids)
        return torch.norm(za - zb, p=self.p, dim=-1)


# ---------- training ----------
def train_factored(model, trans, steps, batch, lr, device,
                   rep_weight=1.0, rep_offset=10.0, dyn_weight=1.0, anchor_weight=1.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    A, ACT, B = trans[:, 0].to(device), trans[:, 1].to(device), trans[:, 2].to(device)
    is_move = (A != B)                       # non-self-loop = real cost-1 edge
    n = trans.shape[0]
    model.train()
    for s in range(steps):
        idx = torch.randint(0, n, (batch,), device=device)
        a_id, act, b_id = A[idx], ACT[idx], B[idx]
        ap, at = model.pos_torus(a_id)
        bp, bt = model.pos_torus(b_id)
        tok_a = model.state_tokens(ap, at)
        tok_b = model.state_tokens(bp, bt)

        # dynamics consistency: predicted next ~ actual next  (D -> 0)
        pred = model.dynamics(tok_a, act)
        d_dyn = model.distance(pred, tok_b)
        loss_dyn = d_dyn.square().mean()

        # scale anchor: real one-step neighbors have distance ~ 1
        mv = is_move[idx]
        if mv.any():
            d_ab = model.distance(tok_a[mv], tok_b[mv])
            loss_anchor = (d_ab - 1.0).square().mean()
        else:
            loss_anchor = torch.zeros((), device=device)

        # repulsion: random pairs pushed apart (anti-collapse)
        r_id = torch.randint(0, N_STATES, (batch,), device=device)
        rp, rt = model.pos_torus(r_id)
        tok_r = model.state_tokens(rp, rt)
        d_rand = model.distance(tok_a, tok_r)
        loss_rep = F.softplus(rep_offset - d_rand).mean()

        loss = dyn_weight * loss_dyn + anchor_weight * loss_anchor + rep_weight * loss_rep
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (s + 1) % max(1, steps // 10) == 0:
            print(f"[factored] step {s+1}/{steps} loss {loss.item():.4f} "
                  f"(dyn {loss_dyn.item():.4f} anchor {loss_anchor.item():.4f} rep {loss_rep.item():.4f})")
    return model


def train_baseline(model, trans, steps, batch, lr, device,
                   rep_weight=1.0, rep_offset=10.0):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    A, B = trans[:, 0].to(device), trans[:, 2].to(device)
    move_mask = (A != B)
    Am, Bm = A[move_mask], B[move_mask]          # real edges only (no actions used)
    nm = Am.shape[0]
    model.train()
    for s in range(steps):
        idx = torch.randint(0, nm, (batch,), device=device)
        a_id, b_id = Am[idx], Bm[idx]
        d_ab = model.distance_ids(a_id, b_id)
        loss_anchor = (d_ab - 1.0).square().mean()
        r_id = torch.randint(0, N_STATES, (batch,), device=device)
        d_rand = model.distance_ids(a_id, r_id)
        loss_rep = F.softplus(rep_offset - d_rand).mean()
        loss = loss_anchor + rep_weight * loss_rep
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (s + 1) % max(1, steps // 10) == 0:
            print(f"[baseline] step {s+1}/{steps} loss {loss.item():.4f} "
                  f"(anchor {loss_anchor.item():.4f} rep {loss_rep.item():.4f})")
    return model


# ---------- evaluation ----------
@torch.no_grad()
def factored_distance_ids(model, a_ids, b_ids, device, chunk=4096):
    model.eval()
    out = []
    for i in range(0, len(a_ids), chunk):
        a = a_ids[i:i + chunk].to(device)
        b = b_ids[i:i + chunk].to(device)
        ap, at = model.pos_torus(a)
        bp, bt = model.pos_torus(b)
        d = model.distance(model.state_tokens(ap, at), model.state_tokens(bp, bt))
        out.append(d.cpu())
    return torch.cat(out).numpy()


def evaluate(model_f, model_b, geo, tg_bridge, device, n_pairs=4000, seed=0):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, N_STATES, size=n_pairs)
    b = rng.integers(0, N_STATES, size=n_pairs)
    mask = a != b
    a, b = a[mask], b[mask]
    g = geo[a, b]
    a_t = torch.tensor(a, dtype=torch.long)
    b_t = torch.tensor(b, dtype=torch.long)

    d_f = factored_distance_ids(model_f, a_t, b_t, device)
    with torch.no_grad():
        d_b = model_b.distance_ids(a_t.to(device), b_t.to(device)).cpu().numpy()

    sp_f = spearmanr(d_f, g).statistic
    sp_b = spearmanr(d_b, g).statistic

    # detour signature: D((p,x),(p,y)) vs torus_geo(p, bridge)
    px = torch.tensor([gid(p, 0) for p in range(N_POS)], dtype=torch.long)
    py = torch.tensor([gid(p, 1) for p in range(N_POS)], dtype=torch.long)
    d_cross_f = factored_distance_ids(model_f, px, py, device)
    with torch.no_grad():
        d_cross_b = model_b.distance_ids(px.to(device), py.to(device)).cpu().numpy()
    detour_f = spearmanr(d_cross_f, tg_bridge).statistic
    detour_b = spearmanr(d_cross_b, tg_bridge).statistic

    print("\n================ RESULTS ================")
    print(f"Spearman(D, geodesic)     factored={sp_f:.3f}   baseline={sp_b:.3f}   "
          f"margin={sp_f - sp_b:+.3f} (need >= {SPEARMAN_MARGIN})")
    print(f"Detour signature Spearman factored={detour_f:.3f}   baseline={detour_b:.3f}   "
          f"(factored need >= {DETOUR_MIN})")

    pass1 = (sp_f - sp_b) >= SPEARMAN_MARGIN
    pass2 = detour_f >= DETOUR_MIN
    verdict = "PASS" if (pass1 and pass2) else "FAIL"
    print(f"kill-criterion 1 (beat baseline): {'ok' if pass1 else 'NO'}")
    print(f"kill-criterion 2 (detour signal): {'ok' if pass2 else 'NO'}")
    print(f"VERDICT: {verdict}")
    print("=========================================")
    return dict(sp_f=sp_f, sp_b=sp_b, detour_f=detour_f, detour_b=detour_b, verdict=verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    print(f"device={device} steps={args.steps} batch={args.batch} lr={args.lr} seed={args.seed}")

    print("building graph + transitions ...")
    G = build_graph()
    print(f"nodes={G.number_of_nodes()} edges={G.number_of_edges()} connected={nx.is_connected(G)}")
    trans = build_transitions()
    print(f"transitions={trans.shape[0]}")
    geo = all_pairs_geodesic(G)
    tg_bridge = torus_geo_to_bridge(G)
    print(f"max geodesic={geo.max():.0f}  max torus->bridge={tg_bridge.max():.0f}")

    print("\ntraining factored+attention model ...")
    model_f = FactoredAttentionModel().to(device)
    train_factored(model_f, trans, args.steps, args.batch, args.lr, device)

    print("\ntraining plain baseline ...")
    model_b = PlainEmbeddingModel().to(device)
    train_baseline(model_b, trans, args.steps, args.batch, args.lr, device)

    evaluate(model_f, model_b, geo, tg_bridge, device, seed=args.seed)


if __name__ == "__main__":
    main()

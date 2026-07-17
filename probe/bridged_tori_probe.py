"""
Probe: factored latent + attention distance head on a two-torus bridged graph.

Tests whether a learned attention distance D(A,B), trained ONLY on one-step
transitions (local supervision), recovers the true graph geodesic on a graph where
crossing between two 15x15 tori is only possible through a single bridge node.

Compares against a plain-embedding + L-p distance baseline.

Standalone: no Hydra, no path-JSON. Builds graph + transitions in memory.
Runs on CPU (tiny) or CUDA if available.

Distance-head variants (--head):
  cross_scalar : B3 as designed. Cross-attention over the 4 factor tokens of the
                 two states -> a free scalar distance. No metric prior.
  cross_reg    : cross_scalar + triangle-inequality and identity regularizers
                 (the metric "knobs" reserved in the design).
  self_norm    : self-attention over each state's 2 factor tokens -> a state
                 embedding; distance = ||e_u - e_v||_p. Attention still computes the
                 similarity, but the readout is a proper metric (triangle inequality
                 + identity by construction).

Kill criterion (set before running):
  PASS if BOTH
    1. Spearman(D, geodesic) beats the baseline by >= SPEARMAN_MARGIN.
    2. Detour signature: D((p,x),(p,y)) rank-correlates with torus_geo(p, 150)
       (Spearman >= DETOUR_MIN).
"""

import argparse
import json
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

# ---- graph constants ----
SIDE = 15
N_POS = SIDE * SIDE       # 225 positions per torus
N_STATES = 2 * N_POS      # 450 states
BRIDGE_POS = 150
N_ACTIONS = 6
UP, DOWN, LEFT, RIGHT, NEXT, PREV = range(N_ACTIONS)

# ---- kill-criterion thresholds ----
SPEARMAN_MARGIN = 0.05
DETOUR_MIN = 0.5


# ---------- graph + transitions ----------
def gid(pos, torus):
    return torus * N_POS + pos


def move(pos, action):
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
    if action in (UP, DOWN, LEFT, RIGHT):
        return move(pos, action), torus
    if action == NEXT:
        if pos == BRIDGE_POS and torus == 0:
            return pos, 1
        return pos, torus
    if action == PREV:
        if pos == BRIDGE_POS and torus == 1:
            return pos, 0
        return pos, torus
    raise ValueError(action)


def build_graph():
    G = nx.Graph()
    G.add_nodes_from(range(N_STATES))
    for torus in (0, 1):
        for pos in range(N_POS):
            u = gid(pos, torus)
            for a in (UP, DOWN, LEFT, RIGHT):
                G.add_edge(u, gid(move(pos, a), torus))
    G.add_edge(gid(BRIDGE_POS, 0), gid(BRIDGE_POS, 1))
    return G


def build_transitions():
    rows = []
    for torus in (0, 1):
        for pos in range(N_POS):
            for a in range(N_ACTIONS):
                npos, nt = step(pos, torus, a)
                rows.append((gid(pos, torus), a, gid(npos, nt)))
    return torch.tensor(rows, dtype=torch.long)


def all_pairs_geodesic(G):
    D = np.zeros((N_STATES, N_STATES), dtype=np.float32)
    for src, lengths in nx.all_pairs_shortest_path_length(G):
        for dst, d in lengths.items():
            D[src, dst] = d
    return D


def torus_geo_to_bridge(G):
    lengths = nx.shortest_path_length(G, source=gid(BRIDGE_POS, 0))
    return np.array([lengths[gid(p, 0)] for p in range(N_POS)], dtype=np.float32)


# ---------- factored + attention model ----------
class FactoredAttentionModel(nn.Module):
    def __init__(self, head="self_norm", d_model=32, n_layers=2, n_heads=4,
                 dyn_hidden=128, emb_dim=16, p=1.5):
        super().__init__()
        self.head = head
        self.d_model = d_model
        self.p = p
        self.E_pos = nn.Embedding(N_POS, d_model)      # shared across BOTH tori
        self.E_id = nn.Embedding(2, d_model)
        self.type_emb = nn.Embedding(2, d_model)       # 0=pos token, 1=id token
        self.role_emb = nn.Embedding(2, d_model)       # 0=state u, 1=state v (cross heads)
        self.query_token = nn.Parameter(torch.randn(d_model) * 0.02)  # DIST / POOL token
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        if head in ("cross_scalar", "cross_reg"):
            self.dist_out = nn.Linear(d_model, 1)
        else:  # self_norm
            self.proj = nn.Linear(d_model, emb_dim)
        # dynamics net: (t_pos, t_id, action) -> (t_pos', t_id')  residual
        self.act_emb = nn.Embedding(N_ACTIONS, d_model)
        self.dyn = nn.Sequential(
            nn.Linear(3 * d_model, dyn_hidden), nn.GELU(),
            nn.Linear(dyn_hidden, dyn_hidden), nn.GELU(),
            nn.Linear(dyn_hidden, 2 * d_model),
        )

    def state_tokens(self, pos, torus):
        return torch.stack([self.E_pos(pos), self.E_id(torus)], dim=1)  # (B,2,d)

    # --- cross-attention scalar head (B3) ---
    def _raw_cross(self, tok_u, tok_v):
        B = tok_u.shape[0]
        dev = tok_u.device
        typ = self.type_emb(torch.tensor([0, 1], device=dev))
        ru = self.role_emb(torch.tensor(0, device=dev))
        rv = self.role_emb(torch.tensor(1, device=dev))
        u = tok_u + typ + ru
        v = tok_v + typ + rv
        q = self.query_token.expand(B, 1, self.d_model)
        seq = torch.cat([q, u, v], dim=1)                      # (B,5,d)
        out = self.encoder(seq)
        return F.softplus(self.dist_out(out[:, 0]).squeeze(-1))

    # --- self-attention embedding head (metric readout) ---
    def embed(self, tok):
        B = tok.shape[0]
        dev = tok.device
        typ = self.type_emb(torch.tensor([0, 1], device=dev))
        x = tok + typ
        q = self.query_token.expand(B, 1, self.d_model)
        seq = torch.cat([q, x], dim=1)                         # (B,3,d)
        out = self.encoder(seq)
        return self.proj(out[:, 0])                            # (B,emb_dim)

    def distance(self, tok_u, tok_v):
        if self.head in ("cross_scalar", "cross_reg"):
            return 0.5 * (self._raw_cross(tok_u, tok_v) + self._raw_cross(tok_v, tok_u))
        return torch.norm(self.embed(tok_u) - self.embed(tok_v), p=self.p, dim=-1)

    def dynamics(self, tok, action):
        t_pos, t_id = tok[:, 0], tok[:, 1]
        inp = torch.cat([t_pos, t_id, self.act_emb(action)], dim=-1)
        return tok + self.dyn(inp).view(-1, 2, self.d_model)

    def pos_torus(self, gids):
        return gids % N_POS, gids // N_POS


class PlainEmbeddingModel(nn.Module):
    """one vector per state, fixed L-p distance. no factoring/attention/actions."""
    def __init__(self, latent=16, p=1.5):
        super().__init__()
        self.emb = nn.Embedding(N_STATES, latent)
        self.p = p

    def distance_ids(self, a_ids, b_ids):
        return torch.norm(self.emb(a_ids) - self.emb(b_ids), p=self.p, dim=-1)


# ---------- eval helpers ----------
def _sample_pairs(n_pairs, seed):
    rng = np.random.default_rng(seed)
    a = rng.integers(0, N_STATES, size=n_pairs)
    b = rng.integers(0, N_STATES, size=n_pairs)
    m = a != b
    return a[m], b[m]


@torch.no_grad()
def factored_distance_ids(model, a_ids, b_ids, device, chunk=8192):
    model.eval()
    out = []
    for i in range(0, len(a_ids), chunk):
        a = torch.as_tensor(a_ids[i:i + chunk], dtype=torch.long, device=device)
        b = torch.as_tensor(b_ids[i:i + chunk], dtype=torch.long, device=device)
        ap, at = model.pos_torus(a)
        bp, bt = model.pos_torus(b)
        out.append(model.distance(model.state_tokens(ap, at), model.state_tokens(bp, bt)).cpu())
    model.train()
    return torch.cat(out).numpy()


def spearman_geo(model, geo, a, b, device):
    d = factored_distance_ids(model, a, b, device)
    return spearmanr(d, geo[a, b]).statistic


def detour_signature(model, tg_bridge, device):
    px = np.array([gid(p, 0) for p in range(N_POS)])
    py = np.array([gid(p, 1) for p in range(N_POS)])
    d = factored_distance_ids(model, px, py, device)
    return spearmanr(d, tg_bridge).statistic, d


# ---------- training ----------
def train_factored(model, trans, geo, tg_bridge, steps, batch, lr, device,
                   rep_weight=1.0, rep_offset=10.0, dyn_weight=1.0, anchor_weight=1.0,
                   tri_weight=1.0, id_weight=1.0, eval_every=500):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    A, ACT, B = trans[:, 0].to(device), trans[:, 1].to(device), trans[:, 2].to(device)
    is_move = (A != B)
    n = trans.shape[0]
    ea, eb = _sample_pairs(3000, seed=123)          # fixed eval set for trajectory
    use_reg = (model.head == "cross_reg")
    model.train()
    for s in range(steps):
        idx = torch.randint(0, n, (batch,), device=device)
        a_id, act, b_id = A[idx], ACT[idx], B[idx]
        ap, at = model.pos_torus(a_id)
        bp, bt = model.pos_torus(b_id)
        tok_a = model.state_tokens(ap, at)
        tok_b = model.state_tokens(bp, bt)

        pred = model.dynamics(tok_a, act)
        loss_dyn = model.distance(pred, tok_b).square().mean()

        mv = is_move[idx]
        loss_anchor = ((model.distance(tok_a[mv], tok_b[mv]) - 1.0).square().mean()
                       if mv.any() else torch.zeros((), device=device))

        r_id = torch.randint(0, N_STATES, (batch,), device=device)
        rp, rt = model.pos_torus(r_id)
        tok_r = model.state_tokens(rp, rt)
        loss_rep = F.softplus(rep_offset - model.distance(tok_a, tok_r)).mean()

        loss = dyn_weight * loss_dyn + anchor_weight * loss_anchor + rep_weight * loss_rep

        if use_reg:
            # identity: D(x,x) -> 0
            loss_id = model.distance(tok_a, tok_a).square().mean()
            # triangle: D(a,c) <= D(a,b) + D(b,c)
            c_id = torch.randint(0, N_STATES, (batch,), device=device)
            cp, ct = model.pos_torus(c_id)
            tok_c = model.state_tokens(cp, ct)
            d_ac = model.distance(tok_a, tok_c)
            d_ab = model.distance(tok_a, tok_b)
            d_bc = model.distance(tok_b, tok_c)
            loss_tri = F.relu(d_ac - d_ab - d_bc).mean()
            loss = loss + id_weight * loss_id + tri_weight * loss_tri

        opt.zero_grad()
        loss.backward()
        opt.step()

        if (s + 1) % eval_every == 0 or s == 0:
            sp = spearman_geo(model, geo, ea, eb, device)
            det, _ = detour_signature(model, tg_bridge, device)
            print(f"[{model.head}] step {s+1}/{steps} loss {loss.item():.3f} "
                  f"| dyn {loss_dyn.item():.3f} anc {loss_anchor.item():.3f} rep {loss_rep.item():.3f} "
                  f"| spearman {sp:.3f} detour {det:.3f}")
    return model


def train_baseline(model, trans, geo, steps, batch, lr, device,
                   rep_weight=1.0, rep_offset=10.0, eval_every=500):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    A, B = trans[:, 0].to(device), trans[:, 2].to(device)
    mm = (A != B)
    Am, Bm = A[mm], B[mm]
    nm = Am.shape[0]
    ea, eb = _sample_pairs(3000, seed=123)
    model.train()
    for s in range(steps):
        idx = torch.randint(0, nm, (batch,), device=device)
        a_id, b_id = Am[idx], Bm[idx]
        loss_anchor = (model.distance_ids(a_id, b_id) - 1.0).square().mean()
        r_id = torch.randint(0, N_STATES, (batch,), device=device)
        loss_rep = F.softplus(rep_offset - model.distance_ids(a_id, r_id)).mean()
        loss = loss_anchor + rep_weight * loss_rep
        opt.zero_grad()
        loss.backward()
        opt.step()
        if (s + 1) % eval_every == 0 or s == 0:
            with torch.no_grad():
                d = model.distance_ids(torch.as_tensor(ea, device=device),
                                       torch.as_tensor(eb, device=device)).cpu().numpy()
            sp = spearmanr(d, geo[ea, eb]).statistic
            print(f"[baseline] step {s+1}/{steps} loss {loss.item():.3f} "
                  f"| anc {loss_anchor.item():.3f} rep {loss_rep.item():.3f} | spearman {sp:.3f}")
    return model


def final_eval(model_f, model_b, geo, tg_bridge, device, n_pairs=8000, seed=7):
    a, b = _sample_pairs(n_pairs, seed)
    g = geo[a, b]
    d_f = factored_distance_ids(model_f, a, b, device)
    with torch.no_grad():
        d_b = model_b.distance_ids(torch.as_tensor(a, device=device),
                                   torch.as_tensor(b, device=device)).cpu().numpy()
    sp_f = spearmanr(d_f, g).statistic
    sp_b = spearmanr(d_b, g).statistic
    det_f, _ = detour_signature(model_f, tg_bridge, device)
    px = np.array([gid(p, 0) for p in range(N_POS)])
    py = np.array([gid(p, 1) for p in range(N_POS)])
    with torch.no_grad():
        d_cross_b = model_b.distance_ids(torch.as_tensor(px, device=device),
                                         torch.as_tensor(py, device=device)).cpu().numpy()
    det_b = spearmanr(d_cross_b, tg_bridge).statistic

    pass1 = (sp_f - sp_b) >= SPEARMAN_MARGIN
    pass2 = det_f >= DETOUR_MIN
    verdict = "PASS" if (pass1 and pass2) else "FAIL"
    print("\n================ RESULTS ================")
    print(f"head={model_f.head}")
    print(f"Spearman(D, geodesic)     factored={sp_f:.3f}  baseline={sp_b:.3f}  "
          f"margin={sp_f - sp_b:+.3f} (need >= {SPEARMAN_MARGIN})")
    print(f"Detour signature Spearman factored={det_f:.3f}  baseline={det_b:.3f}  "
          f"(factored need >= {DETOUR_MIN})")
    print(f"crit1 beat-baseline: {'ok' if pass1 else 'NO'}   crit2 detour: {'ok' if pass2 else 'NO'}")
    print(f"VERDICT: {verdict}")
    print("=========================================")
    return dict(head=model_f.head, sp_f=float(sp_f), sp_b=float(sp_b),
                det_f=float(det_f), det_b=float(det_b), verdict=verdict)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default="self_norm",
                    choices=["cross_scalar", "cross_reg", "self_norm"])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--json_out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    print(f"device={device} head={args.head} steps={args.steps} batch={args.batch} "
          f"lr={args.lr} seed={args.seed}")

    G = build_graph()
    print(f"nodes={G.number_of_nodes()} edges={G.number_of_edges()} connected={nx.is_connected(G)}")
    trans = build_transitions()
    geo = all_pairs_geodesic(G)
    tg_bridge = torus_geo_to_bridge(G)
    print(f"transitions={trans.shape[0]} max_geodesic={geo.max():.0f} max_to_bridge={tg_bridge.max():.0f}")

    print(f"\ntraining factored ({args.head}) ...")
    model_f = FactoredAttentionModel(head=args.head).to(device)
    train_factored(model_f, trans, geo, tg_bridge, args.steps, args.batch, args.lr, device,
                   eval_every=args.eval_every)

    print("\ntraining baseline ...")
    model_b = PlainEmbeddingModel().to(device)
    train_baseline(model_b, trans, geo, args.steps, args.batch, args.lr, device,
                   eval_every=args.eval_every)

    res = final_eval(model_f, model_b, geo, tg_bridge, device, seed=args.seed)
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()

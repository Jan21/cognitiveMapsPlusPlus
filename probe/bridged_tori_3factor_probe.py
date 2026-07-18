"""
Experiment C: THREE factors of variation, different tasks/actions touch different ones.

Factors:
  position : 15x15 wraparound torus  (changed by up/down/left/right)
  id       : which torus {0,1}, bridged at node 150 (changed by next/prev, only at 150)
  color    : {0,1}, a global attribute (changed by `recolor`, available everywhere)
=> 225*2*2 = 900 states, 7 actions.

Model: 3 factor tokens [z_pos, z_id, z_color] -> self_norm distance head + dynamics.
Question: (a) does the factored approach still recover the 900-state geodesic with 3
factors, and (b) do the 3 tokens disentangle -- each token sensitive to exactly one
factor? Compared: no aux vs generalized invar (min over tokens of change/spread, per
action-group), which asks one token to be invariant under each action type.

Disentanglement measured attention-free as a 3x3 token-vs-factor sensitivity matrix.
"""

import argparse
import json
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

SIDE = 15
N_POS = SIDE * SIDE
N_ID = 2
N_COLOR = 2
N_STATES = N_POS * N_ID * N_COLOR          # 900
BRIDGE_POS = 150
N_ACTIONS = 7
UP, DOWN, LEFT, RIGHT, NEXT, PREV, RECOLOR = range(N_ACTIONS)


def gid3(pos, idx, color):
    return color * (N_POS * N_ID) + idx * N_POS + pos


def decode(g):
    color = g // (N_POS * N_ID)
    rem = g % (N_POS * N_ID)
    return rem % N_POS, rem // N_POS, color        # pos, id, color


def move(pos, action):
    r, c = divmod(pos, SIDE)
    if action == UP:    r = (r - 1) % SIDE
    elif action == DOWN:  r = (r + 1) % SIDE
    elif action == LEFT:  c = (c - 1) % SIDE
    elif action == RIGHT: c = (c + 1) % SIDE
    return r * SIDE + c


def step(pos, idx, color, a):
    if a < 4:
        return move(pos, a), idx, color
    if a == NEXT:
        return (pos, 1, color) if (pos == BRIDGE_POS and idx == 0) else (pos, idx, color)
    if a == PREV:
        return (pos, 0, color) if (pos == BRIDGE_POS and idx == 1) else (pos, idx, color)
    if a == RECOLOR:
        return pos, idx, 1 - color


def build_graph():
    G = nx.Graph()
    G.add_nodes_from(range(N_STATES))
    for color in range(N_COLOR):
        for idx in range(N_ID):
            for pos in range(N_POS):
                u = gid3(pos, idx, color)
                for a in (UP, DOWN, LEFT, RIGHT):
                    G.add_edge(u, gid3(move(pos, a), idx, color))
                G.add_edge(u, gid3(pos, idx, 1 - color))               # recolor
        G.add_edge(gid3(BRIDGE_POS, 0, color), gid3(BRIDGE_POS, 1, color))  # bridge
    return G


def build_transitions():
    rows = []
    for color in range(N_COLOR):
        for idx in range(N_ID):
            for pos in range(N_POS):
                for a in range(N_ACTIONS):
                    np_, ni, nc = step(pos, idx, color, a)
                    rows.append((gid3(pos, idx, color), a, gid3(np_, ni, nc)))
    return torch.tensor(rows, dtype=torch.long)


def all_pairs_geodesic(G):
    D = np.zeros((N_STATES, N_STATES), dtype=np.float32)
    for src, lengths in nx.all_pairs_shortest_path_length(G):
        for dst, d in lengths.items():
            D[src, dst] = d
    return D


ID_MARKERS = (0, 224)            # value 2 when id==1
COLOR_MARKERS = (14, 210)        # value 3 when color==1


class ThreeFactorModel(nn.Module):
    def __init__(self, d_model=32, n_layers=2, n_heads=4, emb_dim=16, dyn_hidden=128, p=1.5,
                 image=False):
        super().__init__()
        self.d_model = d_model
        self.p = p
        self.K = 3
        self.image = image
        if image:
            # shared 15x15 image, pixel values {0=bg,1=agent,2=id-marker,3=color-marker};
            # 3 query vectors must LEARN to attend to their factor (emergence test).
            self.pixel_val_emb = nn.Embedding(4, d_model)
            self.pixel_pos_emb = nn.Embedding(N_POS, d_model)
            self.img_query = nn.Parameter(torch.randn(self.K, d_model) * 0.02)
            self.cross_attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
            self.enc_norm = nn.LayerNorm(d_model)
            self._last_attn = None
        else:
            self.E_pos = nn.Embedding(N_POS, d_model)
            self.E_id = nn.Embedding(N_ID, d_model)
            self.E_color = nn.Embedding(N_COLOR, d_model)
        self.type_emb = nn.Embedding(self.K, d_model)
        self.query = nn.Parameter(torch.randn(d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=0.0,
                                         batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(enc, n_layers)
        self.proj = nn.Linear(d_model, emb_dim)
        self.act_emb = nn.Embedding(N_ACTIONS, d_model)
        self.dyn = nn.Sequential(nn.Linear(self.K * d_model + d_model, dyn_hidden), nn.GELU(),
                                 nn.Linear(dyn_hidden, dyn_hidden), nn.GELU(),
                                 nn.Linear(dyn_hidden, self.K * d_model))

    def decode(self, ids):
        color = ids // (N_POS * N_ID)
        rem = ids % (N_POS * N_ID)
        return rem % N_POS, rem // N_POS, color

    def render(self, ids):
        pos, idx, color = self.decode(ids)
        B = ids.shape[0]
        grid = torch.zeros(B, N_POS, dtype=torch.long, device=ids.device)
        is_id = (idx == 1); is_col = (color == 1)
        for m in ID_MARKERS:
            grid[is_id, m] = 2
        for m in COLOR_MARKERS:
            grid[is_col, m] = 3
        grid[torch.arange(B, device=ids.device), pos] = 1        # agent overwrites (occlusion)
        return grid

    def encode(self, grid):
        B = grid.shape[0]
        pix = self.pixel_val_emb(grid) + self.pixel_pos_emb(torch.arange(N_POS, device=grid.device)).unsqueeze(0)
        Q = self.img_query.unsqueeze(0).expand(B, -1, -1)
        z, attn = self.cross_attn(Q, pix, pix, need_weights=True, average_attn_weights=True)
        self._last_attn = attn.detach()
        return self.enc_norm(z)

    def state_tokens(self, ids):
        if self.image:
            return self.encode(self.render(ids))
        pos, idx, color = self.decode(ids)
        return torch.stack([self.E_pos(pos), self.E_id(idx), self.E_color(color)], dim=1)  # (B,3,d)

    def embed(self, tok):
        B = tok.shape[0]
        typ = self.type_emb(torch.arange(self.K, device=tok.device))
        seq = torch.cat([self.query.expand(B, 1, self.d_model), tok + typ], dim=1)
        return self.proj(self.encoder(seq)[:, 0])

    def distance(self, tok_u, tok_v):
        return torch.norm(self.embed(tok_u) - self.embed(tok_v), p=self.p, dim=-1)

    def dynamics(self, tok, action):
        B = tok.shape[0]
        inp = torch.cat([tok.reshape(B, -1), self.act_emb(action)], dim=-1)
        return tok + self.dyn(inp).view(B, self.K, self.d_model)


FACTOR_NAMES = ["position", "id", "color"]


@torch.no_grad()
def sensitivity_matrix(model, device):
    """3x3: rows = tokens, cols = factors. entry = mean token change when that factor
    alone flips/moves. Clean disentanglement = a permutation (each token one factor)."""
    model.eval()
    rng = np.random.default_rng(0)
    ids = torch.tensor(rng.integers(0, N_STATES, 400), device=device)
    pos, idx, color = model.decode(ids)
    base = model.state_tokens(ids)
    # position: move right
    r = pos // SIDE; c = (pos % SIDE + 1) % SIDE
    posn = r * SIDE + c
    tp = model.state_tokens(gid3(posn, idx, color))
    # id: flip torus (only valid crossing at 150 conceptually, but token change is defined for any)
    ti = model.state_tokens(gid3(pos, 1 - idx, color))
    # color: flip
    tc = model.state_tokens(gid3(pos, idx, 1 - color))
    M = np.stack([
        (base - tp).norm(dim=2).mean(0).cpu().numpy(),
        (base - ti).norm(dim=2).mean(0).cpu().numpy(),
        (base - tc).norm(dim=2).mean(0).cpu().numpy(),
    ], axis=1)                                     # (3 tokens, 3 factors)
    return M


def invar_aux(model, tok_a, tok_b, tok_r, act, a_id, b_id, device):
    """generalized invariance: for each action-group, min over tokens of change/spread."""
    spread = (tok_a - tok_r).norm(dim=2).mean(0) + 1e-4            # (3,)
    changed = (a_id != b_id)
    total = torch.zeros((), device=device)
    for grp in [(act < 4), ((act >= 4) & (act < 6)), (act == RECOLOR)]:
        m = grp & changed
        if m.any():
            ch = (tok_a[m] - tok_b[m]).norm(dim=2).mean(0)         # (3,)
            total = total + (ch / spread).min()
    return total


def train(model, trans, geo, steps, batch, lr, device, aux_mode="none", aux_weight=3.0,
          aux_warmup=1000, rep_offset=10.0, eval_every=1000):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    A, ACT, B = trans[:, 0].to(device), trans[:, 1].to(device), trans[:, 2].to(device)
    is_move = (A != B)
    n = trans.shape[0]
    rng = np.random.default_rng(123)
    ea = torch.tensor(rng.integers(0, N_STATES, 3000)); eb = torch.tensor(rng.integers(0, N_STATES, 3000))
    model.train()
    for s in range(steps):
        idx = torch.randint(0, n, (batch,), device=device)
        a_id, act, b_id = A[idx], ACT[idx], B[idx]
        tok_a = model.state_tokens(a_id); tok_b = model.state_tokens(b_id)
        r_id = torch.randint(0, N_STATES, (batch,), device=device)
        tok_r = model.state_tokens(r_id)
        pred = model.dynamics(tok_a, act)
        loss_dyn = model.distance(pred, tok_b).square().mean()
        mv = is_move[idx]
        loss_anc = ((model.distance(tok_a[mv], tok_b[mv]) - 1.0).square().mean()
                    if mv.any() else torch.zeros((), device=device))
        loss_rep = F.softplus(rep_offset - model.distance(tok_a, tok_r)).mean()
        loss = loss_dyn + loss_anc + loss_rep
        if aux_mode == "invar":
            w = aux_weight * min(1.0, (s + 1) / max(1, aux_warmup))
            loss = loss + w * invar_aux(model, tok_a, tok_b, tok_r, act, a_id, b_id, device)
        opt.zero_grad(); loss.backward(); opt.step()
        if (s + 1) % eval_every == 0 or s == 0:
            with torch.no_grad():
                d = model.distance(model.state_tokens(ea.to(device)), model.state_tokens(eb.to(device))).cpu().numpy()
            sp = spearmanr(d, geo[ea.numpy(), eb.numpy()]).statistic
            print(f"[{aux_mode}] step {s+1}/{steps} loss {loss.item():.3f} | spearman {sp:.3f}")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--aux", default="none", choices=["none", "invar"])
    ap.add_argument("--input", default="image", choices=["image", "tokens"])
    ap.add_argument("--eval_every", type=int, default=1500)
    ap.add_argument("--json_out", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    device = torch.device(args.device)
    print(f"device={device} 3-FACTOR (pos,id,color) 900 states, 7 actions, "
          f"input={args.input}, aux={args.aux}, seed={args.seed}")

    G = build_graph()
    print(f"nodes={G.number_of_nodes()} edges={G.number_of_edges()} connected={nx.is_connected(G)}")
    trans = build_transitions()
    geo = all_pairs_geodesic(G)
    print(f"transitions={trans.shape[0]} max_geodesic={geo.max():.0f}")

    model = ThreeFactorModel(image=(args.input == "image")).to(device)
    train(model, trans, geo, args.steps, args.batch, args.lr, device,
          aux_mode=args.aux, eval_every=args.eval_every)

    rng = np.random.default_rng(7)
    a = rng.integers(0, N_STATES, 8000); b = rng.integers(0, N_STATES, 8000)
    m = a != b; a, b = a[m], b[m]
    with torch.no_grad():
        d = model.distance(model.state_tokens(torch.tensor(a, device=device)),
                           model.state_tokens(torch.tensor(b, device=device))).cpu().numpy()
    sp = spearmanr(d, geo[a, b]).statistic
    M = sensitivity_matrix(model, device)

    print("\n================ 3-FACTOR RESULTS ================")
    print(f"Spearman(D, 900-state geodesic) = {sp:.3f}")
    print("token-vs-factor sensitivity matrix (rows=token 0/1/2, cols=pos,id,color):")
    for k in range(3):
        print(f"  token{k}: pos={M[k,0]:.2f}  id={M[k,1]:.2f}  color={M[k,2]:.2f}")
    # disentanglement score: each token's dominant factor, is the assignment a permutation?
    dom = M.argmax(1)
    clean = len(set(dom.tolist())) == 3
    print(f"dominant factor per token = {[FACTOR_NAMES[i] for i in dom]}  -> "
          f"{'DISENTANGLED (permutation)' if clean else 'NOT clean'}")
    print("==================================================")

    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump(dict(sp=float(sp), sens_matrix=M.tolist(),
                           dominant=[FACTOR_NAMES[i] for i in dom], clean=bool(clean)), f, indent=2)
        print("wrote", args.json_out)


if __name__ == "__main__":
    main()

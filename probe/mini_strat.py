"""Minimal stratified debug harness. ONE agent on an 8x8 torus, knob in {all=2D, horiz=1D(col),
vert=1D(row)}. No flipping. Only 3*64 = 192 states -> enumerate and embed ALL of them, so we can
read the embedding geometry exactly (no sampling, no BFS).

Question: at a horiz state the agent is locked to its row (legal = column moves). Does the embedding
treat a ROW move (illegal) as near (naive, not stratified) or far/collapsed (stratified)?
"""
import argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F

G = 8; NK = 3; NPOS = G * G; P = NPOS + 1; VOCAB = max(2, NK)
# knob 0 all (row&col), 1 horiz (col only), 2 vert (row only)
def axes(knob):  # returns (row_ok, col_ok) as bool tensors
    return (knob == 0) | (knob == 2), (knob == 0) | (knob == 1)

def render(pos, knob):
    B, dev = pos.shape[0], pos.device
    grid = torch.zeros(B, P, dtype=torch.long, device=dev)
    grid[torch.arange(B, device=dev), pos] = 1
    grid[:, NPOS] = knob
    return grid

class ImageEnc(nn.Module):
    def __init__(self, d=32, D=16, slots=4):
        super().__init__()
        self.val = nn.Embedding(VOCAB, d); self.pe = nn.Embedding(P, d)
        self.slots = nn.Parameter(torch.randn(slots, d) * 0.02)
        self.attn = nn.MultiheadAttention(d, 4, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(slots * d, 64), nn.GELU(), nn.Linear(64, D))
    def embed(self, pos, knob):
        grid = render(pos, knob); B = grid.shape[0]
        pix = self.val(grid) + self.pe(torch.arange(P, device=grid.device))[None]
        z, _ = self.attn(self.slots[None].expand(B, -1, -1), pix, pix)
        return self.proj(z.reshape(B, -1))

class FactEnc(nn.Module):
    def __init__(self, d=32, D=16, gate="hardwired"):
        super().__init__(); self.gate = gate
        self.knob = nn.Embedding(NK, d); self.row = nn.Embedding(G, d); self.col = nn.Embedding(G, d)
        self.gnet = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 2))
        self.proj = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, D))
    def _gate(self, knob):
        if self.gate == "hardwired":
            return ((knob == 0) | (knob == 2)).float(), ((knob == 0) | (knob == 1)).float()
        if self.gate == "none":
            o = torch.ones_like(knob, dtype=torch.float); return o, o
        g = torch.sigmoid(self.gnet(self.knob(knob))); return g[..., 0], g[..., 1]
    def embed(self, pos, knob):
        r, c = pos // G, pos % G; gr, gc = self._gate(knob)
        tok = self.knob(knob) + gr[..., None] * self.row(r) + gc[..., None] * self.col(c)
        return self.proj(tok)

def rand_state(n, dev):
    return torch.randint(0, NPOS, (n,), device=dev), torch.randint(0, NK, (n,), device=dev)

def move_pair(pos, knob, K):
    n, dev = pos.shape[0], pos.device
    r, c = pos // G, pos % G; row_ok, col_ok = axes(knob)
    dr = torch.randint(-K, K + 1, (n,), device=dev) * row_ok.long()
    dc = torch.randint(-K, K + 1, (n,), device=dev) * col_ok.long()
    tgt = ((r + dr) % G) * G + ((c + dc) % G)
    return tgt, (dr.float() ** 2 + dc.float() ** 2).sqrt()

def illegal_move(pos, knob):
    """one step along a DISALLOWED axis. mask=1 where such an axis exists (knob!=all)."""
    n, dev = pos.shape[0], pos.device
    r, c = pos // G, pos % G; row_ok, col_ok = axes(knob)
    do_row = ~row_ok                                  # horiz: row disallowed
    mask = (~row_ok | ~col_ok).float()                # all-knob has neither -> mask 0
    nr = torch.where(do_row, (r + 1) % G, r)
    nc = torch.where(~do_row & ~col_ok, (c + 1) % G, c)
    return nr * G + nc, mask

def train(model, steps, dev, K=2, lam_col=0.0, lam_rep=0.0, margin=6.0, lr=3e-3):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for s in range(steps):
        pos, knob = rand_state(256, dev)
        eu = model.embed(pos, knob)
        tgt, dist = move_pair(pos, knob, K)
        loss = (torch.norm(eu - model.embed(tgt, knob), dim=-1) - dist).square().mean()
        rp, rk = rand_state(256, dev)
        loss = loss + F.softplus(8.0 - torch.norm(eu - model.embed(rp, rk), dim=-1)).mean()
        pim, m = illegal_move(pos, knob)
        d_im = torch.norm(eu - model.embed(pim, knob), dim=-1)
        if lam_col > 0:
            loss = loss + lam_col * (d_im.square() * m).sum() / m.sum().clamp(min=1)
        if lam_rep > 0:
            loss = loss + lam_rep * (F.relu(margin - d_im).square() * m).sum() / m.sum().clamp(min=1)
        opt.zero_grad(); loss.backward(); opt.step()
    return model

def vgt(dist, lo=0.05, hi=0.6):
    d = np.sort(dist[dist > 1e-9]); N = d.size
    if N < 10: return np.nan
    a, b = max(3, int(lo * N)), int(hi * N)
    if b - a < 4 or d[b - 1] - d[a] < 1e-6: return np.nan
    return float(np.polyfit(np.log(d[a:b]), np.log(np.arange(1, N + 1, dtype=float)[a:b]), 1)[0])

@torch.no_grad()
def analyse(model, dev, tag):
    P0 = torch.arange(NPOS, device=dev)
    allp = torch.cat([P0, P0, P0]); allk = torch.cat([torch.zeros(NPOS), torch.ones(NPOS), 2 * torch.ones(NPOS)]).long().to(dev)
    E = model.embed(allp, allk).cpu().numpy()
    print(f"\n=== {tag} ===")
    names = {0: "all (2D)", 1: "horiz (1D col)", 2: "vert (1D row)"}
    for k in range(NK):
        idx = np.where(allk.cpu().numpy() == k)[0]
        pos = allp[idx].cpu().numpy(); Ek = E[idx]
        r, c = pos // G, pos % G
        # step neighbours in the SAME knob
        def emb_of(rr, cc):
            p = (rr % G) * G + (cc % G)
            return E[k * NPOS + p]
        drow = np.linalg.norm(Ek - emb_of(r + 1, c), axis=1)   # row move
        dcol = np.linalg.norm(Ek - emb_of(r, c + 1), axis=1)   # col move
        # emb-NN local dim over ALL 192 states, for a representative state of this knob
        s0 = k * NPOS + (3 * G + 3)
        dall = np.linalg.norm(E - E[s0], axis=1)
        nn = np.sort(dall)[:40]
        row_ok = k in (0, 2); col_ok = k in (0, 1)
        print(f"  {names[k]:15s}  d_row={np.median(drow):.2f}{' (legal)' if row_ok else ' (ILLEGAL)':>10}"
              f"   d_col={np.median(dcol):.2f}{' (legal)' if col_ok else ' (ILLEGAL)':>10}"
              f"   embNN_dim={vgt(dall):.2f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dev = torch.device("cpu")
    for tag, kind in [("baseline (iso only, image)", dict()),
                      ("collapse (image)", dict(lam_col=1.0)),
                      ("repel (image)", dict(lam_rep=1.0)),
                      ("factored HARDWIRED gate", dict(fact="hardwired")),
                      ("factored LEARNED gate + collapse", dict(fact="learned", lam_col=1.0))]:
        torch.manual_seed(args.seed); np.random.seed(args.seed)
        if kind.get("fact"):
            model = FactEnc(gate=kind["fact"])
        else:
            model = ImageEnc()
        train(model, args.steps, dev, lam_col=kind.get("lam_col", 0.0), lam_rep=kind.get("lam_rep", 0.0))
        analyse(model, dev, tag)

if __name__ == "__main__":
    main()

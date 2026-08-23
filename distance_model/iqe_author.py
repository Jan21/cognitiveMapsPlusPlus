"""IQE applied the way Wang & Isola (arXiv:2211.15120) would: NO slot/transformer machinery.

Plain encoder (deep MLP with BatchNorm, or a small CNN for the image obs) -> 512-d latent ->
torchqmet IQE head. Training recipe from the paper: MSE on gamma-discounted distances
(gamma=0.9), Adam with cosine decay to 0, batch 1024. Same environment, bed, pools, and
held-out evaluation as switchyard.py (imported), so numbers are directly comparable.

Input: the same 12 pureimage planes + objectness channel (13 x 7 x 7), flattened for the MLP.

Usage: python3 iqe_author.py --encoder mlp --red maxmean --l 64 --loss disc --lr 1e-4 \
       --steps 16000 --nmaps 683 --poolq 6800 --seed 0 --tag v1
"""
import argparse, json, math
import numpy as np
import torch, torch.nn as nn

import switchyard as sw


def build_render(a, yards, dev):
    G = a.G; ncell = G * G
    n = len(yards)
    GC = torch.zeros(n, a.ngate, dtype=torch.long); LC = torch.zeros(n, a.nlever, dtype=torch.long)
    LW = torch.zeros(n, a.nlever, a.ngate); PC = torch.zeros(n, dtype=torch.long)
    PW = torch.zeros(n, a.ngate); CC = torch.zeros(n, dtype=torch.long); CD = torch.zeros(n, dtype=torch.long)
    CELL = torch.zeros(n, ncell, dtype=torch.long); WALLIMG = torch.zeros(n, ncell)
    cell = lambda y, rc: rc[0] * G + rc[1]
    for i, y in enumerate(yards):
        for r in range(G):
            for c in range(G):
                if y.wall[r, c] and (r, c) not in y.gates: WALLIMG[i, r * G + c] = 1.0
        GC[i] = torch.tensor([cell(y, g) for g in y.gates]); LC[i] = torch.tensor([cell(y, l) for l in y.levers])
        for li, wm in enumerate(y.wiring): LW[i, li] = torch.tensor([(wm >> g) & 1 for g in range(a.ngate)], dtype=torch.float)
        PC[i] = cell(y, y.plate); PW[i] = torch.tensor([(y.platemask >> g) & 1 for g in range(a.ngate)], dtype=torch.float)
        ch = list(y.chutes.items())[0] if y.chutes else ((0, 0), 0)
        CC[i] = cell(y, ch[0]); CD[i] = ch[1]
        CELL[i, :len(y.cells)] = torch.tensor([cell(y, rc) for rc in y.cells])
    GC, LC, LW, PC, PW, CC, CD, CELL, WALLIMG = (t.to(dev) for t in (GC, LC, LW, PC, PW, CC, CD, CELL, WALLIMG))
    PIMGC = 5 + a.nlever + 1 + 4

    def render(x, m):
        """(B,3) state, (B,) mapid -> (B, PIMGC+1, G, G): pureimage planes + objectness."""
        B = x.shape[0]
        wc = CELL[m].gather(1, x[:, 0:2])
        bits = torch.stack([(x[:, 2] >> g) & 1 for g in range(a.ngate)], 1).float()
        img = torch.zeros(B, PIMGC, ncell, device=x.device)
        img[:, 0] = WALLIMG[m]
        img[:, 1].scatter_(1, wc[:, 0:1], 1.0); img[:, 2].scatter_(1, wc[:, 1:2], 1.0)
        img[:, 3].scatter_(1, GC[m], 1.0); img[:, 4].scatter_(1, GC[m], (bits > 0).float())
        for l in range(a.nlever):
            img[:, 5 + l].scatter_(1, LC[m][:, l:l + 1], 1.0)
            img[:, 5 + l].scatter_(1, GC[m], LW[m][:, l, :])
        pc = 5 + a.nlever
        img[:, pc].scatter_(1, PC[m][:, None], 1.0); img[:, pc].scatter_(1, GC[m], PW[m])
        for dd in range(4):
            img[:, pc + 1 + dd].scatter_(1, CC[m][:, None], (CD[m] == dd).float()[:, None])
        obj = (img[:, 1:] > 0.75).any(1, keepdim=True).float()
        return torch.cat([img, obj], 1).view(B, PIMGC + 1, G, G)
    return render, PIMGC + 1


class MLPEnc(nn.Module):
    """Paper recipe: input-2048-2048-2048-512 ReLU with BatchNorm after each activation."""
    def __init__(s, nin, dlat=512):
        super().__init__()
        L = []
        d0 = nin
        for _ in range(3):
            L += [nn.Linear(d0, 2048), nn.ReLU(), nn.BatchNorm1d(2048)]; d0 = 2048
        L += [nn.Linear(2048, dlat)]
        s.net = nn.Sequential(*L)
    def forward(s, img): return s.net(img.flatten(1))


class CNNEnc(nn.Module):
    """Small conv encoder as the authors use for image observations, then the MLP tail."""
    def __init__(s, cin, G, dlat=512):
        super().__init__()
        s.conv = nn.Sequential(nn.Conv2d(cin, 32, 3, padding=1), nn.ReLU(),
                               nn.Conv2d(32, 64, 3, padding=1), nn.ReLU())
        s.tail = nn.Sequential(nn.Linear(64 * G * G, 2048), nn.ReLU(), nn.BatchNorm1d(2048),
                               nn.Linear(2048, dlat))
    def forward(s, img): return s.tail(s.conv(img).flatten(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encoder", choices=["mlp", "cnn"], default="mlp")
    ap.add_argument("--red", choices=["maxmean", "sum"], default="maxmean")
    ap.add_argument("--l", type=int, default=64, help="IQE dim_per_component (latent 512 -> k=512/l components)")
    ap.add_argument("--loss", choices=["disc", "mse"], default="disc", help="disc = MSE on gamma^d (paper); mse = raw distance")
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--steps", type=int, default=16000)
    ap.add_argument("--bs", type=int, default=1024)
    ap.add_argument("--dlat", type=int, default=512)
    ap.add_argument("--nmaps", type=int, default=683); ap.add_argument("--poolq", type=int, default=6800)
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--split", default="map")
    ap.add_argument("--tag", default="")
    # env constants matching switchyard defaults
    for k, v in dict(G=7, ngate=3, nlever=2, nchute=1, Rmax=28).items():
        ap.add_argument(f"--{k}", type=int, default=v)
    a = ap.parse_args()
    a.wire1 = a.noplate = a.nopush = a.gatesopen = False; a.Rtrain = 0
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)

    yards, tr_ids, te_ids = sw.make_yards(a)
    S1, S2, D, C = sw.build_pool(a, rng, yards, tr_ids, a.Rmax)
    E1, E2, ED, EC = sw.build_pool(a, np.random.default_rng(a.seed + 99), yards, te_ids, a.Rmax)
    S1t, S2t, Dt, Ct = (torch.as_tensor(x, device=dev) for x in (S1, S2, D, C))
    E1t, E2t, EDt, ECt = (torch.as_tensor(x, device=dev) for x in (E1, E2, ED, EC))
    print(f"pool train={len(S1)} test={len(E1)} split={a.split}", flush=True)

    render, CIN = build_render(a, yards, dev)
    import torchqmet
    enc = (MLPEnc(CIN * a.G * a.G, a.dlat) if a.encoder == "mlp" else CNNEnc(CIN, a.G, a.dlat)).to(dev)
    head = torchqmet.IQE(a.dlat, dim_per_component=a.l, reduction=a.red).to(dev)
    params = list(enc.parameters()) + list(head.parameters())
    npar = sum(p.numel() for p in params if p.requires_grad)
    print(f"params {npar}", flush=True)
    opt = torch.optim.Adam(params, a.lr)

    for step in range(a.steps):
        cd = 0.5 * (1 + math.cos(math.pi * step / a.steps))        # cosine to 0, no restarts (paper)
        for g in opt.param_groups: g["lr"] = a.lr * cd
        b = torch.randint(0, len(S1t), (a.bs,), device=dev)
        z1 = enc(render(S1t[b], Ct[b])); z2 = enc(render(S2t[b], Ct[b]))
        pred = head(z1, z2)
        if a.loss == "disc":
            loss = ((a.gamma ** pred) - (a.gamma ** Dt[b])).pow(2).mean()
        else:
            loss = (pred - Dt[b]).pow(2).mean()
        opt.zero_grad(); loss.backward(); opt.step()
        if step % max(1, a.steps // 8) == 0:
            print(f"step {step} loss {loss.item():.5f}", flush=True)

    enc.eval(); head.eval()
    with torch.no_grad():
        pr = torch.cat([head(enc(render(E1t[i:i + 4000], ECt[i:i + 4000])),
                             enc(render(E2t[i:i + 4000], ECt[i:i + 4000]))) for i in range(0, len(E1t), 4000)])
    corr = float(np.corrcoef(pr.cpu().numpy(), ED)[0, 1]); mae = float((pr - EDt).abs().mean())
    print("RESULT " + json.dumps(dict(tag=a.tag, encoder=a.encoder, red=a.red, l=a.l, loss=a.loss,
                                      lr=a.lr, steps=a.steps, bs=a.bs, dlat=a.dlat, params=npar,
                                      nmaps=a.nmaps, poolq=a.poolq, split=a.split, seed=a.seed,
                                      test_corr=round(corr, 3), test_mae=round(mae, 3))), flush=True)


if __name__ == "__main__":
    main()

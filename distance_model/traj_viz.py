"""Extract decoded-walk visualization data: for a few held-out pairs, the map layout, the true
BFS worker path, and the decoded worker-slot cell distribution per pass. JSON out; rendered as
SVG in the report page.

Usage: python3 traj_viz.py <ckpt.pt> <traj.npz> --out viz.json [--n 6]
"""
import sys, os, json, argparse, collections
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import switchyard as sw
from traj_switchyard import bfs_path, decompose, gcell
from traj_night import train_dec, path_events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("npz"); ap.add_argument("--out", default="viz.json")
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()
    torch.manual_seed(0)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = argparse.Namespace(**ck["args"])
    yards, _, _ = sw.make_yards(a)
    d = np.load(args.npz)
    Z = d["Z"].astype(np.float32); att = d["att_s"].astype(np.float32)
    S1, S2, MID = d["s1"], d["s2"], d["mapid"]
    dt = d["d_true"].astype(np.float64)
    B, T1, K, dim = Z.shape

    w_slot = np.zeros(B, int); wc0 = np.zeros(B, int)
    for i in range(B):
        wc0[i] = gcell(yards[int(MID[i])], S1[i][0])
        w_slot[i] = int(att[i, :, wc0[i]].argmax())
    maps = np.array([int(m) for m in MID]); um = np.unique(maps)
    tr_m = set(um[: int(0.7 * len(um))]); tr = np.array([m in tr_m for m in maps]); te = ~tr
    Zw = Z[np.arange(B), :, w_slot]
    dec = train_dec(Zw[:, 0], wc0, tr, dim)
    with torch.no_grad():
        P = torch.softmax(dec(torch.tensor(Zw.reshape(-1, dim))), 1).reshape(B, T1, 49).numpy()

    picks = []
    for i in np.where(te)[0]:
        if not (8 <= dt[i] <= 16): continue
        yard = yards[int(MID[i])]
        s = tuple(int(x) for x in S1[i]); g = tuple(int(x) for x in S2[i])
        p = bfs_path(yard, s, g)
        if p is None or len(p) - 1 != int(dt[i]): continue
        wcells, ccells, press, nw, npu, nl = path_events(yard, p)
        if nl == 0: continue
        if P[i, 0].argmax() != wc0[i]: continue                    # only well-decoded examples
        pulled = decompose(yard, p)[3]
        picks.append(dict(
            pair=int(i), map=int(MID[i]), d=int(dt[i]), nw=int(nw), npush=int(npu), npull=int(nl),
            wall=[[int(x) for x in row] for row in yard.wall.astype(int)],
            gates=[[int(r), int(c)] for (r, c) in yard.gates],
            levers=[[int(r), int(c)] for (r, c) in yard.levers],
            plate=[int(yard.plate[0]), int(yard.plate[1])],
            start=int(wc0[i]), goal=int(gcell(yard, S2[i][0])),
            crate0=int(gcell(yard, S1[i][1])), crateT=int(gcell(yard, S2[i][1])),
            true_wpath=[int(c) for c in wcells], pulled=[int(c) for c in sorted(pulled)],
            decoded_top=[int(P[i, t].argmax()) for t in range(T1)],
            decoded_dist=[[round(float(x), 4) for x in P[i, t]] for t in range(T1)]))
        if len(picks) >= args.n: break
    with open(args.out, "w") as f:
        json.dump(dict(tag=a.tag, T=int(T1 - 1), examples=picks), f)
    print(f"VIZ {len(picks)} examples -> {args.out}", flush=True)


if __name__ == "__main__":
    main()

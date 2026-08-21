"""Stale-wiring probe: on the rewired-causality pool, does the walk visit the levers that
were correct under the TRAINING wiring of the same layout, rather than the currently correct ones?

Wire split: yards[k] = training wiring of layout k, yards[nmaps+k] = resampled test wiring.
For each eval pair, compute the pulled-lever cells on a BFS-optimal path under BOTH wirings and
compare decoded worker-slot transient mass on stale-only vs current-only lever cells.

Usage: python3 traj_stale.py <ckpt.pt> <traj.npz>   (ckpt args must have split=wire)
"""
import sys, os, json, argparse, collections
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import switchyard as sw
from traj_switchyard import bfs_path, decompose, gcell
from traj_night import train_dec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("npz"); ap.add_argument("--json", default="")
    args = ap.parse_args()
    torch.manual_seed(0)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = argparse.Namespace(**ck["args"])
    assert a.split == "wire", "stale probe needs a wire-split eval dump"
    yards, _, _ = sw.make_yards(a)
    d = np.load(args.npz)
    Z = d["Z"].astype(np.float32); att = d["att_s"].astype(np.float32)
    S1, S2, MID = d["s1"], d["s2"], d["mapid"]
    dt = d["d_true"].astype(np.float64)
    B, T1, K, dim = Z.shape
    res = {"tag": a.tag, "B": int(B)}

    cur_p = [None] * B; stale_p = [None] * B
    groups = collections.defaultdict(list)
    for i in range(B):
        groups[(int(MID[i]), tuple(int(x) for x in S1[i]))].append(i)
    for (m, s), idxs in groups.items():
        ycur = yards[m]; ystale = yards[m - a.nmaps]        # same layout, training wiring
        for i in idxs:
            g = tuple(int(x) for x in S2[i])
            p = bfs_path(ycur, s, g)
            if p is None or len(p) - 1 != int(dt[i]): continue
            cur_p[i] = decompose(ycur, p)[3]                # pulled lever gcells, current wiring
            ps = bfs_path(ystale, s, g)
            stale_p[i] = decompose(ystale, ps)[3] if ps is not None else None

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
    res["dec_acc_t0"] = round(float((P[te, 0].argmax(1) == wc0[te]).mean()), 3)
    tp = slice(1, T1 - 1) if T1 > 2 else slice(1, 2)
    Pt = P[:, tp, :].mean(1)

    ms, mc = [], []
    for i in np.where(te)[0]:
        if cur_p[i] is None or stale_p[i] is None: continue
        so = stale_p[i] - cur_p[i]; co = cur_p[i] - stale_p[i]   # disjoint lever-cell sets
        if not so or not co: continue
        ms.append(float(np.mean([Pt[i, c] for c in so])))
        mc.append(float(np.mean([Pt[i, c] for c in co])))
    res["n_pairs"] = len(ms)
    res["mass_stale_only"] = round(float(np.mean(ms)), 5) if ms else None
    res["mass_current_only"] = round(float(np.mean(mc)), 5) if mc else None
    res["stale_winfrac"] = round(float(np.mean(np.array(ms) > np.array(mc))), 3) if ms else None
    print("STALE " + json.dumps(res), flush=True)
    if args.json:
        with open(args.json, "w") as f: json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

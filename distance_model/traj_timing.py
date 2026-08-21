"""Timing + gate probes on a --dumptraj npz.

  T   lever-visit timing: decoded worker-slot mass on lever cells PER PASS, pull-requiring vs
      pull-free pairs. Early peak = plan-prefix visit; flat = smeared.
  Gt  gate visitation within-pair: decoded mass on gates the true path crosses vs the map's
      other gates (per cell), pairs crossing a strict subset of gates.

Usage: python3 traj_timing.py <ckpt.pt> <traj.npz> [--json out.json]
"""
import sys, os, json, argparse, collections
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import switchyard as sw
from traj_switchyard import bfs_path, gcell, nopull_dist
from traj_night import train_dec, path_events


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("npz"); ap.add_argument("--json", default="")
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
    res = {"tag": a.tag, "B": int(B), "T": int(T1 - 1)}

    wcross = [None] * B; dnp = np.full(B, np.nan); ok = np.zeros(B, bool)
    groups = collections.defaultdict(list)
    for i in range(B):
        groups[(int(MID[i]), tuple(int(x) for x in S1[i]))].append(i)
    for (m, s), idxs in groups.items():
        yard = yards[m]
        gc_map = {(r * yard.G + c) for (r, c) in yard.gates}
        for i in idxs:
            g = tuple(int(x) for x in S2[i])
            p = bfs_path(yard, s, g)
            if p is None or len(p) - 1 != int(dt[i]): continue
            wcells = path_events(yard, p)[0]
            wcross[i] = set(wcells) & gc_map
            dn = nopull_dist(yard, s, g)
            dnp[i] = np.inf if dn is None else dn
            ok[i] = True

    w_slot = np.zeros(B, int); wc0 = np.zeros(B, int)
    lever_cells = [[] for _ in range(B)]; gate_cells = [[] for _ in range(B)]
    for i in range(B):
        yard = yards[int(MID[i])]
        wc0[i] = gcell(yard, S1[i][0]); w_slot[i] = int(att[i, :, wc0[i]].argmax())
        lever_cells[i] = [r * yard.G + c for (r, c) in yard.levers]
        gate_cells[i] = [r * yard.G + c for (r, c) in yard.gates]
    maps = np.array([int(m) for m in MID]); um = np.unique(maps)
    tr_m = set(um[: int(0.7 * len(um))]); tr = np.array([m in tr_m for m in maps]); te = ~tr
    Zw = Z[np.arange(B), :, w_slot]
    dec = train_dec(Zw[:, 0], wc0, tr, dim)
    with torch.no_grad():
        P = torch.softmax(dec(torch.tensor(Zw.reshape(-1, dim))), 1).reshape(B, T1, 49).numpy()
    res["dec_acc_t0"] = round(float((P[te, 0].argmax(1) == wc0[te]).mean()), 3)

    lm = np.zeros((B, T1))
    for i in range(B):
        for c in lever_cells[i]: lm[i] += P[i, :, c]
    need = ok & te & (dnp > dt); free = ok & te & (dnp == dt)
    res["T_lever_per_pass_need"] = [round(float(lm[need, t].mean()), 4) for t in range(T1)]
    res["T_lever_per_pass_free"] = [round(float(lm[free, t].mean()), 4) for t in range(T1)]
    res["T_n"] = [int(need.sum()), int(free.sum())]

    onc, offc = [], []
    for i in np.where(ok & te)[0]:
        if not wcross[i]: continue
        oth = [c for c in gate_cells[i] if c not in wcross[i]]
        if not oth: continue
        tp = slice(1, T1 - 1) if T1 > 2 else slice(1, 2)
        Pi = P[i, tp].mean(0)
        onc.append(float(np.mean([Pi[c] for c in wcross[i]])))
        offc.append(float(np.mean([Pi[c] for c in oth])))
    res["G_n"] = len(onc)
    res["G_crossed"] = round(float(np.mean(onc)), 5) if onc else None
    res["G_other"] = round(float(np.mean(offc)), 5) if offc else None
    res["G_winfrac"] = round(float(np.mean(np.array(onc) > np.array(offc))), 3) if onc else None

    print("TIMING " + json.dumps(res), flush=True)
    if args.json:
        with open(args.json, "w") as f: json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

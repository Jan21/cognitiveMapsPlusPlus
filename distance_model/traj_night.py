"""Night-shift probes on --dumptraj npz dumps (extends traj_switchyard.py).

  M   migration: decoded worker/crate position per pass; P(start cell) should fall and
      P(goal cell) rise across passes if the walk is a walk.
  V1  plate visitation: crate-slot transient mass on the plate cell, press-requiring pairs
      (optimal path puts crate on plate) vs no-press pairs.
  V2  route-following: transient decoded worker mass on the true worker-path cells vs a
      size-matched random set of free off-path cells.
  G   geometry: per-slot straightness (net / gross displacement) and consecutive-step
      direction cosine, worker slot.
  C   calibration: latent cost per true move, per entity (worker / crate / lever).

Usage: python3 traj_night.py <ckpt.pt> <traj.npz> [--json out.json]
"""
import sys, os, json, argparse, collections
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import switchyard as sw
from traj_switchyard import bfs_path, gcell


def path_events(yard, path):
    """Worker path gcells, crate path gcells, plate pressed?, per-entity move counts."""
    wcells, ccells = [], []
    nw = npu = nl = 0; press = False
    for u, v in zip(path, path[1:]):
        if u[2] != v[2]: nl += 1
        elif u[1] != v[1]: npu += 1
        else: nw += 1
    for st in path:
        r, c = yard.cells[st[0]]; wcells.append(r * yard.G + c)
        r, c = yard.cells[st[1]]; ccells.append(r * yard.G + c)
        if yard.cells[st[1]] == yard.plate: press = True
    return wcells, ccells, press, nw, npu, nl


def train_dec(X, Y, tr, dim):
    dec = torch.nn.Linear(dim, 49); opt = torch.optim.Adam(dec.parameters(), 1e-2)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    for _ in range(300):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(dec(Xt[tr]), Yt[tr]).backward(); opt.step()
    return dec


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
    disp = np.linalg.norm(Z[:, 1:] - Z[:, :-1], axis=-1)
    res = {"tag": a.tag, "decodehead": int(a.decodehead), "B": int(B), "T": int(T1 - 1)}

    # ---- BFS events (grouped by shared src), slot assignment
    wpath = [None] * B; cpath = [None] * B; press = np.zeros(B, bool)
    nw = np.zeros(B); npu = np.zeros(B); nl = np.zeros(B); ok = np.zeros(B, bool)
    groups = collections.defaultdict(list)
    for i in range(B):
        groups[(int(MID[i]), tuple(int(x) for x in S1[i]))].append(i)
    for (m, s), idxs in groups.items():
        yard = yards[m]
        for i in idxs:
            p = bfs_path(yard, s, tuple(int(x) for x in S2[i]))
            if p is None or len(p) - 1 != int(dt[i]): continue
            wpath[i], cpath[i], press[i], nw[i], npu[i], nl[i] = path_events(yard, p)
            ok[i] = True
    w_slot = np.zeros(B, int); c_slot = np.zeros(B, int)
    wc0 = np.zeros(B, int); wcT = np.zeros(B, int); cc0 = np.zeros(B, int); ccT = np.zeros(B, int)
    platec = np.zeros(B, int)
    for i in range(B):
        yard = yards[int(MID[i])]
        wc0[i] = gcell(yard, S1[i][0]); wcT[i] = gcell(yard, S2[i][0])
        cc0[i] = gcell(yard, S1[i][1]); ccT[i] = gcell(yard, S2[i][1])
        r, c = yard.plate; platec[i] = r * yard.G + c
        w_slot[i] = int(att[i, :, wc0[i]].argmax()); c_slot[i] = int(att[i, :, cc0[i]].argmax())
    sel = ok & (w_slot != c_slot)
    res["ok_frac"] = round(float(ok.mean()), 3)

    # ---- decoders on t=0 (70% of maps)
    maps = np.array([int(m) for m in MID]); um = np.unique(maps)
    tr_m = set(um[: int(0.7 * len(um))]); tr = np.array([m in tr_m for m in maps]); te = ~tr
    Zw = Z[np.arange(B), :, w_slot]; Zc = Z[np.arange(B), :, c_slot]
    decw = train_dec(Zw[:, 0], wc0, tr, dim); decc = train_dec(Zc[:, 0], cc0, tr, dim)
    with torch.no_grad():
        Pw = torch.softmax(decw(torch.tensor(Zw.reshape(-1, dim))), 1).reshape(B, T1, 49).numpy()
        Pc = torch.softmax(decc(torch.tensor(Zc.reshape(-1, dim))), 1).reshape(B, T1, 49).numpy()
    res["decw_acc_t0"] = round(float((Pw[te, 0].argmax(1) == wc0[te]).mean()), 3)
    res["decc_acc_t0"] = round(float((Pc[te, 0].argmax(1) == cc0[te]).mean()), 3)

    # ---- M: migration start -> goal across passes (held-out maps, moving entities only)
    mv = te & sel & (wc0 != wcT)
    res["M_worker_startmass"] = [round(float(Pw[mv, t, wc0[mv]].mean()), 3) for t in range(T1)]
    res["M_worker_goalmass"] = [round(float(Pw[mv, t, wcT[mv]].mean()), 3) for t in range(T1)]
    mvc = te & sel & (cc0 != ccT)
    res["M_crate_startmass"] = [round(float(Pc[mvc, t, cc0[mvc]].mean()), 3) for t in range(T1)]
    res["M_crate_goalmass"] = [round(float(Pc[mvc, t, ccT[mvc]].mean()), 3) for t in range(T1)]
    res["M_n_worker"] = int(mv.sum()); res["M_n_crate"] = int(mvc.sum())

    # ---- V1: plate visitation (transient crate-slot mass on plate cell; exclude plate endpoints)
    tp = slice(1, T1 - 1) if T1 > 2 else slice(1, 2)
    pm = Pc[:, tp, :].mean(1)[np.arange(B), platec]
    v = te & sel & (cc0 != platec) & (ccT != platec)
    pres, nofree = v & press, v & ~press & (npu > 0)
    res["V1_platemass_press"] = round(float(pm[pres].mean()), 4) if pres.any() else None
    res["V1_platemass_nopress"] = round(float(pm[nofree].mean()), 4) if nofree.any() else None
    res["V1_n"] = [int(pres.sum()), int(nofree.sum())]

    # ---- V2: route-following (worker transient mass on true path cells vs matched random cells)
    rng = np.random.default_rng(0)
    onp, offp = [], []
    Pwt = Pw[:, tp, :].mean(1)
    for i in np.where(te & sel & (nw >= 3))[0]:
        yard = yards[int(MID[i])]
        mid = [c for c in set(wpath[i]) if c not in (wc0[i], wcT[i])]
        if len(mid) < 2: continue
        free = [r * yard.G + c for (r, c) in yard.cells]
        offs = [c for c in free if c not in wpath[i]]
        if len(offs) < len(mid): continue
        pick = rng.choice(len(offs), len(mid), replace=False)
        onp.append(float(np.mean([Pwt[i, c] for c in mid])))
        offp.append(float(np.mean([Pwt[i, offs[j]] for j in pick])))
    res["V2_n"] = len(onp)
    res["V2_onpath"] = round(float(np.mean(onp)), 5) if onp else None
    res["V2_offpath"] = round(float(np.mean(offp)), 5) if offp else None
    res["V2_winfrac"] = round(float(np.mean(np.array(onp) > np.array(offp))), 3) if onp else None

    # ---- G: geometry (straightness = net/gross per slot; direction cosine of worker slot)
    net = np.linalg.norm(Z[:, -1] - Z[:, 0], axis=-1)               # (B, K)
    gross = disp.sum(1) + 1e-9
    st = net / gross
    res["G_straight_worker"] = round(float(st[np.arange(B), w_slot][sel].mean()), 3)
    res["G_straight_crate"] = round(float(st[np.arange(B), c_slot][sel].mean()), 3)
    res["G_straight_all"] = round(float(st[sel].mean()), 3)
    dz = Z[:, 1:] - Z[:, :-1]
    dzw = dz[np.arange(B), :, w_slot]
    cosn = []
    for t in range(T1 - 2):
        u, v = dzw[:, t], dzw[:, t + 1]
        cosn.append((u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1) + 1e-9))
    res["G_dircos_worker"] = [round(float(c[sel].mean()), 3) for c in cosn]

    # ---- C: latent cost per true move, per entity (median of per-pair cost/move, moves>0)
    cost_k = disp.sum(1); scale = float(d["scale"])
    cw = cost_k[np.arange(B), w_slot] * scale; cc2 = cost_k[np.arange(B), c_slot] * scale
    lever_mask = np.zeros((B, K), bool)
    for i in range(B):
        yard = yards[int(MID[i])]
        for (r, c) in yard.levers:
            lever_mask[i, int(att[i, :, r * yard.G + c].argmax())] = True
        lever_mask[i, w_slot[i]] = False; lever_mask[i, c_slot[i]] = False
    cl = (cost_k * lever_mask).sum(1) * scale
    for nm, cv, tv in (("worker", cw, nw), ("crate", cc2, npu), ("lever", cl, nl)):
        m = sel & (tv > 0)
        res[f"C_cost_per_{nm}_move"] = round(float(np.median(cv[m] / tv[m])), 3) if m.any() else None

    print("NIGHT " + json.dumps(res), flush=True)
    if args.json:
        with open(args.json, "w") as f: json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

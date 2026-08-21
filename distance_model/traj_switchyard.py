"""Interpretability analysis of recall-flow integrator checkpoints on Switchyard.

Consumes a --dumptraj npz plus its --save checkpoint (for args), rebuilds the yards
deterministically, and tests three claims (switchyard analogue of path_integration/traj_analysis):

  P1  walk-tracks-distance: corr(sum-of-slot-displacements, d_true); the accumulation model
      is supervised through the walk, the decode-head trunk is not, so its number is the finding.
  P2  per-slot cost vs per-entity moves on a BFS-optimal path: worker slot cost ~ worker moves,
      crate slot ~ pushes, lever slots ~ pulls; residual corr (control d_true) separates a real
      decomposition from the entangled null where every slot tracks the total.
  P3  enabling-state visitation: linear-decode the worker slot per pass to a grid cell; on pairs
      whose task REQUIRES lever pulls (no-pull BFS distance > d_true) the transient should put
      more mass on lever cells than on distance-matched pull-free pairs.

Usage: python3 traj_switchyard.py <ckpt.pt> <traj.npz> [--json out.json]
"""
import sys, os, json, argparse, collections
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import switchyard as sw


def bfs_path(yard, s, g):
    par = {s: None}; dq = collections.deque([s])
    while dq:
        u = dq.popleft()
        if u == g: break
        for v in yard.neighbours(u):
            if v not in par: par[v] = u; dq.append(v)
    if g not in par: return None
    path = [g]
    while par[path[-1]] is not None: path.append(par[path[-1]])
    return path[::-1]


def nopull_dist(yard, s, g):
    """BFS distance in the lever-free graph (pulls removed); None if unreachable."""
    dist = {s: 0}; dq = collections.deque([s])
    while dq:
        u = dq.popleft()
        if u == g: return dist[u]
        for v in yard.neighbours(u):
            if v[2] != u[2]: continue                              # drop lever edges
            if v not in dist: dist[v] = dist[u] + 1; dq.append(v)
    return None


def decompose(yard, path):
    """(worker moves, pushes, pulls, pulled lever grid-cells) along one optimal path."""
    nw = npu = nl = 0; pulled = set()
    for u, v in zip(path, path[1:]):
        if u[2] != v[2]:
            nl += 1; r, c = yard.cells[u[0]]; pulled.add(r * yard.G + c)
        elif u[1] != v[1]:
            npu += 1
        else:
            nw += 1
    return nw, npu, nl, pulled


def gcell(yard, cellid):
    r, c = yard.cells[int(cellid)]; return r * yard.G + c


def residual_corr(x, y, z):
    """corr(x, y) after regressing z out of both (the entanglement control)."""
    z = np.stack([z, np.ones_like(z)], 1)
    rx = x - z @ np.linalg.lstsq(z, x, rcond=None)[0]
    ry = y - z @ np.linalg.lstsq(z, y, rcond=None)[0]
    if rx.std() < 1e-9 or ry.std() < 1e-9: return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt"); ap.add_argument("npz"); ap.add_argument("--json", default="")
    args = ap.parse_args()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    a = argparse.Namespace(**ck["args"])
    yards, _, _ = sw.make_yards(a)
    d = np.load(args.npz)
    Z = d["Z"].astype(np.float32)                                  # (B, T+1, K, dim)
    att = d["att_s"].astype(np.float32)                            # (B, K, 49)
    S1, S2, MID = d["s1"], d["s2"], d["mapid"]
    dt, dp = d["d_true"].astype(np.float64), d["d_pred"].astype(np.float64)
    B, T1, K, _ = Z.shape
    disp = np.linalg.norm(Z[:, 1:] - Z[:, :-1], axis=-1)           # (B, T, K)
    cost_k = disp.sum(1)                                           # per-slot walk length
    cost = cost_k.sum(1)                                           # total walk length
    res = {"tag": a.tag, "decodehead": int(a.decodehead), "B": int(B), "T": int(T1 - 1),
           "test_corr_ckpt": ck["result"].get("test_corr")}

    # ---- P1: does the walk track the distance? (per-pass amortization profile too)
    res["P1_corr_walk_dtrue"] = round(float(np.corrcoef(cost, dt)[0, 1]), 3)
    res["P1_corr_pred_dtrue"] = round(float(np.corrcoef(dp, dt)[0, 1]), 3)
    res["P1_pass_cost_frac"] = [round(float(x), 3) for x in (disp.sum(2).mean(0) / disp.sum(2).mean(0).sum())]

    # ---- BFS ground truth per pair (grouped by (map, src): eval pool reuses each src ~24x)
    nw = np.zeros(B); npu = np.zeros(B); nl = np.zeros(B); dnp = np.full(B, np.nan)
    pulled = [set()] * B; ok = np.zeros(B, bool)
    groups = collections.defaultdict(list)
    for i in range(B):
        groups[(int(MID[i]), tuple(int(x) for x in S1[i]))].append(i)
    for (m, s), idxs in groups.items():
        yard = yards[m]
        for i in idxs:
            g = tuple(int(x) for x in S2[i])
            p = bfs_path(yard, s, g)
            if p is None or len(p) - 1 != int(dt[i]): continue     # sanity: path length must equal label
            nw[i], npu[i], nl[i], pulled[i] = decompose(yard, p)
            dn = nopull_dist(yard, s, g)
            dnp[i] = np.inf if dn is None else dn
            ok[i] = True
    res["bfs_ok_frac"] = round(float(ok.mean()), 3)
    ok &= np.isfinite(dt)

    # ---- slot assignment per pair: slot with max attention on the entity's grid cell
    w_slot = np.zeros(B, int); c_slot = np.zeros(B, int); wcell = np.zeros(B, int)
    lever_cells = [[] for _ in range(B)]
    for i in range(B):
        yard = yards[int(MID[i])]
        wcell[i] = gcell(yard, S1[i][0])
        w_slot[i] = int(att[i, :, wcell[i]].argmax())
        c_slot[i] = int(att[i, :, gcell(yard, S1[i][1])].argmax())
        lever_cells[i] = [r * yard.G + c for (r, c) in yard.levers]
    ambig = w_slot == c_slot
    res["slot_ambig_frac"] = round(float(ambig.mean()), 3)

    # ---- P2: per-slot cost vs per-entity moves (residual corr controls total distance)
    sel = ok & ~ambig
    cw = cost_k[np.arange(B), w_slot]; cc = cost_k[np.arange(B), c_slot]
    lever_slot_mask = np.zeros((B, K), bool)
    for i in range(B):
        for lc in lever_cells[i]:
            lever_slot_mask[i, int(att[i, :, lc].argmax())] = True
        lever_slot_mask[i, w_slot[i]] = False; lever_slot_mask[i, c_slot[i]] = False
    cl = (cost_k * lever_slot_mask).sum(1)
    for nm, cv, tv in (("worker", cw, nw), ("crate", cc, npu), ("lever", cl, nl)):
        if not sel.any() or np.std(tv[sel]) == 0 or np.std(cv[sel]) == 0:
            res[f"P2_corr_{nm}"] = None; continue
        res[f"P2_corr_{nm}"] = round(float(np.corrcoef(cv[sel], tv[sel])[0, 1]), 3)
        res[f"P2_rescorr_{nm}"] = round(residual_corr(cv[sel], tv[sel].astype(float), dt[sel]), 3)
        res[f"P2_null_corr_{nm}_dtrue"] = round(float(np.corrcoef(cv[sel], dt[sel])[0, 1]), 3)

    # ---- P3: decode worker slot per pass -> grid cell; lever-cell mass, pull-required vs matched pull-free
    Zw = Z[np.arange(B), :, w_slot]                                # (B, T+1, dim)
    maps = np.array([int(m) for m in MID]); um = np.unique(maps)
    tr_m = set(um[: int(0.7 * len(um))]); tr = np.array([m in tr_m for m in maps]); te = ~tr
    dec = torch.nn.Linear(Z.shape[-1], 49)
    opt = torch.optim.Adam(dec.parameters(), 1e-2)
    X = torch.tensor(Zw[:, 0]); Y = torch.tensor(wcell)
    for _ in range(300):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(dec(X[tr]), Y[tr]); loss.backward(); opt.step()
    with torch.no_grad():
        acc = float((dec(X[te]).argmax(1) == Y[te]).float().mean())
        P = torch.softmax(dec(torch.tensor(Zw.reshape(-1, Z.shape[-1]))), 1).reshape(B, T1, 49).numpy()
    res["P3_decoder_acc_t0"] = round(acc, 3)
    lm = np.zeros((B, T1))
    for i in range(B):
        for lc in lever_cells[i]: lm[i] += P[i, :, lc]
    need = sel & te & (dnp > dt)                                   # pulls strictly required (inf = unreachable without pulls)
    free = sel & te & (dnp == dt)                                  # pull-free optimal exists
    trans = lm[:, 1:T1 - 1].mean(1) if T1 > 2 else lm[:, 1]
    bins = np.clip((dt / 4).astype(int), 0, 5)                     # distance-matched comparison
    diffs = []
    for b in np.unique(bins):
        i1 = need & (bins == b); i0 = free & (bins == b)
        if i1.sum() >= 10 and i0.sum() >= 10:
            diffs.append((float(trans[i1].mean()), float(trans[i0].mean()), int(i1.sum()), int(i0.sum())))
    res["P3_n_need"] = int(need.sum()); res["P3_n_free"] = int(free.sum())
    res["P3_levermass_need"] = round(float(trans[need].mean()), 4) if need.any() else None
    res["P3_levermass_free"] = round(float(trans[free].mean()), 4) if free.any() else None
    res["P3_matched_bins"] = [[round(x, 4) if isinstance(x, float) else x for x in t] for t in diffs]
    res["P3_levermass_chance"] = round(float(np.mean([len(l) / 49 for l in lever_cells])), 4)
    # within-pair control: on pairs whose optimal path pulls a strict subset of levers, compare
    # per-cell transient mass on the PULLED lever cells vs the map's other (non-pulled) lever cells
    mp, mo = [], []
    for i in np.where(sel & te & (nl > 0))[0]:
        oth = [c for c in lever_cells[i] if c not in pulled[i]]
        if not pulled[i] or not oth: continue
        tr_mass = P[i, 1:T1 - 1] if T1 > 2 else P[i, 1:2]
        mp.append(float(np.mean([tr_mass[:, c].mean() for c in pulled[i]])))
        mo.append(float(np.mean([tr_mass[:, c].mean() for c in oth])))
    res["P3_within_n"] = len(mp)
    res["P3_within_pulled"] = round(float(np.mean(mp)), 5) if mp else None
    res["P3_within_other"] = round(float(np.mean(mo)), 5) if mo else None
    res["P3_within_winfrac"] = round(float(np.mean(np.array(mp) > np.array(mo))), 3) if mp else None

    print("ANALYSIS " + json.dumps(res), flush=True)
    if args.json:
        with open(args.json, "w") as f: json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()

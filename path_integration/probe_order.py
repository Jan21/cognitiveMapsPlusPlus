"""Deeper probes on a CLEAN recall-flow checkpoint:
H8 topological ORDER: does a parent factor reach the unlock cell (pos 0) BEFORE its child factor starts moving?
    (i.e. is the brief transient a compressed, correctly-ordered simulation of the guarded path?)
H9 calibration: fit c_i = a*path_moves_i + b per factor; is the slope ~1 (cost literally counts moves)?
CPU only. Outputs metrics + a plot.
"""
import argparse, json, os, numpy as np, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import probe

def load_model(ckpt):
    c = torch.load(ckpt, map_location="cpu", weights_only=False)
    a = c["args"]; a = a if isinstance(a, argparse.Namespace) else argparse.Namespace(**a); R = c["R"]
    m = probe.make_model(a, a.n, R)
    if hasattr(m, "recall"): m.recall = bool(getattr(a, "recall", 0)); m.cost = getattr(a, "cost", "length")
    if hasattr(m, "input"): m.input = getattr(a, "input", "index"); m.R = R
    m.load_state_dict(c["state_dict"], strict=False); m.eval(); return m, a, R

def cyc(si, gi, m): d = abs(int(si) - int(gi)); return min(d, m - d)

def path_moves(dist, se, ge, n, R, gt, ms, parents, keyset, guarded):
    if ge not in dist: return None
    cur = ge; per = [0] * n
    while cur != se:
        dc = dist[cur]; nxt = None
        for v in probe.neighbours(cur, n, R, gt, ms, parents, keyset, guarded):
            if dist.get(v, 1e9) == dc - 1: nxt = v; break
        if nxt is None: return None
        pc, pn = probe.dec(cur, n, R), probe.dec(nxt, n, R)
        for i in range(n):
            if pc[i] != pn[i]: per[i] += 1
        cur = nxt
    return per

@torch.no_grad()
def run_flow(model, s, g, Tan):
    n = model.n; zs = model._emb(s, 0); zg = model._emb(g, 1); zr = model._emb(s, 2) if model.recall else None
    base = torch.cat([zg, zr], 1) if model.recall else zg; tok = torch.cat([zs, base], 1); blk = model.blocks[0]
    traj = [tok[:, :n].clone()]; norms = []
    for t in range(Tan):
        z = blk(tok); nz = z[:, :n]; dz = nz - tok[:, :n]
        norms.append((dz.abs().sum(-1) if model.disp == "l1" else dz.norm(dim=-1))); traj.append(nz.clone()); tok = torch.cat([nz, base], 1)
    scale = F.softplus(model.scale).item()
    return torch.stack(traj, 0), torch.stack(norms, 0), scale

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--npairs", type=int, default=1500); ap.add_argument("--Tan", type=int, default=20); ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    model, ma, R = load_model(a.ckpt); n, m = ma.n, ma.m; gt, ms = getattr(ma, "graphtype", "cycle"), ma.m
    keyset = set(range(getattr(ma, "key", 1))); parents = probe.build_guards(n, getattr(ma, "guardmode", "chain"), np.random.default_rng(getattr(ma, "seed", 0) + 99)); guarded = bool(getattr(ma, "guarded", 1))
    tag = os.path.basename(a.ckpt).replace(".pt", ""); rng = np.random.default_rng(a.seed)
    S, Gg, Path = [], [], []; cache = {}; tries = 0
    while len(S) < a.npairs and tries < a.npairs * 20:
        tries += 1; s = tuple(int(rng.integers(m)) for _ in range(n)); se = probe.enc(list(s), R)
        if se not in cache: cache[se] = probe.bfs(se, n, R, gt, ms, parents, keyset, guarded)
        d = cache[se]; ks = [k for k, v in d.items() if v > 0]
        if not ks: continue
        ge = ks[rng.integers(len(ks))]; per = path_moves(d, se, ge, n, R, gt, ms, parents, keyset, guarded)
        if per is None: continue
        S.append(list(s)); Gg.append(probe.dec(ge, n, R)); Path.append(per)
    S, Gg, Path = np.array(S), np.array(Gg), np.array(Path, float)
    traj, sn, scale = run_flow(model, torch.as_tensor(S), torch.as_tensor(Gg), a.Tan)
    c_per = (scale * sn.sum(0)).numpy()
    fid = model.fid.weight.detach(); role0 = model.role(torch.tensor(0)).detach(); posw = model.pos.weight[:m].detach()
    dec = np.stack([torch.cdist((traj[t] - fid[None] - role0).reshape(-1, traj.shape[-1]), posw).argmin(1).reshape(len(S), n).numpy() for t in range(a.Tan + 1)], 0)  # (T+1,B,n)

    M = {"ckpt": os.path.basename(a.ckpt), "n": n}
    # H9 calibration slope per factor
    slopes = []
    for i in range(n):
        x, y = Path[:, i], c_per[:, i]
        if x.std() > 0: slopes.append(round(float(np.polyfit(x, y, 1)[0]), 3))
    M["calib_slope_per_factor"] = slopes; M["calib_slope_mean"] = round(float(np.mean(slopes)), 3)
    # H8 topological order (chain): for child i+1 that MUST move, does parent i reach pos0 before child i+1 first moves?
    def first(cond2d):                                                    # (T,B) bool -> first t (else Tan+9)
        t = np.where(cond2d.any(0), cond2d.argmax(0), a.Tan + 9); return t
    reach0 = {i: first(dec[1:, :, i] == 0) for i in range(n)}             # step child parent reaches unlock
    firstmove = {i: first(dec[1:, :, i] != dec[0, :, i][None]) for i in range(n)}   # step factor i first changes position
    ok, tot, gaps = 0, 0, []
    for i in range(n - 1):                                                # parent i, child i+1
        need = (Path[:, i + 1] > 0)                                       # child must move
        for b in np.where(need)[0]:
            tot += 1; gap = firstmove[i + 1][b] - reach0[i][b]            # >0 => parent unlocked before child moved
            gaps.append(gap)
            if reach0[i][b] < firstmove[i + 1][b]: ok += 1
    M["order_parent_before_child_frac"] = round(ok / tot, 3) if tot else None
    M["order_gap_median"] = float(np.median(gaps)) if gaps else None

    json.dump(M, open(f"{a.out}/order_{tag}.json", "w"), indent=2); print("TAG", tag); print(json.dumps(M, indent=2))
    plt.figure(figsize=(10, 3.6))
    plt.subplot(1, 2, 1)
    for i in range(n): plt.scatter(Path[:, i] + rng.normal(0, .08, len(S)), c_per[:, i], s=5, alpha=.25, label=f"f{i}")
    xs = np.linspace(0, Path.max(), 10); plt.plot(xs, xs, "k--", lw=.8, label="slope 1")
    plt.xlabel("true path moves"); plt.ylabel("c_i"); plt.legend(fontsize=6); plt.title(f"H9 calibration slope~{M['calib_slope_mean']}")
    plt.subplot(1, 2, 2)
    g = np.array(gaps); g = g[np.abs(g) < a.Tan]
    plt.hist(g, bins=range(-8, 9), color="teal"); plt.axvline(0, c="k", ls=":")
    plt.xlabel("child_first_move_step - parent_reach_unlock_step"); plt.ylabel("count")
    plt.title(f"H8 order: parent-before-child {M['order_parent_before_child_frac']}")
    plt.tight_layout(); plt.savefig(f"{a.out}/order_calib_{tag}.png", dpi=115); plt.close()

if __name__ == "__main__":
    main()

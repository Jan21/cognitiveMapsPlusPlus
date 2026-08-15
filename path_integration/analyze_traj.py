"""Trajectory analysis of the recall-flow distance model (v2).

Loads a trained checkpoint, runs the flow with per-factor trajectory logging, and probes the internal process.
Core question: total cost = sum over factors of integrated displacement c_i; does c_i reflect the distance factor i
must actually traverse in the guarded shortest path? Plus: does the flow walk or jump; do transients route through
the unlock cell; is the position embedding metric.

Outputs metrics.json + PNG plots into --out. CPU-only.
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
    m.load_state_dict(c["state_dict"], strict=False); m.eval(); return m, a, R   # strict=False: old ckpts lack the unused image encoder

def cycle_dist(si, gi, m): d = abs(int(si) - int(gi)); return min(d, m - d)

def path_from_bfs(dist, se, ge, n, R, gt, ms, parents, keyset, guarded):
    """count per-factor moves along ONE shortest path start->goal, using a precomputed dist-from-start dict."""
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
    n = model.n; zs = model._emb(s, 0); zg = model._emb(g, 1)
    zr = model._emb(s, 2) if model.recall else None
    base = torch.cat([zg, zr], 1) if model.recall else zg
    tok = torch.cat([zs, base], 1); blk = model.blocks[0]
    traj = [tok[:, :n].clone()]; norms = []
    for t in range(Tan):
        z = blk(tok); nz = z[:, :n]; dz = nz - tok[:, :n]
        per = dz.abs().sum(-1) if model.disp == "l1" else dz.norm(dim=-1)
        norms.append(per); traj.append(nz.clone()); tok = torch.cat([nz, base], 1)
    scale = F.softplus(model.scale).item()
    stepnorm = torch.stack(norms, 0)
    return torch.stack(traj, 0), stepnorm, scale * stepnorm.sum(0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--npairs", type=int, default=1200); ap.add_argument("--Tan", type=int, default=30); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    model, ma, R = load_model(a.ckpt); n, m = ma.n, ma.m; gt, ms = getattr(ma, "graphtype", "cycle"), ma.m
    keyset = set(range(getattr(ma, "key", 1))); parents = probe.build_guards(n, getattr(ma, "guardmode", "chain"), np.random.default_rng(getattr(ma, "seed", 0) + 99)); guarded = bool(getattr(ma, "guarded", 1))
    tag = f"n{n}m{m}D{getattr(ma,'D','?')}_{getattr(ma,'input','index')}_{os.path.basename(a.ckpt).replace('.pt','')}"
    rng = np.random.default_rng(a.seed)

    S, Gg, Cyc, Path, Tot = [], [], [], [], []
    bfs_cache = {}; tries = 0
    while len(S) < a.npairs and tries < a.npairs * 20:
        tries += 1
        s = tuple(int(rng.integers(m)) for _ in range(n)); se = probe.enc(list(s), R)
        if se not in bfs_cache: bfs_cache[se] = probe.bfs(se, n, R, gt, ms, parents, keyset, guarded)
        dist = bfs_cache[se]; keys = [k for k, v in dist.items() if v > 0]
        if not keys: continue
        ge = keys[rng.integers(len(keys))]; g = probe.dec(ge, n, R)
        per = path_from_bfs(dist, se, ge, n, R, gt, ms, parents, keyset, guarded)
        if per is None: continue
        S.append(list(s)); Gg.append(g); Tot.append(dist[ge])
        Cyc.append([cycle_dist(s[i], g[i], m) for i in range(n)]); Path.append(per)
    S, Gg = np.array(S), np.array(Gg); Cyc, Path, Tot = np.array(Cyc, float), np.array(Path, float), np.array(Tot, float)
    traj, stepnorm, c_per = run_flow(model, torch.as_tensor(S), torch.as_tensor(Gg), a.Tan)
    c_per = c_per.numpy(); c_tot = c_per.sum(1); sn = stepnorm.numpy()

    M = {"ckpt": os.path.basename(a.ckpt), "n": n, "m": m, "npairs": len(S)}
    # H1 total + per-factor
    M["corr_total_geodesic"] = round(float(np.corrcoef(c_tot, Tot)[0, 1]), 3); M["mae_total"] = round(float(np.abs(c_tot - Tot).mean()), 3)
    M["corr_cfactor_cycledist"] = round(float(np.corrcoef(c_per.reshape(-1), Cyc.reshape(-1))[0, 1]), 3)
    M["corr_cfactor_pathmoves"] = round(float(np.corrcoef(c_per.reshape(-1), Path.reshape(-1))[0, 1]), 3)
    M["per_factor_corr_vs_pathmoves"] = [round(float(np.corrcoef(c_per[:, i], Path[:, i])[0, 1]), 3) if c_per[:, i].std() > 0 and Path[:, i].std() > 0 else None for i in range(n)]
    M["per_factor_corr_vs_cycledist"] = [round(float(np.corrcoef(c_per[:, i], Cyc[:, i])[0, 1]), 3) if c_per[:, i].std() > 0 and Cyc[:, i].std() > 0 else None for i in range(n)]
    M["per_factor_corr_vs_total"] = [round(float(np.corrcoef(c_per[:, i], Tot)[0, 1]), 3) if c_per[:, i].std() > 0 else None for i in range(n)]   # entangled: c_i tracks TOTAL (even split) not own moves?
    # guard overhead: extra moves beyond own cycle arc
    overhead = Path - Cyc                                                # >=0 for chain
    M["mean_overhead_per_factor"] = [round(float(overhead[:, i].mean()), 2) for i in range(n)]
    M["corr_cfactor_overhead"] = round(float(np.corrcoef(c_per.reshape(-1), overhead.reshape(-1))[0, 1]), 3)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    for i in range(n): plt.scatter(Path[:, i] + rng.normal(0, 0.08, len(S)), c_per[:, i], s=6, alpha=0.3, label=f"f{i}")
    plt.xlabel("true per-factor PATH moves"); plt.ylabel("model per-factor cost c_i"); plt.legend(fontsize=7)
    plt.title(f"c_i vs path-moves  r={M['corr_cfactor_pathmoves']}")
    plt.subplot(1, 3, 2)
    for i in range(n): plt.scatter(Cyc[:, i] + rng.normal(0, 0.08, len(S)), c_per[:, i], s=6, alpha=0.3)
    plt.xlabel("per-factor CYCLE distance"); plt.ylabel("c_i"); plt.title(f"c_i vs cycle-dist  r={M['corr_cfactor_cycledist']}")
    plt.subplot(1, 3, 3)
    plt.scatter(Tot, c_tot, s=6, alpha=0.3, c="tab:green"); lim = [0, max(Tot.max(), c_tot.max())]; plt.plot(lim, lim, "k--", lw=.8)
    plt.xlabel("true geodesic"); plt.ylabel("total cost"); plt.title(f"total r={M['corr_total_geodesic']} MAE={M['mae_total']}")
    plt.tight_layout(); plt.savefig(f"{a.out}/H1_{tag}.png", dpi=110); plt.close()

    # H2 jump vs walk
    cum = np.cumsum(sn.sum(2), 0); frac = (cum / np.clip(cum[-1], 1e-9, None)).mean(1)
    M["steps_to_50pct"] = int(np.searchsorted(frac, 0.5) + 1); M["steps_to_90pct"] = int(np.searchsorted(frac, 0.9) + 1)
    # H4 transient unlock routing: does factor i's token dip to position 0 in the transient, and does that track needing-to-unlock?
    fid = model.fid.weight.detach(); role0 = model.role(torch.tensor(0)).detach(); posw = model.pos.weight[:m].detach()
    def decode_batch(tok_ti):                                            # (B,n,d) -> (B,n) nearest position
        q = tok_ti - fid[None] - role0
        return torch.cdist(q.reshape(-1, q.shape[-1]), posw).argmin(1).reshape(q.shape[0], n).numpy()
    Ttr = min(5, a.Tan)
    dec = np.stack([decode_batch(traj[t]) for t in range(Ttr + 1)], 0)   # (Ttr+1,B,n)
    visitK = np.isin(dec[1:Ttr + 1], list(keyset)).any(0)              # (B,n) factor visited a KEYSET (unlock) position in transient
    # needs_unlock (guard-general): factor i must reach keyset for some CHILD j (i in parents[j]) to move
    child_of = {i: [j for j in range(n) if i in parents[j]] for i in range(n)}
    needs = np.zeros((len(S), n), bool)
    for i in range(n):
        for j in child_of[i]: needs[:, i] |= (Path[:, j] > 0)
    visit0 = visitK
    M["visit0_rate_when_needs_unlock"] = round(float(visit0[needs].mean()), 3) if needs.any() else None
    M["visit0_rate_when_not"] = round(float(visit0[~needs].mean()), 3) if (~needs).any() else None
    plt.figure(figsize=(11, 3.4))
    plt.subplot(1, 3, 1); plt.plot(range(1, a.Tan + 1), np.cumsum(sn.sum(2), 0).mean(1) / cum[-1].mean(), "-o", ms=3)
    plt.axhline(.9, ls=":", c="gray"); plt.xlabel("flow step"); plt.ylabel("cum. displacement frac")
    plt.title(f"jump: 90%@step{M['steps_to_90pct']}")
    plt.subplot(1, 3, 2)
    idx = np.argsort(-Path.max(1))[:1][0]
    for i in range(n):
        dd = [np.argmin(np.linalg.norm((traj[t, idx, i] - fid[i] - role0).numpy() - posw.numpy(), axis=1)) for t in range(a.Tan + 1)]
        plt.plot(range(a.Tan + 1), dd, "-o", ms=2, label=f"f{i}"); plt.hlines(Gg[idx, i], 0, a.Tan, colors="gray", ls=":", lw=.5)
    plt.ylim(-.5, m - .5); plt.xlabel("flow step"); plt.ylabel("decoded pos"); plt.title(f"example pair (legend=factor)"); plt.legend(fontsize=6)
    plt.subplot(1, 3, 3)
    plt.bar([0, 1], [M["visit0_rate_when_needs_unlock"] or 0, M["visit0_rate_when_not"] or 0], color=["tab:red", "tab:blue"])
    plt.xticks([0, 1], ["needs\nunlock", "not"]); plt.ylabel("P(visit pos0 in transient)"); plt.title("H4 unlock routing")
    plt.tight_layout(); plt.savefig(f"{a.out}/H2H4_{tag}.png", dpi=110); plt.close()

    # H5 pos-embedding metricity
    P = model.pos.weight[:m].detach().numpy(); embd = np.linalg.norm(P[:, None] - P[None], axis=-1)
    cyc_gt = np.array([[cycle_dist(i, j, m) for j in range(m)] for i in range(m)], float)
    M["corr_embdist_cycledist"] = round(float(np.corrcoef(embd[np.triu_indices(m, 1)], cyc_gt[np.triu_indices(m, 1)])[0, 1]), 3)

    json.dump(M, open(f"{a.out}/metrics_{tag}.json", "w"), indent=2)
    print("TAG", tag); print(json.dumps(M, indent=2))

if __name__ == "__main__":
    main()

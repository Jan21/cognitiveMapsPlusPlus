"""Compile clean-ness of the per-factor decomposition across analyzed checkpoints, and correlate it with each
checkpoint's LENGTH-EXTRAPOLATION quality (from the saved result.by_d_all). Mechanism -> capability link.
"""
import json, glob, os, numpy as np, torch
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

rows = []
for mj in glob.glob("traj_analysis/*/metrics_*.json"):
    d = json.load(open(mj)); ck = d.get("ckpt")
    if not ck: continue
    pf = d.get("corr_cfactor_pathmoves"); un = d.get("visit0_rate_when_needs_unlock"); un0 = d.get("visit0_rate_when_not")
    ungap = (un - un0) if (un is not None and un0 is not None) else None
    # extrapolation quality from the checkpoint's saved result
    far_mae = flat = D = inp = None
    try:
        c = torch.load(f"ckpts/{ck}", map_location="cpu", weights_only=False)
        res = c.get("result", {}); a = c.get("args"); a = vars(a) if a and not isinstance(a, dict) else (a or {})
        D = a.get("D"); inp = a.get("input")
        bd = res.get("by_d_all")
        if bd:
            items = sorted((int(k), v) for k, v in bd.items())
            far = [v for k, v in items if D and k > D]                 # MAE beyond the training range
            far_mae = float(np.mean(far)) if far else None
            flat = max([k for k, v in items if v < 1.5] or [0])        # flat radius
    except Exception as e:
        pass
    rows.append(dict(ckpt=ck, n=d.get("n"), D=D, input=inp, pf=pf, ungap=round(ungap, 2) if ungap is not None else None,
                     far_mae=round(far_mae, 2) if far_mae is not None else None, flat=flat))

rows.sort(key=lambda r: (-(r["pf"] or 0)))
print(f"{'ckpt':28s} {'n':>2s} {'D':>3s} {'inp':>5s} {'pf_clean':>8s} {'unlock_gap':>10s} {'far_MAE':>8s} {'flat':>5s}")
for r in rows:
    print(f"{r['ckpt']:28s} {str(r['n']):>2s} {str(r['D']):>3s} {str(r['input']):>5s} {str(r['pf']):>8s} {str(r['ungap']):>10s} {str(r['far_mae']):>8s} {str(r['flat']):>5s}")

# correlation: clean-ness vs extrapolation (index only, where far_mae available)
idx = [r for r in rows if r["input"] == "index" and r["pf"] is not None and r["far_mae"] is not None]
if len(idx) >= 4:
    pf = np.array([r["pf"] for r in idx]); fm = np.array([r["far_mae"] for r in idx])
    cc = np.corrcoef(pf, fm)[0, 1]
    print(f"\ncorr(pf_clean, far_MAE) over {len(idx)} index ckpts = {cc:+.3f}  (negative => cleaner extrapolates better)")
    plt.figure(figsize=(5.5, 4.5))
    plt.scatter(pf, fm, s=60, c="tab:purple")
    for r in idx: plt.annotate(r["ckpt"].replace(".pt", "").replace("_Tt", "\nTt")[:16], (r["pf"], r["far_mae"]), fontsize=6)
    plt.xlabel("per-factor clean-ness (corr c_i vs path-moves)"); plt.ylabel("extrapolation error (MAE beyond train D)")
    plt.title(f"mechanism vs capability: r={cc:+.2f}")
    plt.tight_layout(); plt.savefig("traj_analysis/clean_vs_extrap.png", dpi=120); plt.close()
    print("saved traj_analysis/clean_vs_extrap.png")
json.dump(rows, open("traj_analysis/clean_summary.json", "w"), indent=2)

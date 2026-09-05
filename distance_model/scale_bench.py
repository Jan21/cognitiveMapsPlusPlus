"""Scale scaffold for the Switchyard benchmark: run the headline pipeline (all baselines +
DeLPI/DeLPP) at incrementally larger grid sizes, optionally with growing complexity.

Two ladders (edit SCALES freely; S7 is the recorded headline bed and stays the anchor):
  size ladder   S7  S9  S11  S13  S15   - grid grows, complexity fixed (3 gates / 2 levers /
                                          1 chute). Isolates spatial scale; NOTE the coupling
                                          share of distance DILUTES as G grows (more walking
                                          per pull), so this axis alone makes the task easier
                                          in the coupling sense. --envprobe quantifies that
                                          via proxy_corr.
  joint ladder  J9  J11  J13  J15        - grid AND complexity grow (gates -> 4, levers -> 3
                                          then 4, chutes -> 2 then 3), aiming to keep the
                                          factor-coupling pressure roughly constant.

Modes
  --envprobe [names|all]   CPU-only environment probe per scale: state-space size, BFS cost
                           per pool query, distance distribution (diameter, p95/p99), proxy
                           corr (coupling difficulty), projected pool-build time at the
                           headline poolq, recommended Rmax (scaled from the S7 anchor where
                           Rmax=24), required --bfsmax. Writes scale_probe.json. RUN THIS
                           FIRST; it is the go/no-go gate before any GPU is spent.
  --emit [names|all]       write a slurm array file (scale x model x seed x split) using the
                           EXACT tuned model configs of the 683/167k headline bench, with
                           only env geometry, Rmax and bfsmax overridden per scale.
  --report DIR             scan slurm .out files for RESULT lines with sc_* tags, aggregate
                           corr per (scale, model, split) into a table + scale_results.json.

The tuned model flags are copied verbatim from ladder683_ciirc.sbatch (DELPI,
image-based-experiments); do not retune per scale, the point is transfer of the selected
configurations. Training runs go on the cluster (ciirc A40/Volta100), never locally.
"""
import argparse, collections, json, os, re, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import switchyard as sw

# ---------------------------------------------------------------- scale ladders
# Rmax/bfsmax are provisional until --envprobe fills scale_probe.json; emit prefers probed
# values when that file exists.
SCALES = {
    # name  G  ngate nlever nchute
    "S7":  dict(G=7,  ngate=3, nlever=2, nchute=1),   # headline bed (recorded results exist)
    "S9":  dict(G=9,  ngate=3, nlever=2, nchute=1),
    "S11": dict(G=11, ngate=3, nlever=2, nchute=1),
    "S13": dict(G=13, ngate=3, nlever=2, nchute=1),
    "S15": dict(G=15, ngate=3, nlever=2, nchute=1),
    "S17": dict(G=17, ngate=3, nlever=2, nchute=1),
    "S19": dict(G=19, ngate=3, nlever=2, nchute=1),
    "J9":  dict(G=9,  ngate=4, nlever=3, nchute=2),
    "J11": dict(G=11, ngate=4, nlever=3, nchute=2),
    "J13": dict(G=13, ngate=4, nlever=4, nchute=3),
    "J15": dict(G=15, ngate=4, nlever=4, nchute=3),
}
SIZE_LADDER = ["S7", "S9", "S11", "S13", "S15", "S17", "S19"]
JOINT_LADDER = ["J9", "J11", "J13", "J15"]

# ---------------------------------------------------------------- tuned model configs
# Verbatim from the headline 683/167k bench (ladder683_ciirc.sbatch); only env flags differ.
BENCH_BASE = ("--train --enc pureimage --heads 4 --nmaps 683 --poolq 6800 --steps 80000 "
              "--gradclip 1.0 --warmup 2000 --evalevery 8000 --objch 1")
MODELS = {
    "integ":  "--cnnk 1 --cnnw 64 --cnndepth 2 --readout xattn --slots 16 --d 256 --layers 3 --T 4 --lr 1e-3 --nobaseline",
    "dhead":  "--cnnk 1 --cnnw 64 --cnndepth 2 --readout xattn --slots 16 --d 256 --layers 3 --T 1 --lr 5e-4 --nobaseline --decodehead 1",
    "iqe":    "--readout xattn --slots 12 --cnnk 1 --cnnw 64 --basepool mean --baselayers 6 --d 128 --lr 2e-3 --iqeonly",
    "mrn":    "--readout xattn --slots 12 --cnnk 1 --cnnw 64 --basepool mean --baselayers 2 --d 128 --lr 2e-3 --mrnonly",
    "sym":    "--readout pixels --cnnk 1 --cnnw 32 --coordconv 1 --basepool mean --baselayers 0 --d 256 --lr 1e-3 --symonly",
    "scalar": "--readout xattn --slots 12 --cnnk 1 --cnnw 64 --basepool mean --baselayers 2 --d 128 --lr 1e-3 --scalaronly",
}

PROBE_JSON = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scale_probe.json")
ANCHOR_RMAX = 24          # Rmax used at the S7 headline bed
HEADLINE_POOLQ = 6800     # train-pool queries at the headline bed (683 maps / ~167k pairs)


def pick_scales(names):
    if not names or names == ["all"]:
        return list(SCALES)
    if names == ["size"]:
        return list(SIZE_LADDER)
    if names == ["joint"]:
        return list(JOINT_LADDER)
    for n in names:
        if n not in SCALES:
            raise SystemExit(f"unknown scale {n}; known: {', '.join(SCALES)}")
    return names


# ---------------------------------------------------------------- env probe
def env_args(name, seed=0):
    sc = SCALES[name]
    ns = argparse.Namespace(G=sc["G"], ngate=sc["ngate"], nlever=sc["nlever"],
                            nchute=sc["nchute"], seed=seed)
    return ns


def envprobe(names, nmaps, nq, seed):
    """Per scale: build nmaps yards, BFS from nq random sources each, measure everything the
    campaign design needs. proxy_corr is the coupling gauge: it RISING toward 1 with G means
    walking dominates and the interdependence the benchmark exists for is diluting."""
    out = {}
    if os.path.exists(PROBE_JSON):
        with open(PROBE_JSON) as f:
            out = json.load(f)
    anchor_diam = None
    for name in names:
        sc = SCALES[name]
        rng = np.random.default_rng(seed)
        cells = []
        space = []
        dists_all = []
        diams = []
        reach = []
        tsec = []
        tvals, pvals = [], []
        for m in range(nmaps):
            yard = sw.Yard(sc["G"], sc["ngate"], sc["nlever"], sc["nchute"],
                           np.random.default_rng(seed + m))
            cells.append(len(yard.cells))
            space.append(len(yard.cells) * (len(yard.cells) - 1) * (1 << yard.D))
            for _ in range(nq):
                src = yard.rand_state(rng)
                t0 = time.perf_counter()
                dist = yard.bfs(src, maxnodes=10 ** 7)           # uncapped: measure the true set
                tsec.append(time.perf_counter() - t0)
                dv = np.fromiter(dist.values(), dtype=np.int64)
                dists_all.append(dv)
                diams.append(int(dv.max()))
                reach.append(len(dist) / space[-1])
                items = list(dist.items())
                for tv, d_ in [items[i] for i in rng.permutation(len(items))[:60]]:
                    if d_ == 0:
                        continue
                    p = sw.proxy_dist(yard, src, tv)
                    if p is not None:
                        tvals.append(d_)
                        pvals.append(p)
        dv = np.concatenate(dists_all)
        row = dict(
            G=sc["G"], ngate=sc["ngate"], nlever=sc["nlever"], nchute=sc["nchute"],
            cells=int(np.mean(cells)), statespace=int(np.mean(space)),
            reachable_frac=round(float(np.mean(reach)), 3),
            reachable_states=int(np.mean(space) * np.mean(reach)),
            diam_mean=round(float(np.mean(diams)), 1), diam_max=int(np.max(diams)),
            d_p50=int(np.percentile(dv, 50)), d_p95=int(np.percentile(dv, 95)),
            d_p99=int(np.percentile(dv, 99)),
            proxy_corr=round(float(np.corrcoef(tvals, pvals)[0, 1]), 3) if len(tvals) > 10 else None,
            bfs_sec_per_query=round(float(np.mean(tsec)), 2),
            pool_build_min=round(float(np.mean(tsec)) * HEADLINE_POOLQ / 60, 1),
            token_cost_x=round((sc["G"] / 7) ** 2, 2),
        )
        if name == "S7":
            anchor_diam = row["diam_mean"]
        out[name] = row
        print(f"[{name}] " + json.dumps(row), flush=True)
    # recommended Rmax: scale the S7 anchor ratio (24 / S7 mean diameter) to each scale's
    # mean diameter; bfsmax: reachable set + slack.
    ad = anchor_diam or (out.get("S7") or {}).get("diam_mean")
    for name in names:
        row = out[name]
        row["rec_Rmax"] = (int(round(ANCHOR_RMAX * row["diam_mean"] / ad / 2) * 2)
                           if ad else None)
        row["rec_bfsmax"] = int(row["reachable_states"] * 1.3 + 10000)
    with open(PROBE_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {PROBE_JSON}")
    hdr = f"{'scale':<5} {'G':>3} {'gates':>5} {'levers':>6} {'chutes':>6} {'space':>9} {'diam':>6} {'p99':>4} {'proxy':>6} {'s/query':>8} {'pool min':>8} {'Rmax':>5} {'bfsmax':>8}"
    print("\n" + hdr)
    for name in names:
        r = out[name]
        print(f"{name:<5} {r['G']:>3} {r['ngate']:>5} {r['nlever']:>6} {r['nchute']:>6} "
              f"{r['statespace']:>9} {r['diam_mean']:>6} {r['d_p99']:>4} "
              f"{str(r['proxy_corr']):>6} {r['bfs_sec_per_query']:>8} {r['pool_build_min']:>8} "
              f"{str(r['rec_Rmax']):>5} {r['rec_bfsmax']:>8}")


# ---------------------------------------------------------------- emit slurm array
SBATCH_HEADER = """#!/bin/bash
#SBATCH --job-name={job}
#SBATCH --partition=gpu
#SBATCH --gres=gpu:A40:1
#SBATCH --time={time}
#SBATCH --mem=24G
#SBATCH --cpus-per-task=6
#SBATCH --array=0-{last}%{par}
#SBATCH --output={outdir}/{job}_%a.out
# Switchyard SCALE bench: tuned headline configs transferred unchanged; only env geometry,
# Rmax and bfsmax vary. Generated by scale_bench.py --emit; do not hand-edit the A= block.
cd {workdir}
export PYTORCH_ALLOC_CONF=expandable_segments:True
B="python3 switchyard.py {base}"
A=(
"""


def emit(names, models, seeds, splits, workdir, outdir, job, par, time_limit, fname):
    probed = {}
    if os.path.exists(PROBE_JSON):
        with open(PROBE_JSON) as f:
            probed = json.load(f)
    lines = []
    for seed in seeds:                     # seed-major: low seeds land first
        for name in names:
            sc = SCALES[name]
            p = probed.get(name, {})
            rmax = p.get("rec_Rmax") or ANCHOR_RMAX
            bfsmax = p.get("rec_bfsmax") or 200000
            envf = (f"--G {sc['G']} --ngate {sc['ngate']} --nlever {sc['nlever']} "
                    f"--nchute {sc['nchute']} --Rmax {rmax} --bfsmax {bfsmax}")
            for split in splits:
                for mdl in models:
                    tag = f"sc_{name}{split}_{mdl}_s{seed}"
                    lines.append(f' "$B {MODELS[mdl]} {envf} --split {split} --seed {seed} --tag {tag}"')
    body = SBATCH_HEADER.format(job=job, time=time_limit, last=len(lines) - 1, par=par,
                                outdir=outdir, workdir=workdir, base=BENCH_BASE)
    body += "\n".join(lines) + "\n)\n"
    body += 'echo "CFG=$SLURM_ARRAY_TASK_ID ${A[$SLURM_ARRAY_TASK_ID]}"\n'
    body += 'eval "${A[$SLURM_ARRAY_TASK_ID]}"\n'
    with open(fname, "w") as f:
        f.write(body)
    print(f"wrote {fname}: {len(lines)} jobs "
          f"({len(names)} scales x {len(models)} models x {len(seeds)} seeds x {len(splits)} splits), "
          f"array 0-{len(lines) - 1}%{par}")
    if not probed:
        print("NOTE: scale_probe.json missing; emitted with fallback Rmax/bfsmax. "
              "Run --envprobe first for calibrated caps.")


# ---------------------------------------------------------------- report
def report(dirs):
    rows = []
    for d in dirs:
        for fn in os.listdir(d):
            if not fn.endswith(".out"):
                continue
            got_result = False
            tag_line, best_eval = None, None
            with open(os.path.join(d, fn), errors="replace") as f:
                for line in f:
                    if line.startswith("RESULT "):
                        try:
                            r = json.loads(line[7:])
                        except json.JSONDecodeError:
                            continue
                        tag = r.get("tag", "")
                        m = re.match(r"sc_([A-Z]+\d+)(map|wire)_(\w+)_s(\d+)$", tag)
                        if m:
                            rows.append((m.group(1), m.group(2), m.group(3), int(m.group(4)), r))
                            got_result = True
                    elif " --tag sc_" in line and tag_line is None:
                        mt = re.search(r"--tag (sc_\S+)", line)
                        tag_line = mt.group(1) if mt else None
                    else:
                        me = re.search(r"step \d+ evalmae [0-9.]+ evalcorr ([0-9.]+)", line)
                        if me:
                            best_eval = max(best_eval or 0.0, float(me.group(1)))
            if not got_result and tag_line and best_eval is not None:
                # job killed before RESULT (10h window): fall back to best eval checkpoint
                m = re.match(r"sc_([A-Z]+\d+)(map|wire)_(\w+)_s(\d+)$", tag_line)
                if m:
                    rows.append((m.group(1), m.group(2), m.group(3), int(m.group(4)),
                                 {"tag": tag_line, "corr": best_eval, "partial": True}))
    if not rows:
        print("no sc_* RESULT lines found")
        return
    agg = collections.defaultdict(list)
    for scale, split, mdl, seed, r in rows:
        corr = r.get("corr")                              # partial-fallback rows
        for k in ("integ", "sym", "iqe", "mrn", "scalar"):  # RESULT nests under model family
            if isinstance(r.get(k), dict):
                corr = r[k].get("best_corr", r[k].get("test_corr"))
        if corr is not None:
            agg[(scale, split, mdl)].append(float(corr))
    scales = [s for s in SCALES if any(k[0] == s for k in agg)]
    models = sorted({k[2] for k in agg},
                    key=lambda m: (list(MODELS).index(m) if m in MODELS else 99, m))
    splits = sorted({k[1] for k in agg})
    for split in splits:
        print(f"\n== split={split} (corr mean+-std over seeds) ==")
        print(f"{'model':<8}" + "".join(f"{s:>16}" for s in scales))
        for mdl in models:
            cellstr = []
            for s in scales:
                v = agg.get((s, split, mdl), [])
                cellstr.append(f"{np.mean(v):.3f}+-{np.std(v):.3f}({len(v)})" if v else "-")
            print(f"{mdl:<8}" + "".join(f"{c:>16}" for c in cellstr))
    outj = {f"{s}_{sp}_{m}": dict(mean=round(float(np.mean(v)), 4),
                                  std=round(float(np.std(v)), 4), n=len(v),
                                  seeds=sorted(round(x, 4) for x in v))
            for (s, sp, m), v in agg.items()}
    with open("scale_results.json", "w") as f:
        json.dump(outj, f, indent=2)
    print("\nwrote scale_results.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envprobe", nargs="*", default=None,
                    help="scales to probe (names | all | size | joint)")
    ap.add_argument("--nmaps", type=int, default=5, help="envprobe: maps per scale")
    ap.add_argument("--nq", type=int, default=6, help="envprobe: BFS queries per map")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--emit", nargs="*", default=None,
                    help="scales to emit jobs for (names | all | size | joint)")
    ap.add_argument("--models", nargs="*", default=list(MODELS))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3])
    ap.add_argument("--splits", nargs="*", default=["map"])
    ap.add_argument("--workdir", default="/home/hulajan1/swbench")
    ap.add_argument("--outdir", default="/home/hulajan1/swbench/scale")
    ap.add_argument("--job", default="scale")
    ap.add_argument("--par", type=int, default=10, help="array concurrency %%N")
    ap.add_argument("--time", default="24:00:00")
    ap.add_argument("--fname", default="scale_ladder.sbatch")
    ap.add_argument("--report", nargs="*", default=None, help="dirs with slurm .out files")
    a = ap.parse_args()
    if a.envprobe is not None:
        envprobe(pick_scales(a.envprobe), a.nmaps, a.nq, a.seed)
    if a.emit is not None:
        for m in a.models:
            if m not in MODELS:
                raise SystemExit(f"unknown model {m}; known: {', '.join(MODELS)}")
        emit(pick_scales(a.emit), a.models, a.seeds, a.splits,
             a.workdir, a.outdir, a.job, a.par, a.time, a.fname)
    if a.report is not None:
        report(a.report or ["."])


if __name__ == "__main__":
    main()

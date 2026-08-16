# Length generalization + baseline comparison (2026-08-16)

Recorded so we can replicate later. All runs on the CIIRC A40 cluster.

- **Metric.** `corr_beyond` = correlation of predicted vs true distance on test pairs whose geodesic is
  **beyond** the training range (train on distance <= `Rtrain`, evaluate on `> Rtrain`). Also reported:
  `mae_within` / `mae_beyond` (mean abs error split at `Rtrain`). Higher `corr_beyond` = better length extrapolation.
- **Beds.**
  - *crateworld* (`integ_distance.py`): `nag=2` agents on a `G=14` grid, train dist `<= 8`, test to `25`.
  - *switchyard* (`switchyard.py`): the interdependent crate+levers+gates+plate world.
  - *guard-chain / CoT* (`cot_distance.py`): synthetic controllable bed for the checkpoint chain-of-thought model.

---

## 1. Length generalization on the coupled world (crateworld)

**Best result per model (each with its own best recipe):**

| model | recipe | corr_beyond | seeds |
|-------|--------|:---:|:---:|
| **integ (recall-flow accumulator)** | bellman + bellwarm + ematarget + **bellrad** | **0.74** [0.70–0.81] | 3 |
| MRN (torchqmet) | bellman + bellwarm + ematarget (NO bellrad) | 0.64 [0.56–0.72] | 3 |
| IQE (torchqmet) | bellman + bellwarm + ematarget | 0.47 [0.46–0.48] | 2 |

**Key ablation — `bellrad` (radius curriculum) is integ-specific:**

| model | no bellrad | + bellrad | effect |
|-------|:---:|:---:|--------|
| integ | 0.65 | **0.74** | **lifts** |
| MRN | 0.64 | 0.54 | **hurts** |
| IQE | 0.47 | 0.46 | none |

So the accumulator is **not** uniquely liftable by *Bellman* (MRN ties it at ~0.64 there), but it **is**
uniquely liftable by *bellrad*, which is where its ~0.10 length-gen edge comes from. bellrad also tightens
the integ seed spread (0.70–0.81 vs the pre-bellrad 0.63–0.79).

**Other stabilizers tried (integ):** parking (`--arrive 0.2 --Tmin 4`) = 0.62 (stable but lower);
bellrad+parking = 0.60 (combo hurts). `tri` / `bounds` / lower-lr were still running at write time.
Prior finding: `--bellclamp` alone destabilizes (seed 0.20–0.86); `--bellwarm` (ramp weight over first 40%)
is the stable base, resolving the earlier "bellman family spread 2.3–9.4 / EMA-target not implemented" issue.

### Replicate (crateworld)

```bash
BASE="python3 integ_distance.py --enc factored --d 128 --layers 4 --heads 4 --lr 1e-3 --inject 1 \
  --steps 80000 --nag 2 --G 14 --Rmax 25 --Rtrain 8 --maxnodes 400000 --poolq 1200 --nquery 60 --T 24 \
  --bellman 1.0 --bellwarm 1 --ematarget 1"

# integ, best (add bellrad):
$BASE --arch integ --bellrad 1 --seed 0        # -> corr_beyond ~0.74
# MRN, best (NO bellrad):
$BASE --arch mrn   --seed 0                     # -> ~0.64   (adding --bellrad 1 DROPS it to ~0.54)
# IQE:
$BASE --arch iqe   --seed 0                     # -> ~0.47
```
RESULT json field: `all.corr_beyond`. Vary `--seed`.

---

## 2. Length generalization on switchyard: FAILS (the coupling wall)

Every recipe caps at `corr_beyond ~0.36` on switchyard (vs 0.74 on crateworld). Tested: plain, arrive+Ttest,
bellman+ematarget, bellman+bellwarm, across 3 seeds, the full complexity ladder
(gatesopen -> noplate/wire1 -> nopush -> full), and grid sizes G7/G10/G12. Bellman even degrades the
in-distribution fit here (within-MAE 2+ vs 0.4). Conclusion: switchyard's tighter interdependence breaks
length extrapolation for all current readouts.

### Replicate (switchyard length split)
```bash
python3 switchyard.py --train --enc factored --d 128 --layers 4 --T 24 --nmaps 200 --poolq 2000 \
  --Rmax 24 --Rtrain 8 --bellman 1.0 --bellwarm 1 --ematarget 1 --split map     # -> corr_beyond ~0.3
```

---

## 3. Baseline comparison on IMAGE inputs (switchyard, structural generalization)

Same env/data/loss/80k steps for every cell; only the distance head and the image encoder change.
Metric = `test_corr` (predicted vs true distance on the held-out map/wire split). This is NOT length
extrapolation (distances stay in range); it is structural generalization to unseen maps / wirings.

| encoding / axis | **integ** | IQE | MRN | sym |
|---|:--:|:--:|:--:|:--:|
| bmask (lossless canvas) map | **0.91** | 0.76 | 0.75 | 0.73 |
| bmask (lossless canvas) wire | **0.89** | 0.78 | 0.80 | 0.74 |
| marker + aux binding, map | **0.87** | 0.71 | 0.76 | 0.74 |
| marker + aux binding, wire | — | 0.73 | 0.78 | — |
| marker (raw, no aux) map | 0.70 | 0.74 | (nan) | 0.74 |

**Takeaways.** With a working binding (bmask or marker+aux) the **integrator dominates the metric/symmetric
baselines by ~0.10–0.15 on both axes** (0.87–0.91 vs 0.71–0.80). The baselines are binding-robust but
lower-ceiling (~0.74–0.80 regardless of encoding); the accumulator has the higher ceiling but needs clean
binding to reach it (raw marker 0.70 = its binding is the bottleneck, fixed by bmask / aux).

**Marker binding.** `--markeraux` (aux loss decoding the bound token -> true cell) lifts learned marker
binding 0.685 -> 0.88 (map) / 0.91 (wire); weight-insensitive (0.5/1/2 all ~0.88), heads-insensitive (4/8).
This uses privileged cell labels; unsupervised alternatives (`--bindmode gather` value-match, `--bindmode slot`
slot-attention) are under test to remove that supervision.

### Replicate (image comparison)
```bash
C="python3 switchyard.py --train --d 128 --layers 4 --T 14 --nmaps 200 --poolq 2000 --steps 80000"
# integrator (ours):
$C --nobaseline --enc bmask                         --split map   # -> 0.91
$C --nobaseline --enc marker --markeraux 1.0 --markerheads 8 --split map   # -> 0.87
# baselines (swap the head, keep everything identical):
$C --iqeonly --enc bmask --split map     # 0.76      (needs torchqmet)
$C --mrnonly --enc bmask --split map     # 0.75      (needs torchqmet)
$C --symonly --enc bmask --split map     # 0.73
```
RESULT json field: `integ.test_corr` (or `iqe`/`mrn`/`sym`). `--split wire` for the wiring axis.

---

## 4. CoT checkpoint model (overall-length extrapolation): WORKS

On the guard-chain bed, the checkpoint chain-of-thought model **beats the amortized one-shot model on
overall-length extrapolation** (free-run corr 0.93–0.98 vs amortized collapsing to −0.04 as envs get
harder). The apparent "#waypoint wall" (a strict `ntup` split) was a **data-starvation artifact**: on the
natural (geo) split the per-step transition is perfect (`tf_step_acc = 1.00`) at every chain length up to 12
waypoints. Code: `cot_distance.py` (worktree `cot-distance`). See `cot_3h_report.html`.

---

## File / job map (for finding the raw outputs on the cluster)

`~/cognitiveMapsPlusPlus/distance_model/*.out`, grep `^RESULT`:
- crateworld length-gen + bellrad: `integStab_*.out` (integ stabilizers), `iqmFair_*.out` (fair IQE/MRN),
  `iqmBrad_*.out` (IQE/MRN + bellrad), `cwLenSd_*.out` (integ seed confirm), `stab_*.out` (original bellStab).
- switchyard length-gen: `syLen_*`, `syLen2_*`, `syLen3_*`, `syLenLad_*`, `dbg_S0/S1_*` (G10/G12).
- image comparison: `imgBase_*.out`, `mkPush_*.out` (marker aux), `bindMode_*.out` (unsupervised binding).
- CoT: `cotSweep_*`, `cotAmort_*`, `cotDir_*`, `cotDiag_*`, `cotDiagG_*` (in `path_integration` / `distance_model`).

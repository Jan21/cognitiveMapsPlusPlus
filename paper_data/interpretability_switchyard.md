# Interpretability on the image Switchyard checkpoints (2026-08-21)

Switchyard analogue of the path_integration trajectory analysis, run on fresh same-seed
retrainings (A40) of three L5 map-split integrator runs and two decode-head trunks.
Code: `distance_model/traj_switchyard.py` on `--dumptraj` dumps (4096 held-out pairs each).
Raw jsons: `handoff_colleague/analysis/` and DELPI `results/interpretability/`.

Checkpoints (retrained corr): idv_map_s1 0.904 (683/167k), int600_map_s2 0.917,
int600_map_s3 0.956 (600/201k), dh683_lr5e4_s0 0.684, dh683_lr5e4_s1 0.817 (683/167k).
Provenance: originals were never weight-saved; same-arch determinism verified exactly
(codecheck: new code reproduces the old-code A40 run 0.952/1.321 to the last digit; sym
baseline replicated 0.789 exactly), cross-arch retraining drifts by the noise floor.

## P1: the walk tracks the distance (readout contrast)

corr(Sigma-of-slot-displacements, d_true), held-out pairs:

| model | walk~d | pred~d | per-pass cost fraction |
|---|---|---|---|
| integ idv_map_s1 | 0.90 | 0.90 | 0.15 / 0.25 / 0.33 / 0.27 |
| integ int600_map_s2 | 0.92 | 0.92 | 0.14 / 0.27 / 0.31 / 0.28 |
| integ int600_map_s3 | 0.95 | 0.95 | 0.16 / 0.27 / 0.31 / 0.25 |
| decode-head s0 | 0.60 | 0.71 | 0.49 / 0.18 / 0.17 / 0.16 |
| decode-head s1 | 0.67 | 0.82 | 0.39 / 0.23 / 0.20 / 0.18 |

Image analogue of the symbolic 0.95-vs-0.40 result: the decode-head trunk's walk is
unsupervised and only partially tracks distance; supervising through the walk (accumulation)
is what makes walk = distance. The integrator spreads cost over all four passes; the decode
head front-loads.

## P2: per-slot cost vs per-entity moves on a BFS-optimal path

Residual correlation (total distance regressed out of both sides; kills the "every slot
tracks the total" null). Slots assigned per pair by attention on the entity's cell.

| model | worker | crate | lever |
|---|---|---|---|
| integ idv_map_s1 | 0.16 | 0.40 | **0.52** |
| integ int600_map_s2 | -0.04 | 0.26 | **0.47** |
| integ int600_map_s3 | 0.31 | 0.19 | **0.50** |
| decode-head s0 | -0.26 | 0.08 | -0.02 |
| decode-head s1 | -0.43 | 0.39 | 0.02 |

The lever slots (the interdependence carriers) decompose cleanly in every integrator and in
no decode-head trunk. Stronger than expected: the old factored-env image clean-index was 0.24.

## P3: latent paths visit the enabling states (lever visitation)

Linear decoder (worker-slot token t=0 -> grid cell, trained on held-in maps; accuracy below),
transient (passes 1..T-1) decoded mass on lever cells. "Need" = pulls strictly required
(no-pull BFS distance > d_true, incl. unreachable); "free" = a pull-free optimal path exists.
Chance = lever cells / 49 = 0.041.

| model | dec acc | need | free | need>free in ALL distance-matched bins |
|---|---|---|---|---|
| integ idv_map_s1 | 0.55 | 0.066 | 0.036 | yes (5/5) |
| integ int600_map_s2 | 0.19 | 0.034 | 0.011 | yes (5/5) |
| integ int600_map_s3 | **0.72** | **0.079** | 0.037 | yes (5/5), e.g. 0.128 vs 0.032 shortest bin |
| decode-head s0 | 0.13 | 0.086 | 0.049 | yes (weak decoder, interpret cautiously) |
| decode-head s1 | 0.34 | 0.038 | 0.033 | yes |

Within-pair control (mass on the pulled lever vs the map's other levers): win fraction only
0.42-0.60 for the integrators. Same conclusion as the factored-env study: routing is SPATIAL,
not a step-by-step re-enactment; the transient visits the enabling region reliably (2-4x the
distance-matched control) but discriminates WHICH lever only weakly at T=4.

Caveats: decoder is trained unseeded in the analysis script (small run-to-run wobble in acc);
T=4 gives four latent steps, so temporal ordering claims are out of reach by design;
decode-head visitation numbers ride on very weak decoders.

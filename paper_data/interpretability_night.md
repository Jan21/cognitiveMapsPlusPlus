# Night interpretability shift (2026-08-21 19:20 -> 01:20)

Autonomous 6h program on the image Switchyard checkpoints. Daytime baseline results:
`interpretability_switchyard.md`. Artifact page (updated as findings land):
https://claude.ai/code/artifact/26db2624-bbb2-4c30-993a-2299152ea8b3

## Queue (planned probes, ranked)

1. [running 128482] Ttest sweep 1/2/8/16 at inference: does the walk PARK (budget-free
   flat cost, like the old flow) or keep accumulating; Ttest16 dumps extended trajectories.
2. [running 128482] norecall at inference: is goal re-injection load-bearing for the walk.
3. [running 128483] wire-split checkpoint (683/167k, seed 1) + all three probes; wiring is
   pixel-invisible, so lever visitation under resampled wiring is the sharpest routing test.
4. [todo] crate/plate/gate visitation (P3 analogues for the other enabling mechanics),
   pure analysis on existing dumps.
5. [todo] walk geometry: per-slot straightness (net vs gross displacement), where curvature
   lives, decode-head contrast.
6. [todo] slot binding persistence across passes (does the worker slot stay the worker slot).
7. [todo] per-entity cost calibration slopes (moves per unit latent distance; shared scale?).
8. [todo] decoded-trajectory visualizations (grid overlay: decoded worker path vs BFS path).
9. [stretch] --evalpairs extension for counterfactual pair families (bits-only distance).

## Findings

### 23:34 batch 1: the walk does NOT park; recall is load-bearing

Ttest sweep at inference (trained T=4; corr / MAE):

| model | T=1 | T=2 | T=4 (trained) | T=8 | T=16 |
|---|---|---|---|---|---|
| idv_map_s1 | 0.776 / 9.7 | 0.857 / 6.9 | **0.904 / 1.9** | 0.888 / 9.8 | 0.846 / 28.0 |
| int600_map_s3 | 0.810 / 9.6 | 0.914 / 6.5 | **0.956 / 1.2** | 0.949 / 8.0 | 0.922 / 22.4 |

- Cost keeps accumulating roughly linearly past T=4: MAE explodes while corr degrades only
  gently. The image switchyard integrator is NOT budget-free, unlike the factored recall-flow
  (which parked and extrapolated flat). Calibration lives at the trained T; ranking survives.
- Most of the ranking signal exists after 1-2 passes (corr 0.78-0.91) but undershoots in scale.
- norecall at inference: idv 0.814 / 3.16, 600s3 0.758 / 3.85 (vs 0.904 / 1.86, 0.956 / 1.25).
  Goal re-injection is load-bearing but not catastrophic to remove.

Correction: deadline is ~05:00 (6h from the 23:00 request), not 01:20.

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

### 00:10 batch 2+3: migration, route-following, plate negative, geometry, calibration, wire divergence

Wire checkpoint: nw_wire_s1 corr 0.909 / MAE 1.89 (recorded wire seeds ~0.94, cross-arch drift).

- **Migration** (decoded worker position per pass, held-out maps, linear probe trained at t=0):
  startmass decays monotonically (int600_s3: 0.74 -> 0.57 -> 0.33 -> 0.20 -> 0.14; idv: 0.55 -> 0.17),
  goalmass rises weakly (s3: 0.007 -> 0.033, chance ~0.02). The walk demonstrably LEAVES the start;
  arrival at the goal is weak in decoded space. Decode heads: start info collapses after one pass
  (s0: 0.10 -> 0.00). Caveat: decoder is trained on t=0 tokens, later passes are OOD for it.
- **Route-following (V2)**: integ decoded transient mass on true worker-path cells vs matched
  random free cells: 0.027-0.049 vs 0.011-0.017, winfrac 0.61-0.73. Decode head s0: winfrac 0.20
  (anti), s1 0.63. The integ walk follows the actual route, not just the endpoints.
- **Plate visitation (V1): NEGATIVE.** Crate-slot transient mass on the plate cell is ~0 on
  press-requiring pairs (0.000-0.002) vs 0.004-0.038 on no-press pairs; n_press only 11-26.
  Enabling-state visitation is a worker/lever phenomenon, not crate/plate (or undetectable at
  these sample sizes and crate-decoder quality).
- **Geometry**: integ walks CURVE (net/gross 0.75-0.79); decode-head walks are straighter jumps
  (0.86-0.89). Direction cosine rises across passes (0.4 -> 0.75): turning early, straight later.
  At Ttest=16 straightness drops to 0.66-0.69 and dircos saturates ~0.98: the extrapolated passes
  keep marching in a fixed direction, matching the runaway MAE.
- **Calibration (cost per true move)**: worker 0.081-0.085 latent units per move, consistent
  across all three integ models; crate ~0.45; lever pull 0.79-1.43. Coupling-carrying moves are
  charged 5-17x the worker move. Image analogue of "internal factors charged extra unlock moves".
- **Wire-split divergence (single seed, preliminary)**: nw_wire_s1 keeps walk=distance
  (P1 0.907) but LOSES the decomposition (P2 residual lever 0.135 vs ~0.5 map-split) and lever
  visitation REVERSES (free > need in all matched bins; within-pair 0.50). Hypothesis: wiring is
  pixel-invisible and resampled at test, so the wire model cannot know which lever matters and
  stops routing; map-split models on novel layouts use generic route-through-levers behavior.
  Cross-eval (batch 4, running): wire ckpt on unseen-maps pool, map ckpt on rewired pool.

### 00:45 batch 4: cross-eval = clean DOUBLE DISSOCIATION

Accuracy transfers both ways (wire ckpt on unseen maps 0.916, map ckpt on rewired pool 0.910;
home scores 0.909 / 0.904). Mechanisms dissociate:

|  | map ckpt | wire ckpt |
|---|---|---|
| P2 lever residual, unseen-maps pool | 0.52 (home) | 0.20 |
| P2 lever residual, rewired pool | **0.54** | 0.135 (home) |
| P3 lever visitation, unseen-maps pool | 0.066 vs 0.036, 5/5 bins (home) | **0.073 vs 0.057, 5/5 bins** |
| P3 lever visitation, rewired pool | **gone** (0.020 vs 0.017, mixed bins) | reversed (home) |

- **Decomposition is a MODEL property**: the map-trained model keeps lever residual ~0.5 on any
  pool; the wire-trained model lacks it on any pool.
- **Lever visitation is a SPLIT property**: BOTH models route through levers on unseen layouts,
  BOTH lose it on rewired-familiar layouts.
- Hypothesis update: on familiar layouts the walk may visit the levers that were correct under
  the TRAINING wiring (stale memory) rather than the resampled ground truth. Stale-wiring probe
  (batch 5) tests decoded mass on stale-only vs current-only lever cells directly.
- Also running: decoded-walk visualization extraction (batch 6, paper figure data).

### 01:10 batch 5: STALE-WIRING PROBE CONFIRMS MEMORY-DRIVEN ROUTING

On the rewired pool, restricted to pairs where training-wiring and current-wiring optimal paths
pull DISJOINT levers (n=45 strict pairs per model):

| model | mass on stale-only levers | mass on currently-correct levers | ratio |
|---|---|---|---|
| map ckpt (idv_map_s1) on rewired pool | 0.0368 | 0.0020 | 19x |
| wire ckpt (nw_wire_s1) home pool | 0.0588 | 0.0014 | 41x |

The walk visits the levers that were correct under the TRAINING wiring of that layout, not the
resampled ground truth, while the distance OUTPUT stays accurate (0.91). So the routing is a
memory re-enactment, and the visitation reversal on rewired pools is explained: the metric was
keyed to current-truth pulls. Distance prediction evidently does not depend on visiting the
right levers.

Caveats: n=45 (strict disjointness filter); win fraction 0.42/0.47 (means driven by strong
cases, many pairs put ~0 on both); wire-model decoder acc only 0.264; single seed per model.
Second wire seed (nw_wire_s2) launched 02:15 to confirm.

### 02:10 batch 7: gate visitation strong, lever timing not temporal

- **Gate visitation (within-pair, crossed vs uncrossed gates)**: int600_s3 (decoder 0.755):
  0.056 vs 0.002 per cell, 27x, winfrac 0.618, n=636. idv (decoder 0.51): 1.9x, winfrac 0.54.
  T16 dumps weaker/mixed (decoders degrade). Gates get the within-pair discrimination that
  levers lack: gates lie ON the route, so route-following buys gate identity for free; lever
  identity would need plan knowledge.
- **Lever-visit timing: no consistent temporal signature.** s3 need-curve rises to a mid/late
  peak (0.006 -> 0.088 at t=3), idv starts high and decays, T16 flat-smeared (need>free across
  all 16 passes). Real plans pull levers EARLY; the decoded visit is not temporally ordered.
  Confirms the spatial-not-temporal reading at a finer grain.

### 03:30 batch 8: wire seed 2 (0.957) BREAKS the overgeneralized claims

nw_wire_s2: corr 0.957 / MAE 1.25 (vs seed 1 0.909). On this stronger wire model:

- P2 lever residual **0.577**: the decomposition IS there. "Wire-trained models lack the
  decomposition" was a seed-1 artifact; decomposition is seed/quality-dependent, not
  split-determined (present in 4 of 5 integ ckpts, absent only in the weakest, wire s1).
- P3 on its home rewired pool: need 0.0255 vs free 0.0119 (2.1x), within-pair 0.589: it routes
  by the CURRENT wiring even on rewired-familiar layouts.
- Stale probe: mass ~0 on BOTH stale-only and current-only levers (n=28 strict pairs): the
  19-41x stale re-enactment does NOT replicate. Memory-driven routing is a trait of the weaker
  models (idv 0.904 on rewired pool; wire s1 0.909), not a universal mechanism.

Revised honest picture: what is robust across ALL five integrators = walk=distance (0.90-0.96
vs decode-head 0.60-0.67), non-parking accumulation, route-following, gate discrimination,
lever visitation on unseen layouts, plate negative, non-temporal timing, per-move calibration.
What is seed-dependent = behavior under rewired causality: strong models route by current
truth; weaker ones either lose the decomposition or re-enact the stale plan. The cross-eval
"double dissociation" held for the seed-1 pair only. More seeds would be needed to make any
split-level claim; the quality-dependence itself is the finding.

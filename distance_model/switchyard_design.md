# Switchyard: a grid benchmark with interdependent factors

Design document, 2026-08-13. Code: `switchyard.py` (env + BFS ground truth + probe + smoke training, self-contained).

## The game

A railway switchyard. A **worker** pushes a **crate** around a four-room yard. Room passages are **gates** that open and close; wall-mounted **levers** each toggle a wired subset of gates; a floor **pressure plate** forces its wired gates open while the crate sits on it; some corridor cells are one-way **chutes** the worker can only enter from one side. The task: reach a target configuration (worker position, crate position, gate states) in the fewest moves.

The game reads instantly (Sokoban meets Zelda switches), and every element is lifted from a published benchmark; the novelty is only that they are wired into one dependency web.

## Element provenance (deliberate)

| Element | Borrowed from | Dependency it creates |
|---|---|---|
| Gates (open/closed bits) | MiniGrid DoorKey; MAD KeyDoorGridWorld | gate bits gate BOTH worker and crate movement |
| Levers with XOR wiring | OGBench Puzzle (Lights-Out core) | one pull flips several gates at once |
| Pushable crate | DeepNorm `push`; Sokoban | crate moves only by worker contact; pushes can be irreversible (directed graph, asymmetric distances) |
| Pressure plate | MiniGrid ObstructedMaze family | crate POSITION overrides gate bits (factor-on-factor gating) |
| One-way chutes | PQE/IQE one-way-doors gridworld | asymmetric worker movement |

## Why distances are interdependent here

The state is factored as `[worker_cell, crate_cell, gate_bits]`, but the geodesic is a joint object:

- To open a gate you may need a lever behind another gate, which may need the crate on the plate, which needs pushing the crate through a third gate: dependency chains of depth 2-3 arise naturally from random wiring.
- The crate is both cargo and tool: sometimes the shortest route parks the crate on the plate purely to hold gates open for the worker, then retrieves it.
- Lever wiring is XOR over gate subsets, so gate bits interact combinatorially (Lights-Out), and the plate overrides them, so bit distance and crate distance do not add.
- Pushes into corners are irreversible: the graph is directed, some goals unreachable, distances asymmetric.

## Configuration space (the generalization lever)

Per map: wall-gap placement, which gaps are gates, lever positions and wiring masks, plate wiring mask, chute directions. The wiring is the analogue of our coupling world's `(mobility_key, link_key)`: held-out wirings and held-out maps give configuration-generalization splits; `--Rmax` capping gives the length split. `Yard.cfg_key()` identifies a configuration for split assignment (current smoke split: map index mod 4).

## Ground truth and scale (probe, iteration 1)

Defaults G=7, 3 gates, 2 levers, 1 chute, 1 plate: joint state space ~12.5K states per map, BFS exact and instant; mean sampled distance ~22, diameter ~30. Reachable fraction from a random source ~8%: low by design (irreversible pushes + parity-unreachable gate bits); pair sampling draws from the reachable set, and directedness is a feature (asymmetry), not a bug.

## The factorization gap (the property the benchmark exists for)

Probe compares true BFS distance against the best factorized proxy: independent worker walk (gates open, crate ghosted) + independent crate distance + minimal lever pulls (BFS over bits), summed.

- Iteration 1 (full env): proxy correlation **0.58**, proxy MAE **13.6** at mean distance 22.4; gap of at least 3 moves on **84%** of pairs, at least 6 on **74%**. Rewiring alone (same endpoints, new lever wiring) shifts distances with std **11.3**.
- Gates+levers, no chute: corr 0.45: the gap is carried by the gate/lever/plate coupling, not the chute.
- Pure-Sokoban control (no gates/levers/plate effects): run pending; expected to show the residual gap attributable to pushing alone.

Interpretation: factorized structure explains only about a third of the distance variance; any method that scores factors independently (or embeds them into a static per-factor geometry) has a built-in error floor of many moves. This is the regime our joint-flow readout is built for.

## Evaluation protocol (mirrors integ_distance.py)

Supervised pairs (state, goal, BFS distance) bucketed per distance; smooth L1; metrics MAE and correlation on held-out configurations; optional `Rmax` cap for length extrapolation. Baselines plug in exactly as in the coupling-gridworld experiments (shared encoder + IQE / MRNFixed / scalar head / symmetric norm).

## Open design knobs (for iteration)

- Reachability: raise G or soften irreversibility (allow pulls?) if 8% reachable proves too tight for pair diversity.
- Dependency depth control: number of gates/levers and wiring density tune coupling strength; could expose an explicit "dependency depth" parameter for a difficulty ladder.
- Image mode: render worker/crate/gates/levers/plate on a shared canvas (same bmask/marker path as integ_distance.py) later.

## Status log

- Iter 1: env implemented; probe confirms large factorization gap and strong wiring sensitivity (proxy corr 0.58, gap >= 3 on 84% of pairs, rewiring std 11.3). ngate=0 crash fixed.
- Iter 2 (dgx): pure-Sokoban control shows pushing alone carries much of the proxy gap (corr 0.66), so the cleaner interdependence metric is WIRING SENSITIVITY (distance shift under rewiring at fixed endpoints: 4.6-11.3 std; exactly 0 for pure Sokoban). Gates/levers deepen the gap further (full corr 0.42-0.46). Reachability at 16 maps fell to 3.5%: too tight. Smoke train (12 maps, toy model, 6k steps): integrator transfers to held-out maps better than a scalar head (test MAE 4.24/corr 0.70 vs 4.51/0.59, mean distance ~21); both under-trained, direction correct.
- Iter 3: connectivity fix (extra always-open wall gaps; reachable fraction 3.5% -> 17%, diameter 40, coupling intact: proxy corr 0.47, gap >= 3 on 88% of pairs, wiring std 4.3). Full-scale run (dgx, 40 maps, d=128, T=14, 20k steps, held-out-map split): **integrator test MAE 3.47 / corr 0.74 vs scalar head 4.41 / 0.62** (mean distance ~24; train MAE 0.49 vs 0.65). The integrator's lead over the scalar head GREW with scale (0.27 MAE at iter-2 toy size -> 0.94 now), i.e. the benchmark separates methods in the direction the interdependence thesis predicts, while held-out-map error remains high enough that the benchmark is far from saturated.
- Next knobs if needed: more maps (layout diversity is the generalization bottleneck at 12), a wiring-only split (same layouts, unseen wirings) to mirror the coupling world's combo split, image rendering mode, IQE/MRN baselines via torchqmet.

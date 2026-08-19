# Environment: Switchyard and the interdependence ladder

Code: `distance_model/switchyard.py` (class `Yard`, `make_yards`, `build_pool`). Playable examples of every rung with
BFS-optimal solutions: artifact "Switchyard Ladder Playbook" (`distance_model/switchyard_ladder_playbook.html`).

## The world
A 7×7 grid with a fixed wall cross (row 3, column 3) making four rooms. Per map, the generator opens one gap per wall
arm (4 gaps) and, with probability 0.5 per arm, one extra gap (connectivity); of the 4 arm gaps, `ngate` (3) become
**gates**, the remaining opened cells are permanent doorways (mean 2.5 per map). A **worker** moves N/S/W/E; a **crate**
moves only when pushed by the worker (needs a free cell behind it; pushes can be irreversible). Gates are open/closed
bits; both worker and crate need an open gate to pass. `nlever` (2) **levers**: pulling lever l XORs its wiring mask
(a non-zero subset of gates; `--wire1` restricts to one distinct gate per lever) into the gate bits. A **pressure
plate** holds the gates in its mask open while the crate sits on it. A one-way **chute** cell admits the worker only
from one direction. State = (worker cell, crate cell, gate bits); configuration = layout + gate positions + lever
wiring + plate wiring + chute direction. Labels = exact BFS distance on the joint state graph (directed).

## Observation (image-only setting)
12 binary 7×7 planes: walls · worker · crate · gate cells · gate-open bits · per lever: lever cell + the gate cells it
toggles (wiring drawn as shared colour) · plate cell + the gate cells it holds · chute cell in one of four direction
planes. Optional 13th plane: objectness (1 at any cell lit by an entity plane). No coordinates, identities or wiring
tables are given symbolically.

## The ladder (one mechanic added per rung)
| rung | adds | flags | reachable states / map (mean, 20 maps) | eccentricity of a random state |
|---|---|---|---|---|
| L0 | plain maze, crate static | `--gatesopen --nopush` | ≈ 38 | ≈ 10 |
| L1 | pushable crate | `--gatesopen` | ≈ 870 | ≈ 30 |
| L2 | gates; one lever = one gate | `--wire1 --noplate` | ≈ 2 600 | ≈ 36 |
| L3 | XOR multi-gate wiring | `--noplate` | ≈ 2 570 | ≈ 37 |
| L4 | pressure plate | `--nchute 0` | ≈ 3 050 | ≈ 37 |
| L5 | one-way chute (full) | (defaults) | ≈ 2 580 | ≈ 37 |
| L6 | 4 gates / 3 levers, dense wiring | `--ngate 4 --nlever 3` | ≈ 4 950 | ≈ 47 |
Coupling probes (full env, `switchyard_design.md`): best factorised proxy distance correlates only 0.45–0.58 with the
true distance; rewiring alone (same endpoints) shifts distances with std 4–11 moves.

## Data and splits
200 maps per run (`--nmaps 200`), map m from rng `seed+m`, wiring from rng `50000+seed+m`.
`--split map`: train on maps with m % 4 ≠ 0 (150), test on m % 4 = 0 (50): unseen layouts **and** unseen wiring.
`--split wire`: the 200 layouts appear in both train and test; test wiring is resampled (`wire_rng` offset 90000).
Pairs: `--poolq 2000` random anchor states on random training maps; BFS from each anchor; up to `40 // #distinct
distances` targets per distance value, distances 1..24 (`--Rmax 24`): ≈ 50–80 k distance-stratified training pairs.
Test pool built the same way (rng `seed+99`) on the test maps. Seeds change maps, pairs and initialisation together.

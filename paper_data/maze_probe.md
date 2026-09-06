# Maze distance probe (2026-09-06, ciirc 132794/132812/132840)

Bed: perfect DFS mazes (light braiding), inputs = layout + start id + goal id; start
and goal token embeddings evolved by a weight-tied 2-block transformer T times with
attention to the layout; distance = softplus scale x L2 between the final two
embeddings (fullinteg: accumulated token motion instead); smooth-L1 on BFS shortest
path; validation on UNSEEN layouts. Path lengths: G11 mean 17 / max 48; G15 mean 30 /
max 86. Script: distance_model/maze_probe.py.

## Round 1, G15 T8 (best_corr, 2 seeds)
full .508/.493 > frozen .489/.481 > fullinteg .467/.469 > frozenenc .430/.453.
All stuck ~.5. (frozen = maze tokens fixed, only start/goal evolve via cross-attn;
full = everything evolves; frozenenc = one-shot static encoder, then frozen - WORST.)

## Round 2: depth killed, cliff located
G15 more iterations: T16 .508, T32 .487 = T8 (kill hit). G11 T8: full .9975, frozen
.9633 (SOLVED; paths up to 48 >> 8 iterations, so attention shortcuts, no step-wise
propagation). G13: .71. Cliff is a gradual generalization slope, not depth.

## Round 3: coupled curriculum solves it
tcurr = 4 stages, distance cap at train quartiles + T growing T/4 -> T (user's idea).

| arm (G15) | best_corr | mae |
|---|---|---|
| 9k maps + tcurr (full, T16) | **.980** | **1.75** |
| frozen + tcurr (3k) | .675 | 8.6 |
| 9k maps alone | .649 | 8.9 |
| full + tcurr (3k) | .601 | 9.7 |
| fullinteg + tcurr | .475 | 11.3 |
| d256 (capacity control) | .446 (killed) | |
| plain full T8 (round 1) | .508 | 10.9 |

Synergy: neither 3x data (.649) nor curriculum (.60-.68) alone; together .980. The
wall was optimization/generalization schedule, not capacity (d256 flat) and not depth.
Notes: curriculum helps frozen (cross-attn read) more than full at 3k; the integration
readout consistently trails final-embedding L2 on this bed (metric task with no
compositional gating structure - consistent with switchyard, where integ's edge comes
from guarded/compositional structure, absent in plain mazes).

Transfer bet for the beat-conv-baselines mission: coupled difficulty+compute curricula
are the lever for scale walls (2-for-2 now: switchyard gcurr lucky seeds, maze solved).

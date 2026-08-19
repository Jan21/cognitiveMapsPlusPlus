# Configurations

## Environment

| | |
|---|---|
| switchyard | 7×7, level 5 — crate, gates, XOR wiring, pressure plate, one-way chute |
| diversity | 512 training maps, 48 held-out |
| pairs | 40 per map, distances 1–24, exact BFS |
| encoder | factored, `--seewalls 1` (per-map wall openings exposed) |
| slots | 5 dynamic (worker, crate, 3 gates) × 3 roles + 9 context = 24 tokens |

## Model

| | |
|---|---|
| dimension | `d = 128`, 4 layers, 4 heads |
| iterations | `T = 4` |
| batch | 128 |
| steps | 40k–80k (results given at 60k) |
| seeds | 5 |

## Learning rate

| method | lr | schedule |
|---|---|---|
| **DELPI** | 2e-3 | constant |
| Scalar MLP | 1e-3 | constant |
| IQE | 3e-4 | cosine |
| MRN-fixed | 3e-4 | cosine |
| Sym-Embed | 3e-4 | cosine |

Each selected from a sweep over 8 learning rates × 2 schedules × 2 pooling modes, 5 seeds each.

## DELPI learning-rate sweep

Held-out MAE, 10k steps.

| lr | const | cosine |
|---|---|---|
| 1e-5 | 4.73 | 5.39 |
| 3e-5 | 3.90 | 4.20 |
| 1e-4 | 3.59 | 3.68 |
| 3e-4 | 3.38 | 3.43 |
| 1e-3 | 2.34 | 2.62 |
| **2e-3** | **2.26** | 2.28 |
| 3e-3 | 2.16 | 2.18 |
| 5e-3 | 4.09 | 2.17 |

3e-3 edges 2e-3 at 10k but loses at 40k (1.32 vs 1.26), at 60k (1.10 vs 1.07) and on 5 of 7 ladder
rungs, so 2e-3 is used.

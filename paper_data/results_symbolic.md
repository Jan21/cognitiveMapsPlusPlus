# Reference: the same ladder with symbolic (factored) tokens, no perception

Purpose: the perception-free version of the interdependence ladder. Each state is given as exact symbolic tokens
(`--enc factored`: worker cell, crate cell, gate bits, gate cells, levers with their wiring masks, plate with its mask,
chute with its direction, pooled wall cells; one learned embedding per piece). Same 200 maps / splits / loss / 80k steps /
3 seeds as the image runs. Integrator: d128, 4-layer shared block, T 14, lr 2e-3. Baselines tuned (depth 2/4/6 × lr)
on this bed: IQE 4 layers lr 1e-3, MRN 2 layers lr 1e-3, sym 2 layers lr 1e-3, scalar 2 layers lr 1e-3 (all with
grad-clip 1.0). Run: `leo_symLadder.sbatch` (Leonardo, 105 runs). Unseen-map split.

## Pearson correlation
| rung | integ | IQE | MRN | sym | scalar | margin |
|---|---|---|---|---|---|---|
| L0 | **0.906 ± 0.010** | 0.889 ± 0.009 | 0.876 ± 0.005 | 0.887 ± 0.004 | 0.894 ± 0.004 | +0.013 |
| L1 | **0.946 ± 0.003** | 0.906 ± 0.004 | 0.911 ± 0.005 | 0.888 ± 0.002 | 0.940 ± 0.004 | +0.006 |
| L2 | **0.921 ± 0.004** | 0.848 ± 0.005 | 0.841 ± 0.006 | 0.796 ± 0.010 | 0.882 ± 0.001 | +0.039 |
| L3 | **0.890 ± 0.001** | 0.777 ± 0.007 | 0.754 ± 0.005 | 0.741 ± 0.005 | 0.789 ± 0.013 | +0.101 |
| L4 | **0.898 ± 0.007** | 0.800 ± 0.007 | 0.762 ± 0.016 | 0.753 ± 0.007 | 0.801 ± 0.008 | +0.098 |
| L5 | **0.904 ± 0.006** | 0.781 ± 0.006 | 0.747 ± 0.012 | 0.742 ± 0.002 | 0.796 ± 0.005 | +0.108 |
| L6 | **0.798 ± 0.054** | 0.730 ± 0.004 | 0.700 ± 0.004 | 0.705 ± 0.003 | 0.728 ± 0.001 | +0.068 |

## MAE (moves)
| rung | integ | IQE | MRN | sym | scalar | margin |
|---|---|---|---|---|---|---|
| L0 | **0.46 ± 0.06** | 0.73 ± 0.05 | 0.83 ± 0.02 | 0.74 ± 0.01 | 0.57 ± 0.06 | +0.11 |
| L1 | **1.21 ± 0.05** | 1.96 ± 0.07 | 1.85 ± 0.06 | 2.15 ± 0.02 | 1.43 ± 0.06 | +0.22 |
| L2 | **1.74 ± 0.05** | 2.74 ± 0.06 | 2.78 ± 0.05 | 3.19 ± 0.04 | 2.31 ± 0.01 | +0.56 |
| L3 | **2.05 ± 0.01** | 3.23 ± 0.06 | 3.34 ± 0.03 | 3.48 ± 0.03 | 3.01 ± 0.08 | +0.96 |
| L4 | **1.97 ± 0.08** | 3.08 ± 0.05 | 3.32 ± 0.11 | 3.43 ± 0.03 | 2.93 ± 0.04 | +0.96 |
| L5 | **1.92 ± 0.10** | 3.21 ± 0.05 | 3.38 ± 0.08 | 3.48 ± 0.02 | 2.96 ± 0.04 | +1.04 |
| L6 | **2.93 ± 0.44** | 3.65 ± 0.04 | 3.78 ± 0.01 | 3.81 ± 0.02 | 3.56 ± 0.03 | +0.63 |

## Reading
Margin ≈ 0 on the uncoupled rungs (L0, L1), then +0.04 (L2) → +0.10 / +0.10 / +0.11 (L3–L5) → +0.07 (L6, one weak seed).
This is the undamped shape of the effect; the image-only ladder reproduces it at +0.05–0.09 once perception is solved.

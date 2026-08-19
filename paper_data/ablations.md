# Ablations and diagnostics of the final integrator

Reference model: 16 slots / d 256 / 3-layer shared block / T 4 / 1×1 encoder width 64 / objectness plane, at L5
(full switchyard), 80k steps, 3 seeds. Reference scores: **map 0.860 ± 0.017 / MAE 2.20 · wire 0.852 ± 0.017 / 2.27**
(plain, without the objectness plane: 0.843 / 0.852). Run: `leo_ablate.sbatch`; raw seeds `paper_data/ablations_raw.json`.

## Architecture ablations (corr, mean ± sd over 3 seeds; map / wire)
| ablation | map | wire | Δ vs reference (map) |
|---|---|---|---|
| T = 1 (no recurrence) | 0.857 ± 0.024 | 0.849 ± 0.005 | −0.003 |
| T = 8 | 0.811 ± 0.015 | 0.817 ± 0.082 | −0.049 |
| T = 14 | 0.834 ± 0.020 | 0.827 ± 0.051 | −0.026 |
| no recall (goal/start not re-injected) | 0.833 ± 0.007 | 0.857 ± 0.032 | −0.027 |
| 8 slots (< #entities) | 0.809 ± 0.035 | 0.849 ± 0.025 | −0.051 |
| 12 slots | 0.814 ± 0.015 | 0.829 ± 0.028 | −0.046 |
| 24 slots | 0.841 ± 0.014 | 0.845 ± 0.014 | −0.019 |
| small model (12 slots, d 128, 4 layers; 0.56 M params) | 0.794 ± 0.014 | 0.822 ± 0.019 | −0.066 (still ≥ best baseline 0.773) |
| encoder width 32 | 0.835 ± 0.018 | 0.838 ± 0.059 | −0.025 |
| **3×3 encoder instead of 1×1** | **0.673 ± 0.020** | **0.636 ± 0.022** | **−0.187** |
| shared block depth 2 | 0.832 ± 0.022 | 0.850 ± 0.018 | −0.028 |
| shared block depth 4 | 0.833 ± 0.041 | 0.833 ± 0.033 | −0.027 |
| without objectness plane | 0.843 ± 0.013 | 0.852 ± 0.024 | −0.017 |

Reading: the decisive component is the **per-pixel (1×1) encoder** (−0.19 with 3×3: spatial convolution blurs per-cell
identity such as gate bits and lever colours; this, not binding, was the original image-model failure). Second order:
16 slots (≥ #entities; −0.05 with 8), model size (−0.07 for the 0.56 M variant, which still beats every baseline), the
objectness plane (−0.02). Recurrence depth is flat at this budget (T 1 ≈ T 4 > T 8); goal/start re-injection is worth
≈ +0.03 on unseen maps and nothing on rewired causality at this rung (on the earlier default recipe, removing it at L3
prevented training from fitting at all, train MAE 2.8, so it is kept).

## Perception diagnosis (why 1×1; earlier runs, 40k)
Intermediate rungs between L1 and L2: L1½ = static gate bits, no levers; L1¾ = one gate + one lever. The 3×3 encoder
drops ~0.10 exactly at L1½ (reading per-cell gate bits), before any lever/pairing exists; the 1×1 encoder does not.
Slot-attention maps (per-slot entropy + entity histograms, `slotDiag` runs): with 3×3 features slots never bind
(entropy ≈ ln 49, argmax on empty cells); with 1×1 features slots specialise (entropy 1.4–3.0, e.g. one slot ≈ crate,
one ≈ a gate together with the lever that toggles it, i.e. the wiring is read from the shared stencil).

## Optimiser
lr 2e-3 without clipping produced bimodal seeds (a slot never binds the worker; training plateaus at train-MAE ≈ 3:
0.73 vs 0.93 across seeds). lr 1e-3 + grad-clip 1.0 + 2k warm-up removed the bimodality. Tested and no better: lr 5e-4 /
7e-4 / 1.5e-3, warm-up 500 / 5000, cosine decay, batch 256, 120k / 160k steps, 400 / 800 maps.

## Attempts that did NOT help (each tested, most with seeds)
Competitive iterative slot attention (Locatello-style; softmax over slots, GRU, 1 or 3 iterations, ± init noise,
± reconstruction): worse than one-shot slots at every rung (collapse mode: 7 of 8 slots one-hot on background).
Reconstruction auxiliaries (tied and exclusive-slot variants): +0.04 on the 2-entity hybrid model, nothing on the
final model. Attention-entropy sharpening: hurts (up to −0.5). Hard / straight-through attention: −0.2. Learned
attention bias on lit cells: small gain, excluded from the final model by design choice (input-feature hint only).
Coordinate input channels with slots: no gain (they matter only for the hand-built fgpix tokenizer). Wire-path
rendering (drawing lever→gate connections as paths): hurts. Slot-overlap penalty, slot LayerNorm: hurt. More maps
(400/800) and longer training were flat *on the pre-final recipes* (3×3 encoder era); on the final recipe more maps is
a large win, see "Training-map diversity" below. Deeper/wider blocks, more heads, 1×1-then-3×3 mixing: flat or worse.
Supervised binding (Hungarian-matched slot→entity cell auxiliary): **ceiling below the unsupervised model on the 1×1
encoder** (0.77 vs 0.84–0.86 at L5/L3) — binding is not the remaining bottleneck; on the 3×3 encoder the same ceiling
is 0.71, confirming the encoder, not the binding, was the block.
The hand-built foreground-pixel tokenizer (fgpix + coordinate planes): matches the final model (0.875 / 0.87 at L5)
but is an object detector written for this rendered world; excluded from headline results and from all baselines.

## Environment variants (probe, seed 0, unseen maps; worktree `switchyard-movers`)
Weighted action costs (push ×2, Dijkstra labels): integ 0.853 vs IQE 0.786 / scalar 0.738 — margin unchanged vs unit
costs. Two independently controlled movers: every model improves (IQE 0.834, scalar 0.820, integ 0.886) and the margin
*shrinks* to ≈ +0.05 — two movers' paths are more decomposable, i.e. less coupled. Both support the interpretation that
the integrator's edge tracks **constraint coupling density**, not the number of moving parts or cost weighting; neither
variant was scaled up.


## Readout ablation: accumulation vs decode head (factored setting, reference code, 512 maps, 60k, L5 unseen maps)
Verbatim reference commands (with eval curve): integrate (D = softplus(scale) * sum ||dz||) FINAL 0.961 corr / 1.18 MAE,
best-on-held-out 1.16; decode head (same joint recurrence, MLP on final tokens) FINAL 0.957 / 1.26, best 1.19;
concat_mlp (no recurrence, separate encodings) 0.758 / 3.32. Late-training test-MAE spread: integrate 1.16-1.31,
decode head 1.19-1.50. => In the FACTORED setting the two readouts tie (accumulation slightly better and more stable late in training);
best-on-held-out checkpoint selection shifts MAE by only 0.02-0.07. Single seed each; code
`paper_data/reference/switchyard_walls.py`.

**On the IMAGE setting the decode head fails to FIT at the shared recipe** (same joint recurrence, slots and recipe
as the final integrator, 3 seeds, 80k): train MAE 4-6, test corr 0.39-0.69 / MAE 4.5-5.7 vs accumulation 0.860 / 0.852
(MAE 2.2, train MAE ~0.5). A decode-head-specific hyper-parameter screen (seed 0, 80k each) did **not** rescue it:
lr 3e-4 → 0.704 / 3.60 (train MAE 2.1), lr 5e-4 → 0.753 / 3.39 (train 3.1), lr 2e-3 → nan, warm-up 8k → 0.525,
T 1 → 0.645, T 8 → 0.588, cosine decay → 0.710. Best screened decode head 0.753 vs accumulation 0.860 at matched
budget, and every configuration remains underfit (train MAE ≥ 2 vs 0.5). Raw numbers:
`ablations_raw.json / decodehead_screen`.

**CAVEAT (open, 2026-08-19): implementation mismatch found.** The reference `scalar_mlp` applies a softplus to the
decoded output; our `--decodehead` did not (unconstrained linear output). All image decode-head numbers above were run
without the softplus. Fixed to match the reference; a probe (image L5, lr 1e-3 / 5e-4, 2 seeds each, plus a
never-before-run SYMBOLIC control of our implementation at the symbolic recipe) is running (`leo_dhfix.sbatch`).
Until it lands, do not claim the decode head cannot fit on images; the safe statement is only the factored-setting
tie from the verbatim reference code.

## Training-map diversity (probe, final recipe, 683 training-split maps vs 200)
Scaling the map pool 200 → 683 (identical protocol, m%4 held-out split, 171 unseen test maps, 80k steps, 3 seeds)
lifts the full-switchyard image model from 0.860 / MAE 2.20 to **0.953 ± 0.007 / MAE 1.32 (map)** and
0.943 ± 0.020 / MAE 1.41 (wire); 120k steps adds nothing (0.957, seed 0). An IQE control at 683 maps on the same
slot encoder (single seed) reaches 0.836 / 2.83, so the margin *widens* with data (+0.09 → +0.12); caveat: this
control is not the phase-B′ best IQE variant (3×3 flatten encoder), which would need a rerun for a headline claim.
Perception, not the metric head, is what the extra maps buy: train MAE drops 0.7 → 0.6-0.9 with far better transfer.
Raw numbers: `ablations_raw.json / nmaps683_probe`.

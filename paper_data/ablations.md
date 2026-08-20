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

**RESOLVED (2026-08-20).** Found and fixed an implementation mismatch (the reference `scalar_mlp` applies softplus to
the decoded output; ours did not), then validated our implementation with symbolic controls. Full picture:
- Softplus does not rescue the image decode head: lr 1e-3 collapses to a dead softplus (constant output, corr nan),
  lr 5e-4 gives 0.699 / 0.728 (2 seeds, train MAE ≥ 3.6), same plateau as the no-softplus screen (best 0.753).
- The decode head is lr-fragile by nature: the reference tie itself ran per-readout lr (integrate 2e-3, scalar_mlp
  1e-3). Our symbolic control at lr 2e-3 collapses; at lr 1e-3 / 5e-4 it FITS (train MAE 0.18-0.26), proving the
  implementation correct.
- Symbolic, our 200-map bed: decode head 0.841 ± 0.015 (4 runs, lr 1e-3/5e-4) vs accumulation 0.904 ± 0.007 → −0.06.
- Symbolic, reference 512-map / 501k-pair bed: tie (0.957 vs 0.961) — the decode head catches up given ~10× data.
- Image bed: decode head ≤ 0.753 vs accumulation 0.860 → −0.11, with persistent underfit at every screened recipe.
Claim for the paper: the accumulation readout is more optimization-robust (trains at the shared recipe and across a
wider lr range) and more sample-efficient than a decode head on the same joint recurrence; the gap grows from 0
(data-rich symbolic) to −0.06 (lean symbolic) to −0.11 (learned perception). Raw: `ablations_raw.json /
decodehead_screen, decodehead_softplus, decodehead_symbolic`.

## Training-data scaling (683 maps; two separate levers, 2026-08-20 correction)
The first report conflated two levers. Disentangled (both at 80k steps, 3 seeds unless noted):
- **Map diversity alone** (683 maps, training-pair budget held at 49k pairs = poolq 2000): integ 0.844 ± 0.016 map /
  0.857 ± 0.017 wire; baselines flat vs 200 maps (iqeO 0.772/0.776, sym 0.758/0.759, mrnO 0.742/0.737, scalar
  0.68-0.71). Margins ≈ +0.07/+0.08, same as the 200-map headline. More layouts alone change little for anyone.
- **Maps AND pairs scaled together** (683 maps, 167k pairs = poolq 6800): integ jumps to **0.953 ± 0.007 / MAE 1.32
  (map)**, 0.943 ± 0.020 / 1.41 (wire); 120k steps adds nothing. Baselines were COMPLETELY RE-TUNED on this bed
  (240-config screen, seed 0: slots and 1×1-pixel families at encoder width 32/64, 3×3 CNN at 64/128, pooling ×
  depth × d × lr, all heads; winners then 3 seeds × both splits). Result: iqe 0.860 ± 0.018 map / 0.836 ± 0.044 wire
  (slots w64, 6L, d128, lr 2e-3); scalar revives with data to 0.839 ± 0.063 / 0.839 ± 0.061 (slots w64, 2L, d128;
  seed 0 hit 0.884 but the spread is ± 0.06, the monolithic-readout instability again); sym 0.791; mrn 0.784.
  **Margins: +0.093 map / +0.104 wire — the integrator's edge grows with data** while its MAE nearly halves
  (2.2 → 1.32 vs best baseline 2.51). Winner objch-off checks (seed 0): iqe/sym/mrn within ± 0.01, scalar needs the
  objectness plane (0.755 vs 0.884).
- Best-checkpoint (held-out eval curve, `--evalevery`): ≤ +0.01 corr over final for every model — checkpoint
  selection is not a factor on this bench.
- Bookkeeping: poolq was not logged in RESULT json, which hid the mismatch (now logged). A 4-cell bisect confirmed
  the code paths are bit-identical (old file = current file = eval-on = eval-off at fixed seed).
Raw numbers: `ablations_raw.json / nmaps683_probe, div683_fixedpairs`.

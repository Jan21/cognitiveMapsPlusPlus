# Nomenclature

Canonical terminology for the CognitiveMaps++ / Switchyard paper. This file is the source of
truth; the editor page is a view of it. `Status`: `proposed` until confirmed, then `fixed`.

## Environment
| Term | Symbol | Definition | Aliases (do not use) | Status |
|---|---|---|---|---|
| Switchyard | — | The 7×7 four-room benchmark with worker, pushable crate, gates, levers, pressure plate and one-way chute (capitalised proper name). | switchyard bed, the yard | proposed |
| interdependence ladder | L0–L6 | The seven environment rungs, one mechanic added per rung. | complexity ladder, coupling ladder | proposed |
| rung | Lk | One step of the ladder; L5 is the full Switchyard. | level, tier | proposed |
| coupled rung | — | A rung (L2 and above) whose mechanics gate one another. | interdependent rung | proposed |
| layout | — | Walls and doorway gaps of a map, without wiring. | maze, grid | proposed |
| wiring | — | The hidden lever-to-gate XOR masks plus the plate mask of a map. | causality, coupling structure, coupling configuration | proposed |
| map | — | One sampled instance: layout + gate positions + wiring + chute direction. | configuration, world, yard | proposed |
| observation plane | — | One binary 7×7 input channel of the rendered state (12 planes + objectness). | stencil, channel, canvas | proposed |
| objectness plane | — | Extra input plane lit at every entity cell; an input feature given to every model. | objectness channel, objectness hint, 13th plane | proposed |
| BFS distance | d* | Exact shortest-path move count on the joint state graph; the training label. | true distance, oracle distance, ground truth | proposed |

## Model
| Term | Symbol | Definition | Aliases (do not use) | Status |
|---|---|---|---|---|
| recall-flow integrator | — | Our model; "the integrator" after first mention. | integ, image integrator, flow | proposed |
| slot | — | One of the K=16 learned queries that cross-attend the 49 cell vectors to form a state's tokens. | object token, state token, query token | proposed |
| per-cell encoder | — | The 1×1-convolution encoder; each cell's feature depends on that cell only. | 1×1 CNN, per-pixel encoder, pointwise encoder | proposed |
| recall | — | Re-injection of the goal and start-anchor token blocks before every pass. | re-injection, goal conditioning | proposed |
| recall loop | T | The T weight-shared passes of the shared transformer block. | recurrence, iterations, flow steps | proposed |
| accumulation readout | D | D = softplus(s) · Σ_t Σ_i ‖z_i(t) − z_i(t−1)‖ over the K start slots. | integrated-path readout, displacement sum, norm readout, Σ‖Δz‖ readout | proposed |
| decode head | — | Readout ablation: the same joint recurrence with an MLP decoding distance from the final tokens. | scalar_mlp (reference-code name, cite once), decode-head ablation, MLP head | proposed |

## Baselines
| Term | Symbol | Definition | Aliases (do not use) | Status |
|---|---|---|---|---|
| IQE | — | Interval Quasimetric Embedding head (long form once at first use). | iqe head, quasimetric baseline | proposed |
| MRN | — | Metric Residual Network head (long form once at first use). | mrn head | proposed |
| metric embedding | ME | Symmetric metric baseline; L1 distance between the two state encodings (long form "symmetric L1 metric embedding" at first use). | sym, symmetric baseline, L1 embedding | fixed |
| scalar | — | MLP on the concatenated state encodings; the no-inductive-bias control. | concat_mlp, scalar MLP, monolithic head | proposed |
| unconstrained best | — | A head's best variant over the full encoder/hyperparameter search. | big baseline | proposed |
| parameter-matched | M | A head's best variant capped near the small integrator's parameter count. | param-matched, matched | proposed |
| variant suffixes | O, M, MO | O = with objectness plane, M = parameter-matched, MO = both (defined once in the main table caption). | objch variant | proposed |
| fgpix tokenizer | — | The hand-built foreground-pixel tokenizer; excluded from comparisons, reported only as an ablation. | foreground tokens, entity tokens, hand-built detector | proposed |

## Data, settings and splits
| Term | Symbol | Definition | Aliases (do not use) | Status |
|---|---|---|---|---|
| image-only setting | — | Input is the observation planes only; no symbolic tokens, coordinates or detectors. | pure-image, pixels-only, pureimage | proposed |
| symbolic setting | — | The perception-free reference: exact factored tokens instead of images. | factored setting | proposed |
| unseen-maps split | map | Test on held-out layouts with their own wiring. | map split, held-out maps, unseen layouts | proposed |
| rewired-causality split | wire | Same layouts in train and test; test wiring resampled. | wire split, rewired split, wiring-resampled | proposed |
| training-pair pool | — | Distance-stratified BFS pairs from `--poolq` anchor queries (≈24.5 pairs per query). | pool, pair budget | proposed |
| standard bed | — | 200 maps / ≈49k training pairs; the ladder campaign scale. | 200-map bed, lean bed | proposed |
| large bed | — | 683 maps / ≈167k training pairs; baselines fully re-tuned on it; the headline scale. | headline bed, 683-map bed, data-rich bed, scaled bed | proposed |
| seed | — | One draw changing maps, training pairs and initialisation together. | run seed | proposed |

## Metrics
| Term | Symbol | Definition | Aliases (do not use) | Status |
|---|---|---|---|---|
| correlation | r | Pearson correlation between predicted and BFS distance on the held-out pool. | corr (tables only), test_corr, accuracy | proposed |
| MAE | — | Mean absolute error of the predicted distance, in moves, on the held-out pool. | test_mae, error | proposed |
| margin | — | Integrator minus best baseline variant in correlation (baseline minus integrator for MAE); positive = integrator better. | edge, gap, delta | proposed |
| best-checkpoint score | — | Best point of the held-out evaluation curve (every 8k steps); reported as an audit column only. | best_corr, best-ckpt | proposed |
| noise floor | — | Run-to-run correlation spread at a fixed seed (≈0.02–0.03); smaller differences are not results. | GPU noise, seed noise | proposed |

# Overnight campaign 2026-08-18/19: improving the pure-image integrator on the full switchyard (L5)

210 runs (Leonardo wave 1: 96 single-seed ideas; CIIRC wave 1b: 20 seed replicas; Leonardo wave 2: 114 confirmation
runs = 14 candidates x 3 seeds x {map, wire} + baselines on the same tokens + top-2 at L3/L6). All pure image
(`--enc pureimage --cnnk 1`), safe recipe (`--lr 1e-3 --gradclip 1.0 --warmup 2000`), 200 maps, 80k steps.
Starting point: 12 one-shot slots, T4, 4 layers -> L5 map 0.799 / MAE 2.93, wire 0.808 / 2.86. Symbolic reference 0.904 / 1.92.

## Result: new best pure-image recipe = foreground-pixel entity tokens + coordinate channels

```
python3 switchyard.py --train --enc pureimage --cnnk 1 --readout fgpix --coordconv 1 --T 4 --layers 4 \
  --d 128 --heads 4 --cnnw 64 --cnndepth 2 --nmaps 200 --poolq 2000 --steps 80000 \
  --lr 1e-3 --gradclip 1.0 --warmup 2000 --nobaseline --split map        # wire: prefer --layers 3
```
`fgpix` = tokens are the (1x1-CNN) features at the cells lit in any entity channel of the input image, each with its
position embedding (exact per-entity tokens derived from the picture, no labels, no identity); `coordconv` = two
x/y coordinate channels appended to the CNN input.

| L5, 3 seeds | corr | MAE |
|---|:--:|:--:|
| previous best, map (12 slots) | 0.799 | 2.93 |
| **fgpix + coordconv, map** | **0.875** (0.89/0.84/0.90) | **2.20** |
| baselines on the same tokens, map: scalar / sym / IQE | 0.803 / 0.762 / 0.744 | 2.96 / 3.33 / 3.65 |
| previous best, wire | 0.808 | 2.86 |
| **fgpix (+cc) + 3 layers, wire** | **0.872 / 0.873** | **2.27 / 2.29** |
| baselines on the same tokens, wire: scalar / sym / IQE | 0.821 / 0.757 / 0.752 | 2.79 / 3.41 / 3.62 |

Also lifts L6 (0.723 -> 0.762 with fgpix+cc) and L3 (0.832 -> 0.852 with 16 slots / d256 / 3 layers).
Gap to symbolic shrinks from ~0.10 to ~0.03; margin over the best baseline (given identical tokens) +0.05..+0.07.

## What else moved (wave 2, 3 seeds, L5 map / wire)
fgcc + curriculum 0.861 / 0.866 · fgcc + 3 layers 0.854 / 0.872 · fgcc + d256/L3 0.849 / 0.868 · fgcc + T1 0.853 / 0.859 ·
fgcc + T8 0.846 / 0.862 · slots16 + d256 + L3 0.843 / 0.852 · fgpix (no cc) + L3 0.839 / 0.873 · slots + coordconv 0.827 / 0.831.

## What did not (wave 1, single seed unless noted, L5 map; base 0.799 +- 0.05 run-to-run)
overlap penalty (0.81 / 0.70 at w=1), slot LayerNorm 0.75, lr 2e-3 0.70, lr 7e-4 0.74, warmup 500 0.74, cosine 0.79,
800 maps 0.78, layers 6 0.79, heads 8 0.80, cnnmix 0.80/0.77, 3x3 reference 0.74, curriculum on map 0.78 (helps wire),
T8 on map 0.77, bs 256 0.82, 120k/160k steps 0.81 (no gain), 400 maps 0.80.
**Supervised (Hungarian) binding ceiling on this encoder: 0.768 (12 slots) / 0.733 (8 slots) -- BELOW unsupervised.**
Binding is no longer the bottleneck at L5; token content + explicit coordinates were.

## Files
Leonardo `$CINECA_SCRATCH/cmpp_out/wave1Leo_*.out`, `wave2Leo_*.out`; CIIRC `wave1b_*.out`, `bindPure_*.out`.
Drivers `leo_wave1.sbatch`, `leo_wave2.sbatch`, `wave1b.sbatch`. Aggregates in `pure_image_results.json` (key `overnight`).
Artifact: Pure-Image Scoreboard (section 0).

# IQE applied the authors' way (Wang & Isola, arXiv:2211.15120)

Question: did our benchmark under-use IQE by wrapping it in our own encoder? Test: apply IQE
exactly as the paper's experiments do, with none of our architecture, on the full Switchyard
(L5), image input, 683 maps / 167k pairs, same pools and held-out evaluation (harness:
`distance_model/iqe_author.py`, ran on ciirc A40s, 2026-08-23).

Paper recipe (from the paper's appendix): deep MLP encoder input-2048-2048-2048-512 with
BatchNorm, 512-d latent read as k x l components; IQE-maxmean or IQE-sum head; MSE on
gamma-discounted distances (gamma 0.9); Adam lr 1e-4, cosine decay to 0, batch 1024; lr tuned
in {1e-4, 3e-4, 1e-3, 3e-3, 1e-2}. Our screen: 16k steps x batch 1024 = 16.4M samples (our
runs: 10.2M), seed 0.

## Round 1 (their defaults, 5 variants)

| variant | corr | MAE |
|---|---|---|
| v1 MLP-BN + IQE-maxmean l=64, discounted loss (paper headline) | 0.652 | 4.84 |
| v2 IQE-sum l=64 | 0.647 | 4.99 |
| v3 IQE-maxmean l=16 | 0.672 | 4.55 |
| v4 CNN encoder + IQE-maxmean l=64 | 0.711 | 4.42 |
| v5 their arch, raw-distance MSE, lr 1e-3 | 0.704 | 4.78 |

Reference points, same bed and eval: IQE inside OUR structured encoder (slot cross-attention +
transformer mix) 0.861 recorded / 0.891 A40 rerun, MAE 2.28; integrator 0.944 / MAE 1.45.

## Verdict (round 1)

Author-faithful IQE lands at 0.65-0.71: our wrapper is +0.15 to +0.24 corr ABOVE faithful
usage, not below it. The benchmark comparison gave IQE our best encoder; the quasimetric head
is not the bottleneck, the plain-encoder binding of the image is. Round 2 (lr tuning over the
authors' own grid on the two best variants + one 3x-budget run) pending; will be appended.

Caveats: seed 0 only so far; 16k steps (sample count above ours, wall-clock below); the
discounted loss underweights far pairs by design (gamma^28 = 0.05), visible in the MAE.

## Round 2 (lr over the authors' grid + 3x budget)

| variant | corr | MAE |
|---|---|---|
| cnn, discounted, lr 3e-4 | 0.718 | 4.32 |
| cnn, discounted, lr 1e-3 | **0.720** | 4.26 |
| cnn, discounted, lr 3e-3 | 0.219 (diverged) | 6.42 |
| cnn, raw MSE, lr 3e-4 | 0.707 | 4.48 |
| cnn, raw MSE, lr 1e-3 | 0.714 | 4.57 |
| mlp, raw MSE, lr 3e-4 | 0.716 | 4.61 |
| mlp, raw MSE, lr 3e-3 | 0.702 | 4.69 |
| cnn, raw MSE, lr 1e-3, 48k steps (3x budget) | 0.713 | 4.70 |

## Final verdict

Author-faithful IQE plateaus at ~0.72 on this bed regardless of reduction, component split,
encoder family, loss convention, learning rate (their own grid), or 3x training budget. Our
structured-encoder wrapper lifts the same head to 0.861-0.891, and the integrator sits at
0.944. The benchmark did not under-use IQE; it over-served it. The limiting factor for plain
author-style pipelines here is binding the image into entities, exactly the thing the
structured encoder (and the integrator) provide. One seed; the 0.15+ gap dwarfs the 0.02-0.03
noise floor.

# Protocol, fairness rules, compute

## Shared across every model and every reported number
- Same generated data per seed (maps, pairs, split), same loss (smooth-L1 to exact BFS distance), same budget (80k
  steps × batch 128), same optimiser family (Adam), same stabilisers (grad-norm clip 1.0, 2k-step warm-up; the
  non-finite-loss skip guard is available to all and only MRN ever triggers it).
- **One setting per model**, tuned on the full switchyard (L5, unseen-map split, seed 0), then used unchanged at every
  rung and on both splits. No per-rung tuning.
- **3 seeds** for every reported cell; seed changes maps, pairs and initialisation together. Tables give mean ± sd.
- Run-to-run noise at a fixed seed (GPU non-determinism) is ≈ 0.02–0.03 corr on this bed; differences under ≈ 0.03
  are not treated as results.
- Baselines were given an encoder search (tokens: 12 slots / all pixels via 1×1 / all pixels via 3×3 CNN depth 2–4,
  width 64–128, ± coordinate planes; pooling: mean or flatten; own transformer depth 0/2/4/6; width 128/256; lr
  5e-4/1e-3/2e-3; ± objectness plane) and appear twice: unconstrained best and parameter-matched best (≤ ~0.8 M).
- Metrics: Pearson r between predicted and true distance on the held-out pool (`test_corr`) and mean absolute error in
  moves (`test_mae`); train MAE and parameter count are logged for every run.
- The integrator's hand-built tokenizer variant (`fgpix`) is excluded from all comparisons and from all baselines.

## Compute (final image-only results only)
Leonardo (CINECA, A100 64 GB, account EUHPC_B38_121): baseline tuning ≈ 180 runs; phase B′ 672 runs; integ rows 84
runs; ablations 72 runs; ≈ 0.8 A100-h per run → ≈ 800 A100-h (≈ 6 400 core-h). CIIRC A40 cluster and the OSU
workstations carried the earlier hybrid-model campaigns, the perception diagnosis and the objectness-hint sweeps
(≈ 400 runs). Everything is reproducible from the sbatch drivers in `distance_model/` (`leo_phaseBp.sbatch`,
`leo_phaseBq.sbatch`, `leo_tuneA*.sbatch`, `leo_ablate.sbatch`, `leo_symLadder.sbatch`).

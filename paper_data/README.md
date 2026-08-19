# paper_data — everything for the experimental section

| file | contents |
|---|---|
| `environment.md` | Switchyard definition, observation planes, the 7-rung ladder with flags and statistics, data generation and splits |
| `model.md` | the recall-flow integrator as used in the final results: recipe, forward pass, design choices and their provenance |
| `baselines.md` | the four baselines, their best tuned versions (unconstrained and parameter-matched), the complete search space |
| `protocol.md` | fairness rules, seeds, metrics, noise floor, compute |
| `results_main.md` | final image-only tables: 7 rungs × {unseen maps, rewired causality} × 3 seeds, corr and MAE, margins, winning baseline variants |
| `results_symbolic.md` | the same ladder with symbolic tokens (perception-free reference), tuned baselines, 3 seeds |
| `ablations.md` | integrator ablations at full complexity (T, recall, slots, size, width, encoder, depth, objectness plane, fgpix), perception diagnosis, negatives |
| `other_beds_and_scope.md` | crateworld / gridworld / hybrid supporting results, and the axes with no edge |

Raw data: `distance_model/phaseBp_results.json` (final tables), `distance_model/pure_image_results.json` (earlier ladders,
symbolic ladder, pinpoint, slot diagnostics), Leonardo `$CINECA_SCRATCH/cmpp_out/`. Interactive artifacts: "Image
Integrator, Explained and Scored", "Pure-Image Scoreboard", "Switchyard Ladder Playbook", "Fair Benchmark Ledger".

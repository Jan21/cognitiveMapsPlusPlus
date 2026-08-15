---
summary: cognitiveMapsPlusPlus current state: distance_model novelty survey done (readout confirmed novel across 4 deep-research runs); reproduction package and results page committed earlier.
---

# cognitiveMapsPlusPlus: current state (2026-08-12)

## Project in one paragraph

Goal-directed distance learning: a recurrent transformer ("recall-flow integrator", `distance_model/integ_distance.py`) predicts the BFS geodesic distance between a state and a goal by accumulating per-step latent displacement norms over T weight-shared steps with start/goal re-injection. Trained only on distances (often only local pairs), it extrapolates to longer distances and unseen environments at fixed compute.

## Where things stand

- **Explainer written (2026-08-12).** Plain-language walkthrough for readers with no context: `distance_model/explainer.html` plus its markdown twin in note 2026-08-12-0814 (readable on a phone).
- **Length extrapolation added + first result (2026-08-12).** New `--Rtrain` flag trains on distances <= 4, evaluates to 12. First run: within-range MAE 0.16, beyond-range MAE 4.1 with no correlation. Matches the known base-folded-per-token artifact from earlier probes; the global-base fix is not yet ported into the clean file. Config generalization is unaffected.
- **Validation sweep running on ciirc-old-cluster (job 125062 + 125069).** All 18 original configs rerunning with the new code on the same hardware (first finished configs match the old numbers within run noise) plus 6 new Rtrain configs. Old sweep outputs preserved in `prev_sweep/` on the cluster.

- **Novelty survey complete (2026-08-11).** Four parallel Deep Research runs (ChatGPT, Claude, Gemini, Grok) unanimously find no exact precedent for the accumulated-displacement readout; the quantity exists elsewhere only as regularizer, diagnostic, or halting signal. Merged, categorized summary with about 60 papers: `distance_model/prior_work.html` (also published privately at https://claude.ai/code/artifact/4425ce67-88e4-46d6-a1ed-c61d47a95025); raw reports in `distance_model/raw/`. See note 2026-08-11-2207.
- **Reproduction package** (`distance_model/`): integ_distance.py, requirements.txt, run_best.sbatch, results.html committed on branch `distance-model`.
- Earlier probe results (strata/DOF readouts, guard-gate navigation, budget-free recall flow) are summarized in the repo docs and project memory.

## Decisions pending

- Paper lead: readout novelty vs. budget-free extrapolation story.
- Verify the two closest 2025-2026 preprints (arXiv:2509.23314, "Truth as a Trajectory") from primary text; sweep amortized geodesic solvers.

## Next steps

- Baselines vs. IQE, MRN, scalar head on extrapolation (the comparison reviewers will demand).
- Draft related work from the five discriminating lenses in prior_work.html.

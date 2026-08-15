---
summary: Four deep-research runs agree the accumulated-latent-displacement distance readout is novel; closest prior art mapped, threats and paper framing collected in prior_work.html.
---

# Prior-work novelty survey: distance as accumulated latent displacement

**When:** 2026-08-11 22:07   **Repo:** cognitiveMapsPlusPlus   **Branch:** distance-model

## The question

Our distance model (`distance_model/integ_distance.py`) predicts the goal-conditioned BFS distance not with a scalar head and not as a norm between two embeddings, but as the accumulated path length of its own latent trajectory: a weight-shared transformer block runs T steps with start and goal tokens re-injected each step, and the prediction is `softplus(scale) * sum over steps of ||z(t+1) - z(t)||`. Question: has anyone done this readout before?

## What I did

- Wrote a precise novelty query (`distance_model/prior_work_query.txt`) describing the mechanism, its distinctive properties (fixed T with magnitude-carrying extrapolation, local-only supervision, no policy), and six search directions.
- Ran it through four Deep Research providers in parallel (ChatGPT, Claude, Gemini, Grok). Full reports saved under `distance_model/raw/` (chatgpt.md, claude.md, gemini.md, grok.md, run.json).
- Merged everything into a categorized summary page: `distance_model/prior_work.html` (about 60 papers in 7 groups, with closeness scores and per-paper "why it is not ours" notes). Also published privately at https://claude.ai/code/artifact/4425ce67-88e4-46d6-a1ed-c61d47a95025

## Result: unanimous verdict

All four reports independently conclude **no exact precedent exists**. Consensus phrasing: partially novel, strongly novel on the readout. The quantity "sum of hidden-state displacement norms" appears repeatedly in the literature, but always as a regularizer (Neural ODE kinetic energy, OT-Flow), a diagnostic (trajectory-length expressivity measures, "Truth as a Trajectory" 2026), or a halting signal (recurrent-depth transformers). Never as the supervised prediction target. Two reports independently rate the second novelty as possibly bigger: budget-free extrapolation, where fixed T carries distance magnitude in increment size, is the exact inverse of the "harder problem means more iterations" pattern the whole thinking-networks literature uses.

Closest neighbors (the must-cite-and-distinguish set):

1. **Deep Thinking with Recall** (Bansal et al., NeurIPS 2022): owns the weight-shared-block-plus-re-injection substrate; decodes the final state and extrapolates by iterating more.
2. **Riemannian latent geodesics** (Arvanitidis et al., ICLR 2018): distance literally as integral of local latent speed; but the curve comes from a per-pair geodesic solve on a frozen generator, not an amortized recurrent pass. Most dangerous prior art if we phrase the claim too broadly.
3. **"Truth as a Trajectory"** (arXiv, March 2026): the identical discrete operator, used only as a post-hoc diagnostic of LLM trajectories.
4. **Two-Scale Latent Dynamics** (arXiv:2509.23314, NeurIPS 2025): computes per-step displacement norms in recurrent-depth transformers, as a halting criterion. Closest mechanical neighbor.
5. **Quasimetric RL + IQE / MRN** (Wang and Isola line; Liu et al.): same target and local-supervision philosophy, one-shot embedding readout.
6. **OT-Flow / Finlay et al.**: the same accumulated quantity as a training regularizer.

## Threats and caveats to keep in mind

- Do not claim: "first latent path-length distance" (Riemannian line blocks it), "first to sum displacements across depth" (Truth-as-a-Trajectory blocks it), "we introduce recall" (Bansal owns it). Claim the readout as a new parameterization of goal-conditioned distance.
- Expected reviewer attack: our step-sum is "a temporal generalization of IQE's dimension-sum". Needs a prepared rebuttal plus head-to-head baselines against IQE, MRN, and a scalar head on extrapolation.
- Honest limitation: the construction guarantees nonnegativity but not the triangle inequality across queries (each pair induces its own flow); quasimetric architectures enforce it structurally.
- The most relevant 2025-2026 preprints were read partly from abstracts. Falsification criterion: if any of them sums per-step displacements into a supervised distance output, the claim downgrades to partial anticipation.

## Open questions / decisions needed

- Read the primary texts of arXiv:2509.23314 and "Truth as a Trajectory" before writing any paper claim; also sweep "amortized geodesic solvers" (Gemini flags it as the likeliest remaining collision zone).
- Decide whether the paper leads with the readout novelty or with the budget-free-extrapolation story (two reports argue the latter is conceptually stronger).

## Next steps

- Add IQE, MRN, and scalar-head baselines to the experiment grid for the extrapolation comparison the reviewers will demand.
- Fold the "five discriminating lenses" section of prior_work.html into the paper's related-work outline.

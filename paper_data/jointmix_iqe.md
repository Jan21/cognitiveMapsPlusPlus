# Joint-mix IQE diagnostic (2026-08-23)

Question (from the scalar baseline's strength): is the IQE gap caused by the quasimetric head
family, or by per-state STATIC embeddings that cannot express coupled interactions? Probe:
`--jointmix` in switchyard.py runs the tuned mix block over BOTH states' token sets jointly,
pools the halves separately, and feeds the same IQE head. Pair-conditioned embeddings void the
quasimetric-over-states property (no reusable embedding space, no cross-state triangle
inequality), so this is a diagnostic in the joint-computation family, not a metric-embedding
method.

## Round 1 (683/167k, L5, map split, seed 0; refs same cell: IQE 0.861, scalar 0.884, integ 0.952)

| config | corr | MAE |
|---|---|---|
| jointmix bl6 lr2e-3 | **0.924** | 1.78 |
| jointmix bl2 lr2e-3 | 0.913 | 1.89 |
| jointmix bl6 lr1e-3 | 0.899 | 2.06 |

Kill criterion was gain <= 0.03 over plain IQE; observed +0.063, past scalar (+0.040). Verdict:
joint pair processing is what scalar's advantage consists of; head family is secondary; the
binding constraint on IQE/MRN/Sym is the static per-state embedding. Directly supports the
paper's coupling thesis, and creates a row the paper should own honestly (a reviewer could
build this hybrid and land ~0.03 under the integrator on one seed).

## Confirmation (job 128532): bl6 lr2e-3, 4 seeds per split

Map: 0.924 / 0.914 / 0.771 / 0.911 -> **0.880 +- 0.073**, MAE 2.19.
Wire: 0.828 / 0.896 / 0.914 / 0.805 -> **0.861 +- 0.052**, MAE 2.40.

Same-cell references (4 seeds): integ 0.944 +- 0.017 map / 0.940 +- 0.018 wire; plain IQE
0.857 / 0.846; scalar 0.848 +- 0.057; decode head 0.836 +- 0.052.

## Final verdict

Seed 0 was the favorable tail. Across seeds, joint pair processing before the IQE head gains
+0.023 mean over plain IQE on map (+0.015 wire) and raises the CEILING (best seeds 0.91-0.92,
above every static head), but it inherits the instability of unconstrained joint computation
(std 0.073, one seed at 0.771), the same pathology as scalar and the decode head. The
integrator keeps +0.064 map / +0.079 wire on means with 4x tighter spread.

Paper treatment: include as a diagnostic row or footnote. It completes the mechanism story in
two steps: (1) static per-state embeddings are the coupling bottleneck (jointmix lifts IQE
past scalar on its best seeds; author-style plain-encoder IQE plateaus at 0.72); (2) joint
computation alone is unstable; the path-integration readout is what makes it reliable
(opposite T-response of the decode head, tight seed spread). Not a method: pair-conditioned
embeddings void the quasimetric-over-states property.

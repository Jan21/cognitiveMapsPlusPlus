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

## Confirmation (running, job 128532): bl6 lr2e-3, seeds 1-3 map + 0-3 wire

(to be appended)

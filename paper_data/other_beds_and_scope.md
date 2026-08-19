# Supporting evidence on other beds, and the honest scope of the claim

All numbers below predate the final image-only protocol (they use earlier encoders/recipes) and are secondary evidence.
Sources: `distance_model/fair_benchmark_report.html` (ledger, 2026-08-18), `length_gen_results.md`, `gridworld/RESULTS.md`.

## 1. Crateworld: held-out coupling configurations (image-like input, tuned baselines, 2 seeds)
Crateworld (`distance_model/integ_distance.py`): 4 agents on a 6×6 grid whose mobility is coupled by knobs and links;
splits hold out **coupling configurations**: `combo` (unseen link/knob combinations), `links2` (unseen two-link sets),
`dofhi` (more free degrees of freedom than in training). Positions read from an image (marker canvas, or CNN +
cross-attention), coupling given symbolically (hybrid).
| encoder | split | integ | best baseline | margin |
|---|---|:--:|:--:|:--:|
| marker canvas | combo | 0.985–0.994 | sym 0.92–0.93 | +0.06 |
| marker canvas | links2 | 0.955–0.987 | sym 0.88–0.92 | +0.08 |
| marker canvas | dofhi | 0.965–0.998 | iqe/sym 0.96–0.97 | ≈ 0 |
| CNN + xattn | combo | 0.972 / 0.971 | scalar 0.954 / 0.956 | +0.02 |
| CNN + xattn | links2 | 0.965 / 0.976 | scalar 0.949 / 0.954 | +0.02 |
| CNN + xattn | dofhi | 0.900 / 0.940 | sym 0.970 / 0.955 | −0.04 |
Same direction as switchyard (edge on unseen coupling structure), and the same pattern that the edge is larger when
perception is harder (marker) than when the encoder cleanly separates entities (CNN).

## 2. Switchyard, hybrid CNN encoder (worker/crate from pixels, structure symbolic), tuned baselines, 3 seeds
integ 0.806 / 0.818 (map / wire) vs best baseline 0.781 (IQE) / 0.776 (scalar): +0.025 / +0.04. Superseded by the
image-only results; consistent with them.

## 3. Where the integrator has NO edge (do not claim)
- **Length extrapolation** (train on distances ≤ 8, test to 19–25): on the keys-&-doors gridworld every metric head
  extrapolates (MRN best: corr_beyond 0.99; integ 0.86–0.96); on crateworld a tuned symmetric embedding with the
  radius-curriculum trick ties the integrator (0.76 vs 0.74); on switchyard nobody extrapolates (corr_beyond ≈ 0.36 for
  every recipe).
- **Unseen-map spatial reading without coupling** (gridworld random layouts in range): integ ≈ IQE (0.12–0.14 MAE).
- **Extrapolating to more degrees of freedom** (crateworld `dofhi`): sym ≥ integ.
- The scalar (no-inductive-bias) control fails completely on unseen gridworld maps (MAE 2.0), so a metric bias is
  needed there, but any metric head supplies it.

## 4. The claim this supports
An integrated-path readout generalises to **unseen coupling structure between factors** (held-out wiring/configurations,
unseen maps with coupled mechanics) better than tuned quasimetric, symmetric-embedding and scalar heads, with a margin
that appears at the first coupled rung and persists as coupling deepens, and that is reproduced from pixels with learned
perception (final results). It does not generalise better along distance (length) or along the number of free factors.

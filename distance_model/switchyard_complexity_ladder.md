# Switchyard: the interdependence (complexity) ladder

Six versions of the same yard; each level adds one coupling mechanism. Same encoder shape throughout.
Source: `switchyard_results.html` E4 (2026-08-13/14), 200 maps, unseen-layout split (`--split map`), seed 0,
factored encoder, integrator vs the scalar head. Metric = test MAE / test corr on held-out maps.

## Levels and flags

| level | adds | how to get it |
|---|---|---|
| L0 | plain maze, crate is a static obstacle | `--gatesopen --nopush` |
| L1 | pushable crate (Sokoban) | `--gatesopen` |
| L2 | gates; one lever toggles exactly one gate | `--wire1 --noplate` |
| L3 | XOR multi-gate wiring (one lever flips several gates) | `--noplate` |
| L4 | pressure plate override (crate on plate forces its gates open) | full, `--nchute 0` |
| L5 | one-way chute (full environment) | full (defaults: `--nchute 1`) |

## Original result (integ vs scalar head, factored encoder)

| level | integrator MAE / corr | scalar head MAE / corr | margin (MAE) |
|---|:--:|:--:|:--:|
| L0 | 0.36 / 0.95 | 0.40 / 0.94 | 0.05 |
| L1 | 0.92 / 0.97 | 1.14 / 0.96 | 0.23 |
| L2 | 1.89 / 0.91 | 2.50 / 0.87 | 0.62 |
| L3 | 2.02 / 0.90 | 3.07 / 0.78 | 1.04 |
| L4 | 2.00 / 0.90 | 2.87 / 0.81 | 0.86 |
| L5 | 2.02 / 0.89 | 3.40 / 0.74 | 1.38 (seed 1: 1.98 vs 3.19, margin 1.21) |

Reading at the time: indistinguishable on the plain maze; every added interdependence widens the gap;
integrator plateaus at ~2.0 from L3 on while the scalar head keeps degrading. Companion results:
E2 map scaling (integ 2.90 -> 1.81 from 40 -> 800 maps, margin ~0.6-1.0), E1 wire split (~1.0 margin at
200 maps), E5 curriculum (helps scalar more, still 0.96 gap).

## Caveats (post fair-benchmark, 2026-08-18)

- The ladder compared the integrator to the **scalar head only** -- no IQE / MRN / sym -- and that scalar
  head was the 2-layer-locked, untuned one. Under fair tuning on the full environment the scalar gap
  shrank to ~0.03 corr, so the ladder's *margins* must be re-measured; the *shape* (margin grows with
  coupling) is the claim still worth testing.
- The encoder was factored (symbolic). The pure-image ladder (`pureLadder.sbatch`, `--enc pureimage`)
  re-runs L0, L1, L2 and full with tuned IQE / MRN / sym / scalar and image-only input; L3 (`--noplate`)
  and L4 (`--nchute 0`) rungs to be added for the complete six-level ledger.

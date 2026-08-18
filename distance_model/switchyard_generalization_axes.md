# Switchyard: generalization axes we measure

All axes are "train on some configurations, test on configurations never seen in training".
Distances stay inside the training range unless the length axis is explicitly on. Metric = `test_corr`
(Pearson r between predicted and true BFS distance on the held-out pool), plus `test_mae`.

| axis | flag | what is unseen at test | what it tests |
|---|---|---|---|
| **Unseen maps** (layout + wiring) | `--split map` | 200 yards; train `m % 4 != 0` (150), test `m % 4 == 0` (50). Test yards have new wall doorways, gate / lever / plate / chute positions **and** new lever→gate / plate→gate wiring. | Reading a new map's geometry and wiring, not memorising maps. |
| **Rewired causality** | `--split wire` | 200 layouts built twice: train with wiring A, test the **same** layouts with wiring B (`wire_rng` seed offset 90000). Geometry is familiar, only which lever toggles which gate (and which gates the plate holds) changed. | Reading the causal wiring vs memorising geometry. |
| **Length / distance extrapolation** | `--Rtrain 8` (with `--Rmax 24`) | Training pairs capped at BFS distance ≤ 8; test pairs up to 24. Reported as `corr_beyond` / `mae_beyond` (pairs with d > Rtrain) and `mae_within`. | Extrapolating to longer paths than any seen in training (this is the "depth" axis; needs `--bellman` on switchyard). |
| **Complexity ladder** (not a split, a difficulty knob) | `--gatesopen`, `--nopush`, `--wire1`, `--noplate` | Same splits, simpler dynamics: L0 gates open + static crate → L1 gates open → L2 one-gate-per-lever wiring, no plate → L3 full. | Where a model/encoder breaks as factor interdependence is added (used for the pure-image debugging ladder). |

## Where each landed (fair benchmark, seeds; integrator at its own recipe vs tuned IQE / MRN / sym / scalar)

- **Unseen maps** (CNN image, hybrid encoder): integ 0.806 vs best baseline 0.781 (IQE) → **+0.025**, within about one seed-sd. Tie.
- **Rewired causality** (CNN image, hybrid encoder): integ 0.818 vs best baseline 0.776 (scalar) → **+0.04**. Weak edge.
- **Length extrapolation**: caps at corr_beyond ~0.36 for every recipe on switchyard (all heads); no edge.
- **Complexity ladder**: earlier factored runs: integ's margin over scalar grows monotonically up the ladder (0.05 → 1.38 MAE margin at L3, `switchyard_results.html`); the pure-image ladder (`pureLadder.sbatch`) is the current run.

## Not measured on switchyard (measured on crateworld instead)

Unseen coupling *combinations* (`--heldout combo / links2`) and more-free-DOF extrapolation (`--heldout dofhi`)
exist only in `integ_distance.py` (crateworld). Switchyard's analogue of "unseen coupling" is the wire split.

## Encoders the axes were run with

`--enc factored` (symbolic cell lookup), `bmask` (= factored in effect), `marker` (additive canvas + query
binding; hybrid), `image` (CNN + cross-attention for worker/crate; **hybrid**: gates / levers / wiring /
plate / chute / walls still symbolic tokens), `pureimage` (everything in pixels → K slots or pixel tokens;
**no symbolic tokens**; new). Only `pureimage` is a true image-only model; the ledger numbers above are `image`.

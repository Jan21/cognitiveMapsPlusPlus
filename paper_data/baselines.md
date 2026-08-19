# Baselines: what they are, how they were tuned, and everything we tried

Scope: the image-only switchyard benchmark (`distance_model/switchyard.py --enc pureimage`), final run = phase B′
(2026-08-19, 672 runs on Leonardo: 7 rungs × {unseen maps, rewired causality} × 3 seeds × all model variants).
All models: identical training data (200 maps, `poolq 2000`, 150/50 map split or wiring-resampled split), identical
loss (smooth-L1 to the exact BFS distance), identical budget (80k steps, batch 128, Adam), identical stabilisers
(grad-clip 1.0, 2k-step linear warm-up). **One setting per model, tuned on the full switchyard (L5), reused unchanged
at every rung.** Metrics: Pearson r (`test_corr`) and MAE (`test_mae`) on held-out pairs; 3 seeds (seed changes maps,
pairs and init). Parameter counts are logged per run (`params` in the RESULT json).

---

## 0. The shared front-end every model gets

Input = the rendered start and goal images only: 12 binary 7×7 stencils (walls, worker, crate, gate cells, gate-open
bits, one stencil per lever marking the lever and the gates it toggles, plate + the gates it holds, four chute-direction
stencils) **plus**, for every model including all baselines, a 13th *objectness* stencil (1 where any entity stencil is
lit, `--objch 1`; purely an input feature, never a mask). No symbolic tokens, no coordinates as numbers (except the
optional `--coordconv` planes, see below), no wiring table.

---

## 1. The four baselines (heads)

Each baseline encodes **each state separately** to a vector `f(s)`, `f(g)` and compares:

| head | distance | inductive bias | source |
|---|---|---|---|
| **IQE** | `torchqmet.IQE(f(s), f(g))`, `dim_per_component=16` | learned quasimetric (triangle inequality by construction, asymmetric) | Wang & Isola 2022, torchqmet |
| **MRN** | `torchqmet.MRNFixed(f(s), f(g))` | metric residual network, fixed variant (triangle inequality holds) | Liu et al. 2022, torchqmet |
| **sym** | `‖f(s) − f(g)‖₁` | symmetric metric embedding | – |
| **scalar** | `MLP([f(s); f(g)]) → ℝ` (2 hidden layers, 2d wide, GELU) | none: the no-inductive-bias control | – |

`f(·)` = encoder → tokens → own (non-shared) transformer of depth `baselayers` → pooling → `Linear(d,d)`
(+ optional LayerNorm, `--latentnorm`, tested and rejected). MRN and IQE additionally use a non-finite-loss step guard
(skips a step whose loss is nan/inf; count reported as `nskip`).

---

## 2. Best tuned version of each baseline (used in phase B′)

Two versions per baseline, both reported: the **unconstrained best** over the whole search, and the **parameter-matched
best** (≤ 1.25 × the integrator's small configuration; in practice the slot/mean-pool family, 0.64–0.76 M). Each is run
with and without the objectness channel (`…O` rows). Selection criterion: `test_corr` on the full switchyard (L5),
unseen-map split, seed 0, at 80k steps.

| head | unconstrained best (L5 tuning corr) | params | param-matched best | params |
|---|---|:--:|---|:--:|
| IQE | pixels → **3×3 CNN (depth 3, width 128) + coords → flatten → MLP**, no transformer, d 128, lr 1e-3 (0.751) | 2.0 M | 12 slots (1×1) → 4-layer transformer → mean-pool, d 128, lr 1e-3 (0.754, 3 seeds) | 0.64 M |
| MRN | 12 slots (1×1) → 4-layer transformer → mean-pool, d 128, **lr 2e-3** (0.731) | 0.69 M | same (it is already matched) | 0.69 M |
| sym | pixels → **1×1 + coords → flatten → MLP**, no transformer, **d 256**, lr 1e-3 (0.747) | 6.7 M | 12 slots → 4-layer transformer → mean-pool, d 128, lr 1e-3 (0.728) | 0.64 M |
| scalar | pixels → **3×3 CNN (depth 4, width 128) + coords → flatten → 4-layer transformer → MLP**, d 128, lr 1e-3 (0.734) | 2.8 M | 12 slots → 4-layer transformer → mean-pool, d 128, lr 1e-3 (0.695) | 0.76 M |

Exact flags (prefix `python3 switchyard.py --train --enc pureimage --heads 4 --nmaps 200 --poolq 2000 --steps 80000
--gradclip 1.0 --warmup 2000 --split map|wire --seed {0,1,2} [rung flags]`):

```
IQE   unconstrained : --readout pixels --cnnk 3 --coordconv 1 --cnndepth 3 --cnnw 128 --basepool flat --baselayers 0 --d 128 --lr 1e-3 --iqeonly
IQE   param-matched : --readout xattn --slots 12 --cnnk 1 --basepool mean --baselayers 4 --d 128 --lr 1e-3 --iqeonly
MRN                 : --readout xattn --slots 12 --cnnk 1 --basepool mean --baselayers 4 --d 128 --lr 2e-3 --mrnonly
sym   unconstrained : --readout pixels --cnnk 1 --coordconv 1 --basepool flat --baselayers 0 --d 256 --lr 1e-3 --symonly
sym   param-matched : --readout xattn --slots 12 --cnnk 1 --basepool mean --baselayers 4 --d 128 --lr 1e-3 --symonly
scalar unconstrained: --readout pixels --cnnk 3 --coordconv 1 --cnndepth 4 --cnnw 128 --basepool flat --baselayers 4 --d 128 --lr 1e-3 --scalaronly
scalar param-matched: --readout xattn --slots 12 --cnnk 1 --basepool mean --baselayers 4 --d 128 --lr 1e-3 --scalaronly
(+ --objch 1 for the "O" variants)
```
Rung flags: L0 `--gatesopen --nopush`, L1 `--gatesopen`, L2 `--wire1 --noplate`, L3 `--noplate`, L4 `--nchute 0`,
L5 (none), L6 `--ngate 4 --nlever 3`.

For comparison, the integrator used in the same tables: 16 learned slots (1×1 encoder, width 64) + the objectness
input channel (no attention steering of any kind), d 256, 3-layer shared block, T = 4, lr 1e-3; 1.9 M params (its
d 128 / 12-slot version: 0.56 M). Flags: `--enc pureimage --cnnk 1 --cnnw 64 --readout xattn --slots 16 --objch 1 --d 256
--layers 3 --T 4 --lr 1e-3 --gradclip 1.0 --warmup 2000 --nobaseline`.

---

## 3. Everything tried for the baselines (alternatives and ranges)

### 3a. Encoder / token representation (phase A2 + A3, L5, seed 0, 80k)
| family | variants tried | best corr per head (IQE / MRN / sym / scalar) |
|---|---|---|
| 12 learned slots, 1×1 encoder | pooling mean or flatten; depth 0/2/4/6; d 128/256 | 0.754 / 0.731 / 0.738 / 0.724 |
| all 49 pixels, 1×1 encoder + coords | mean or flatten; depth 0/4; d 128/256 | 0.742 / 0.704 / **0.747** / 0.687 |
| all 49 pixels, 3×3 CNN + coords | depth 2/3/4 conv layers; width 64/128; flatten or mean; transformer depth 0/4; d 128 | **0.751** / 0.693 / 0.738 / **0.734** |
| foreground-pixel entity tokens (`fgpix` + coords) | mean or flatten; depth 0/2/4/6; d 128/256; lr 5e-4/1e-3/2e-3 | 0.766 / 0.782 / 0.791 / 0.825 — **excluded from the final tables**: a hand-written object detector (see §4) |
| hybrid token sets from earlier campaigns (symbolic factored tokens, marker canvas, bmask, 3×3-CNN slots) | — | superseded (see fair_benchmark_report.html) |

### 3b. Pooling / readout of the per-state tokens
mean over tokens · **flatten** all tokens → 2-layer MLP (keeps which-token-where) · (the integrator's recurrent
readout is the thing under test and is not given to baselines).

### 3c. Capacity
own transformer depth `baselayers` ∈ {0, 2, 4, 6} · width `d` ∈ {128, 256} · CNN depth ∈ {2, 3, 4} · CNN width ∈
{64, 128} · attention heads 4 (8 tested on the hybrid bed, no gain) · resulting parameter range 0.45 M – 6.7 M.

### 3d. Optimisation
lr ∈ {5e-4, 1e-3, 2e-3} (hybrid bed also 3e-4); grad-clip 1.0 (0 tested); linear warm-up 2k (0/500/5k tested on
integ); 80k steps (40k screens; 120k/160k tested on integ, no gain); batch 128 (256 tested, no gain); Adam, no
schedule (cosine tested, no gain); non-finite-loss skip guard.

### 3e. Normalisation / extras
`--latentnorm` (LayerNorm on the latent before the metric head): **hurts** IQE/MRN (IQE 0.77 → 0.56 on the hybrid
bed) — off everywhere. Objectness input channel: on/off for every baseline (helps IQE ≈ +0.03 at L2–L5, others ≤ +0.02).
Softplus on the scalar head output (reference concat_mlp parity, `--scalarsp`): tested at L5 best variants, 2 seeds;
flat on the flatten variant (0.719 vs 0.721) and worse + underfitting on the slot variant (0.682 vs 0.730) — off in
all tables (raw: `ablations_raw.json / scalar_softplus_probe`). IQE / MRN / sym need no such constraint (nonnegative
by construction).

### 3f. Earlier on-bed tuning (slot tokens, 40k/80k) used for the first ladders
depth {2,4,6} × lr {5e-4,1e-3,2e-3} at L2/L3 for IQE/sym/scalar; best then: IQE 4L/1e-3, sym 4L/2e-3, scalar 2L/5e-4
(L2) / 6L/1e-3 (L3). Replaced by the single L5 tuning above.

### 3g. What we deliberately did not do
No per-rung tuning in the final run (one setting per model, chosen on L5). No ResNet-scale backbones (a 7×7 grid; every
deeper 3×3 variant we tried lost to shallower ones). No data augmentation for anyone.

---

## 4. The excluded representation: foreground-pixel tokens (`fgpix`)
`--readout fgpix` takes exactly the cells lit in any entity stencil as tokens (their 1×1 features + position code),
padded with a learned empty token. It is an object detector written by hand for a rendered world where empty means
empty; it lifted every head (scalar 0.70 → 0.80; integ 0.84 → 0.875 at L5). We report it only as an ablation
(integrator with slots + objectness channel reaches the same level, 0.87 / 0.88) and never use it for a baseline or for
the headline numbers.

---

## 5. Result at a glance (phase B′, best baseline variant per rung; full tables in `distance_model/phaseBp_results.json`)
Integrator (slots + objectness channel) minus best baseline, corr, map / wire: L0 −0.040 / −0.021, L1 −0.009 / −0.010,
L2 +0.067 / +0.092, L3 +0.072 / +0.083, L4 +0.060 / +0.033, L5 +0.087 / +0.077, L6 +0.051 / +0.050. Baselines win
only where there is no coupling (plain maze, push); the biggest baseline there is a 2–6.7 M-parameter flatten-MLP.

Files: tuning grids `leo_tuneA.sbatch`, `leo_tuneA2.sbatch`, `leo_tuneA3.sbatch`, `tuneA5.sbatch`; selection
`phaseBp_config.json`; final run `leo_phaseBp.sbatch`; raw outputs Leonardo `$CINECA_SCRATCH/cmpp_out/tuneA*`,
`phaseBp_*`, CIIRC `tuneA5_*`.


## 6. External configuration check (colleague's `CONFIGS.md` scalar recipe)
A configuration sheet found in `paper_data/CONFIGS.md` (external origin, method name "DELPI"; setup 512 training maps,
`--seewalls 1`, scalar at 4 layers / lr 1e-3 / 60k) suggested a substantially stronger scalar baseline in the factored
setting. Re-run under our protocol at L5, unseen maps: 200 maps 0.789 ± 0.016 (3 seeds); 560 maps (matching their
diversity) 0.799 (2 seeds); 560 maps without seewalls 0.796. All within noise of our tuned scalar (0.796) and far from
the integrator (0.904). Neither the extra map diversity, the doorway tokens, nor the 4-layer/lr-1e-3 setting changes
the scalar baseline's level; the sheet's implied strength does not replicate, consistent with the earlier `--seewalls`
finding.


## 7. Naming collision resolved: her "scalar_mlp" is not a scalar baseline
The reference code (`paper_data/reference/switchyard_walls.py`) marks `scalar_mlp` as RECURRENT: it runs the
integrator's full joint recurrence (T weight-shared passes over [state | goal | anchor | context] with goal
re-injection) and only replaces the displacement sum by an MLP decode head -- in our code that exact model is the
integrator's `--decodehead` ablation. Verified by running her code at her exact command (512 maps, seewalls, 60k,
L5 unseen maps): scalar_mlp 0.959 corr / 1.20 MAE ~= integrate 0.958 / 1.24, while her own true scalar baseline
(`concat_mlp`, separate encodings, no recurrence) gets 0.758 / 3.32 -- the same level as our tuned scalar. Two
consequences recorded: (a) scalar_mlp belongs in the READOUT-ABLATION table (accumulation vs decode head; in-range the accumulation
readout ties or slightly beats it, 1.18 vs 1.26 MAE verbatim, and is more stable late in training -- the recurrent
joint processing carries the in-range edge); (b) her
run also shows map diversity is a strong lever for the integrator (0.958 at 512 maps vs 0.904 at our standard 200).

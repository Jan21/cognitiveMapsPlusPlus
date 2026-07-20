# Stratified-Space Probe (varying local dimension) — Results

`probe/stratified_knobs_probe.py`. Question: does an adjacency-trained embedding organize
into **strata of different ambient dimension**, matching each state's degrees of freedom?

Env (tractable stand-in for the 20x20 / 4-agent idea): 2 agents on a 5x5 torus, each with a
knob ∈ {all=2D, horiz=1D, vert=1D, none=0D} restricting its moves. State = (posA, posB, kA,
kB), 10000 states, true local DOF = DOF(kA)+DOF(kB) ∈ {0..4}. Trained on adjacency ONLY
(contrastive: neighbors ~1, random pushed apart; knob-changes always available to keep it
connected; no actions). Measured per-state **participation-ratio dimension** of the
move-neighbors' embeddings vs true DOF.

| variant | corr(local dim, DOF) | mean local dim @ DOF 0/1/2/3/4 |
|---------|---------------------:|--------------------------------|
| plain (free table)     | 0.80 | 0 / 1.7 / 2.6 / 3.3 / 3.7 |
| factored (per-agent/knob tokens) | **0.956** | 0 / 1.7 / 3.2 / 4.7 / 6.1 |
| image (pixel grid + attention)   | **0.956** | 0 / 1.8 / 3.5 / 5.2 / 7.0 |

**Confirmed — the space stratifies.** Local (ambient) dimension rises monotonically with DOF;
DOF-0 states (frozen agents) collapse to exactly dimension 0. Absolute values overshoot the
*intrinsic* DOF (DOF 4 → ~7) because participation ratio measures the *ambient* span, and a
curved torus makes the ±move directions fan apart (ambient ≈ ~1.75× intrinsic) — which is
precisely the "different states have different ambient dimension" phenomenon asked about.
**Structured encoders (factored/image) stratify cleanly** — their UMAP fragments into many
island-strata (corr 0.956); the **free embedding table stays one blob** (corr 0.80, weak).
Figures: `factored_vis/stratified_{plain,factored,image}.png`.

### Dimension estimators (adapting the VGT framework from quasimetric_tests/stratified)

Ported the Volume Growth Transform (VGT: local dim = slope of log count-in-ball vs log
radius), Levina-Bickel MLE, and a graph-BFS volume-growth, and applied them to the trained
embeddings. Spearman(estimate, true DOF), full training:

| estimator | plain | factored | image | reads |
|-----------|------:|---------:|------:|-------|
| participation ratio (ambient) | 0.80 | **0.96** | **0.96** | works on embedding, ambient (~1.75x intrinsic) |
| VGT full-cloud (Euclidean)    | -0.63 | 0.13 | 0.16 | **fails** (flat ~5-15) |
| VGT within-stratum            | -0.12 | -0.07 | -0.11 | **fails** |
| MLE (embedding kNN ratios)    | -0.12 | 0.24 | 0.43 | weak/fails |
| **graph-VGT (true move-graph)** | **1.00** | **1.00** | **1.00** | perfect ordering, ladder 0/.5/1.3/1.9/2.4 |

Conclusion: **VGT/MLE do NOT transfer to the embedded point cloud** here — it is a coarse
5-point-per-axis lattice with no continuous `r^d` scaling regime, so radius/kNN-scaling reads
a flat ~ambient value (structural, confirmed at convergence, not undertraining). Two tools
that DO work: (a) **graph volume-growth** on the adjacency = perfect DOF ordering (intrinsic,
needs only a monotonic calibration for magnitude); (b) **participation ratio** on the embedding
= ambient stratification (0.96). To make Euclidean-VGT/MLE viable on embeddings, need denser
sampling (bigger grid -> many points per movement axis).

### Dense grid — VGT on embeddings DOES recover intrinsic dimension

`probe/stratified_dense_probe.py`. VGT is local, so we do NOT enumerate the state space: a
40x40 torus with 2 agents = ~4e7 states, factored embedding trained on sampled edges
on-the-fly; dimension estimated per query from its locally-sampled (deduped) move-ball.

| estimator | corr(dim, DOF) | ladder @ DOF 1/2/3/4 |
|-----------|---------------:|----------------------|
| **embedding-VGT (local ball)** | **0.85** | 0.7 / 2.1 / 3.0 / 3.4 |
| graph-VGT (L1 reference)        | 0.97 | 0.9 / 1.8 / 2.7 / 3.8 |
| (coarse-5x5-lattice VGT)        | ~0.15 | flat ~5 (fails) |

So **applying VGT directly to the embeddings recovers the intrinsic (not ambient) dimension**
once the point cloud is dense enough — corr 0.85, tracking the graph-VGT reference, reading
~1/2/3/4 (vs the ambient participation-ratio's 1.7/3.2/4.7/6.1). Three fixes made it work:
(1) DENSE grid (a real count~r^d regime), (2) DEDUP the local ball (a 1D stratum has only
~2R+1 distinct points; raw N-sampling duplicated them and broke VGT), (3) small-window robust
slope tolerant of low-D strata. DOF-4 compresses a little (3.4 vs 4: VGT boundary/saturation +
finite N; larger N or a monotonic calibration sharpens it); DOF-0 (a frozen single point) is
correctly unmeasurable. Confirms the key insight: you need enough LOCAL neighbors per query,
not the whole state space.

### 4 agents — dimension 0..8 captured from embeddings

`probe/stratified_4agent_probe.py`. 4 agents on a 40x40 torus, each with a movement knob;
local DOF = sum of knob-DOF in {0..8}. State space ~625^4 * 256, NEVER enumerated. Factored
embedding trained on sampled edges; per-query dimension from a locally-sampled deduped
move-ball; queries stratified by DOF for even coverage; N=30000 for high-D density.

| estimator | corr | ladder @ DOF 2/3/4/5/6/7/8 |
|-----------|-----:|----------------------------|
| **embedding-VGT (local ball)** | **0.93** | 2.6 / 3.6 / 4.5 / 5.4 / 5.9 / 6.3 / 7.1 |
| graph-VGT (reference)          | 0.97 | (DOF 1-6) 0.9 / 1.8 / 2.6 / 3.6 / 4.6 / 6.1 |

The per-state intrinsic dimension of the learned embedding is recovered by VGT across the whole
0..8 range (corr 0.93, ladder near-linear in true DOF). Bigger N + DOF-stratified sampling
raised corr 0.88->0.93 and DOF-8 from 6.4->7.05. Limits: DOF-0/1 thin (a frozen point / 29-point
line give VGT too few points); DOF-7/8 compress (curse of dimensionality, 8D ball undersampled).
The map is smooth+monotone, so a one-time calibration curve gives exact DOF. Confirms the whole
program: a contrastively-learned cognitive map organizes into strata whose local ambient/intrinsic
dimension matches each state's degrees of freedom, measurable without enumerating the state space.

**Improved capture (`probe/stratified_4agent_iso_probe.py`):** multi-scale L2 isometry training
(anchor move-pairs to their EXACT flat-torus geodesic, not just 1-step=1) + D=64 raised count-VGT
to **corr 0.949**, ladder 2.1/3.4/4.6/5.9/7.0/8.1/8.4 -- nearly identity, near the 0.97 graph
ceiling. Isometry fixed both error sources: low-D over-read gone (DOF2 2.6->2.1) and high-D
decompressed (DOF8 7.1->8.4). TwoNN (Facco 2017) tried but FAILED here (corr 0.53, over-reads
low-D with only 2 neighbors on thin patches) -- count-slope VGT is the better estimator. Only
residual: DOF 7/8 nearly merge (last of the curse of dimensionality).

Next: gate knob-changes on knob-adjacency (true deadlock strata); calibration curve VGT->DOF;
per-state dimension heatmaps.

---

# Bridged-Tori Factored-Attention Probe — Results

Date: 2026-07-17 (overnight autonomous run)
Branch: `bridged-tori-factored-attention`
Spec: `docs/superpowers/specs/2026-07-17-bridged-tori-factored-attention-design.md`

## TL;DR

- The distance head **must carry a metric prior**. A free-form scalar attention
  distance (B3 as literally specified) does **not** work; an attention head whose
  distance is read out as a **norm** does, cleanly.
- With the norm readout (`self_norm`), the true geodesic — including the forced
  **detour through the bridge node** — **emerges from local one-step supervision
  only** (loss never sees a geodesic). This is the interesting claim (A1) confirmed.

## Setup

- Graph: two 15x15 wraparound tori (450 states), bridged by a single undirected
  edge `(150,x) <-> (150,y)`; otherwise disconnected, so every cross-torus path
  routes through node 150.
- Supervision: local only. `2700` one-step transitions `(state, action, next)` over
  6 actions. Loss = latent-dynamics consistency + neighbor-distance≈1 anchor +
  random-pair repulsion. No geodesic ever shown to the model.
- Baseline: plain `nn.Embedding(450)` + fixed L-1.5 distance, same local + repulsion
  loss (the repo's current contrastive model). No factoring, attention, or actions.
- Kill criterion (set before running): (1) Spearman(D, geodesic) beats baseline by
  >= 0.05; (2) detour signature Spearman(D((p,x),(p,y)), torus_geo(p,150)) >= 0.5.

## Result — seed 0, 4000 steps

Local (CPU) and cluster (Volta100 GPU, job 111033) agree:

| head | Spearman(D, geodesic) | baseline | detour signature | verdict |
|------|----------------------:|---------:|-----------------:|:-------:|
| `cross_scalar` (B3, free scalar) | 0.45 (local 0.13) | 0.67 | 0.05 | **FAIL** |
| `cross_reg` (B3 + triangle/identity knobs) | 0.68 | 0.67 | 0.16 | **FAIL** |
| `self_norm` (attention + norm readout) | **0.86** | 0.47–0.68 | **0.92** (local 0.90) | **PASS** |

`cross_scalar` is unstable run-to-run (peak-then-collapse; where it lands depends on
the stopping step) — that instability is itself the symptom.

**`cross_reg` finding:** adding soft triangle-inequality + identity regularizers to the
pure scalar head lifts geodesic ranking back to ~baseline (0.68) but **still fails the
detour signature** (0.16 ≪ 0.5). Soft metric penalties are not enough to force the
routing-through-bridge structure; only a **hard** metric readout (`self_norm`) recovers
it. So the pure B3 scalar head is not salvageable for this task.

## Root cause of the B3 failure (confirmed by instrumentation)

Periodic Spearman during `cross_scalar` training shows a **peak-then-collapse**:

| step | repulsion loss | anchor loss | Spearman(D, geodesic) |
|-----:|---------------:|------------:|----------------------:|
| 400  | 3.81 | 0.230 | **0.56** (peak) |
| 1200 | 2.72 | 0.282 | 0.45 |
| 1600 | 2.34 | 0.031 | 0.18 |
| 4000 | 1.70 | 0.012 | 0.13 |

A free scalar attention distance has no metric structure. As it tightens the
pointwise constraints (neighbors → 1, random pairs pushed to `offset`), it satisfies
them by making the distance **non-graded** — a pair at true geodesic 3, if sampled as
a "random" pair, is pushed toward `offset` just like a pair at geodesic 25. The L-p
baseline cannot do this because the triangle inequality forbids `d(A,B)=offset` when a
short chain of ~1-cost steps connects A and B; that same constraint is what makes the
geodesic emerge. The free head lacks it, so grading collapses.

## The fix

`self_norm`: keep the factored latent `[z_pos | z_id]` (shared position table across
both tori) and keep attention, but compute the state embedding by **self-attention
over the two factor tokens** and read distance as `||e_u - e_v||_{1.5}`. The norm
restores identity + triangle inequality. Training is stable (climbs to ~0.86 and
holds). The detour signature (0.90) shows same-position-different-torus distance grows
with distance-to-bridge, i.e. the model learned "you must pass through 150", from
local supervision alone.

## Robustness (multi-seed, self_norm)

All runs PASS; `self_norm` Spearman stays in 0.86–0.91, detour 0.81–0.99. The plain
L-p baseline is itself seed-sensitive (geodesic Spearman 0.41–0.68) but always far
below `self_norm` and always ~0 on the detour.

| seed | hardware | Spearman(D, geodesic) | detour | verdict |
|-----:|----------|----------------------:|-------:|:-------:|
| 0 | local CPU    | 0.860 | 0.898 | PASS |
| 0 | cluster GPU  | 0.863 | 0.917 | PASS |
| 1 | local CPU    | 0.909 | 0.986 | PASS |
| 2 | local CPU    | 0.865 | 0.807 | PASS |

## Cluster confirmation (ciirc-old-cluster, GPU)

- Synced to `~/cognitiveMapsPlusPlus` on `ciirc-old-cluster` (user `hulajan1`).
- Job submitted via `probe/run_cluster.sbatch`, runs all three heads at seed 0; JSONs
  in `~/cognitiveMapsPlusPlus/probe/results/`.
- **Gotcha:** torch 2.9 (cu128) on the cluster dropped Pascal `sm_61`, so a plain
  `--gres=gpu:1` landed on gpu-1's GTX 1080 Ti and crashed with
  `no kernel image available` (job 111032, FAILED). Pinning `--gres=gpu:Volta100:1`
  (sm_70) fixed it (job 111033, COMPLETED). Use A40 or Volta100 on this cluster, not
  GTX 1080 Ti.
- Job 111033 results match local: `self_norm` PASS (Spearman 0.863, detour 0.917),
  `cross_scalar` and `cross_reg` FAIL.

## Reproduce

```bash
# local (CPU is fine)
python probe/bridged_tori_probe.py --head self_norm   --steps 4000   # PASS
python probe/bridged_tori_probe.py --head cross_scalar --steps 4000   # FAIL (B3 as specified)

# cluster
sbatch probe/run_cluster.sbatch   # from ~/cognitiveMapsPlusPlus
```

## Disentanglement study (image input)

Question: can the factored tokens (`[z_pos | z_id]`, and 3-factor `[pos | id | color]`)
be made to *specialize* — one token per factor — and can it happen without a hand-crafted
"trick"? Encoder = 2 (or 3) query vectors cross-attending the rendered image. Metric =
per-token sensitivity (how much a token changes when ONE factor varies); clean split = each
token sensitive to exactly one factor.

| method | trick? | geodesic / detour | token split |
|--------|--------|-------------------|-------------|
| no-aux (single task)          | none              | 0.84 / 0.79  | none (both → agent) |
| decorr                        | loss              | 0.08         | degenerate (kills a factor) |
| actsplit (dynamics-Δ)         | loss              | 0.71 / 0.28  | none (dynamics absorbs it) |
| **invar (move-invariance)**   | loss (principled) | 0.84 / 0.81  | **CLEAN** |
| multi-task (2 tasks)          | no-trick          | 0.90 / 0.99  | weak/partial |
| multi-task (3 tasks)          | no-trick          | 0.90 / 0.96  | collapse (one token dead) |
| multi-task + token bottleneck | no-trick (arch)   | 0.86 / 0.97  | none (one token dominant) |
| equivariant dynamics (A)      | no-trick (arch)   | 0.87 / 0.993 | none (subspace only) |
| 3-factor, no-aux (C)          | none              | 0.867        | none (all → id) |
| 3-factor, invar (C)           | loss              | 0.869        | PARTIAL (id clean; pos/color mixed) |
| sparse code (L1), 3-factor    | none (arch)       | 0.90         | color CLEAN (own neurons); pos+id coupled/shared |
| FACTS-lite (recurrent slots)  | none (temporal)   | n/a (world-model) | none (both slots → position; identity under-encoded) |

**Neuron-level (sparse code) vs token-level.** Measuring per-*neuron* (not per-token) selectivity,
the code is already ~85% factor-pure, and L1 sharpens it. Crucially, **independent factors get
clean dedicated neuron groups for free** (color → its own units), but **coupled factors don't**:
position and identity share neurons because the task couples them (crossing = detour via node 150),
so the metric mixes them.

**Temporal factoring (FACTS-lite) insight.** Carrying recurrent slot-attention slots over
trajectories with next-observation prediction (no disentangle loss) did NOT slot identity — both
slots tracked position and identity was under-encoded. Lesson: a factor earns a slot only if it is
BOTH stable AND predictively relevant. Identity here is stable but nearly prediction-irrelevant
(static markers, rare crossing), so prediction ignores it rather than dedicating a slot.

Conclusions:
1. **Token-level disentanglement needs an encoder-OUTPUT constraint.** Any pressure applied
   downstream (readout heads, dynamics net) is satisfiable without splitting the tokens,
   because the model reads them jointly — multi-task collapses to one token, actsplit is
   absorbed by the dynamics MLP, equivariant dynamics factor only at the subspace level.
2. **The constraint must resist degeneracy.** `invar` = penalize `min_token(move-change /
   overall-spread)`. The spread-normalization is what blocks the "kill a factor" cheat that
   made plain `decorr` collapse.
3. `invar` is **clean for 2 factors, partial for 3** (identity extracts cleanly; position and
   color stay mixed). The "min per action-group" generalization is necessary but not
   sufficient — refining it (pairwise-disjoint or per-factor invariance) is future work.
4. **No-trick methods give the best geometry but not the split**: multi-task and equivariant
   dynamics reach ~0.90 / 0.99, better than the plain model (0.84 / 0.79). Task diversity and
   a constrained world-model help the *metric*, not the token factorization.
5. **Coupled vs independent factors is the key distinction.** *Independent* factors (color) get
   clean dedicated neuron groups for free from sparsity / architecture. *Coupled* factors
   (position + identity, coupled by the bridge detour) resist every emergent method we tried
   (multi-task, bottleneck, equivariant dynamics, sparse coding, temporal/FACTS-lite) — only the
   explicit encoder-level `invar` splits them cleanly, and even it is only partial at 3 factors.

**Practical recipe:** `invar` (spread-normalized move-invariance) for the coupled position/identity
pair + sparsity for any independent extra factors. Best geometry (if disentanglement isn't required):
equivariant dynamics or multi-task (~0.90 / 0.99).

Artifacts: `probe/bridged_tori_{image,multitask,equivariant,3factor}_probe.py`,
`probe/results/*.json`, attention maps `factored_vis/image_attention_maps_*.png`.

## Next steps (only now that the probe shows signal)

1. Productionize `self_norm`: add `bridged_tori` under `generate/graph_types/`, a
   transition dataset yielding `(state, action, next)`, and Hydra configs
   (`config/data/bridged_tori.yaml`, `config/model/factored_attention.yaml`), wiring
   the attention distance + dynamics into the lightning module.
2. Report `cross_reg` outcome: if the pure scalar head is salvageable with the metric
   knobs, note it; otherwise `self_norm` is the recommended head.
3. Visualize learned `z_pos` / embeddings (PCA) colored by torus to see the two tori
   joined only at the bridge (the stratified-space picture from the repo README).
4. Stress tests: move the bridge node, add a second bridge, vary torus size, check the
   emergent metric still matches geodesic.

---

## Gated-knob probe — genuine low-dim states + local-neighbourhood dimension

`probe/stratified_gated_probe.py`. Sharpens the story: a knob can be flipped **only** when its
agent stands on that knob's control cell. So far from all knobs you cannot switch movement-mode
at will — the local neighbourhood is *genuinely* the current movement, and can be truly
low-dimensional. Image input (grid pixels + one always-visible knob-value pixel per agent, never
occupied). Dead states (all frozen, none at a control cell) are excluded and no transition
creates one. Trained on adjacency only: multi-scale isometry within a config (dist = flat-torus
L2 geodesic, per-sample scale 1..K) + gated knob-flip edges at distance 1 + a repulsion term.

Dimension is read **at representative points**, embedding only each point's local neighbourhood
(no UMAP): exact BFS enumeration of the legal R=7 ball, then correlation-dimension slope
(log rank vs log sorted embedding-distance over [0.05, 0.5]N). 2 agents, G=15, 20k steps.

| representative state | seed1 | seed2 | true local dim |
|----------------------|------:|------:|:--------------:|
| all,all far          | 4.25 | 4.36 | 4 |
| all,none far         | 2.13 | 2.19 | 2 (a1 frozen, a0 2D) |
| horiz,vert far       | 2.19 | 2.14 | 2 (1D+1D) |
| none,all far         | 2.14 | 2.10 | 2 (a0 frozen, a1 2D) |
| none,none a0@ctrl    | 2.02 | 2.63 | junction (>1) |
| all,all a0@ctrl      | 4.66 | 4.91 | ~4 (+flip bridges) |

**Confirmed from image input, no UMAP.** A frozen agent far from its knob contributes **0**
(none,all far reads ~2.1 = only the free agent), so the gating genuinely creates low-dim states.
Pure-movement dims are recovered with a clean factor-2 gap (2D ~2.1 vs 4D ~4.3). A **control cell
is a stratum junction**: standing there (able to flip into other movement modes) reads *above*
the surrounding frozen bulk. Mild overshoot on 4D / junctions is embedding noise + correlation-dim
on a curved patch; separation is unaffected.

### Debugging note (measurement, not model)
Early runs returned `nan` on every 1D/2D state. Root-caused on **ideal flat-torus distances**
(zero model): (1) a random-walk ball wasted steps on frozen agents → 62-state balls of radius ~4;
(2) `vgt_slope`'s median-of-windows was dragged to 0 by the saturated tail of a small ball; and,
once those were fixed, (3) the trained embedding squeezed distances into a narrow band with
collapsed near-neighbours, so log-spaced radii left <4 populated bins. Fixes: exact BFS ball +
correlation-dimension estimator (evaluate growth at the observed distances, skip the collapsed
bottom fraction) + multi-scale isometry (adjacent states stay ~1 apart instead of collapsing).

### 4 agents (DOF 0-8), G=24 — the DOF ladder

Scaled the gated probe to 4 agents on a 24x24 torus (image input, d_model=96/D=64/16 slots,
30k steps). Representative "DOF ladder": t//2 agents in 'all' (2D), one 'horiz' (1D) if t is odd,
the rest frozen — giving a pure-movement state of total DOF t, all agents placed far from their
own control cells. Local dimension read by exact BFS ball (R=12, cap=60000) + correlation-dim.

| true DOF | seed1 | seed2 | ideal ceiling |
|---------:|------:|------:|--------------:|
| 1 | 1.22 | ~1*  | 0.96 |
| 2 | 2.11 | 2.23 | 2.00 |
| 3 | 3.47 | 3.36 | 3.00 |
| 4 | 3.74 | 4.35 | 4.00 |
| 5 | 4.96 | 5.29 | 5.00 |
| 6 | 5.07 | 5.97 | 5.78 |
| 7 | 5.34 | 6.51 | 6.34 |
| 8 | 6.14 | 6.86 | 6.75 |
| junction (all-none, a0@ctrl) | 2.08 | 1.93 | — |

*seed2 DOF1 mean is a 12.79±20.30 outlier: one of 4 reps on the thin 24-point 1-D ring hit a
near-flat log-distance range and the slope ran away; the other reps sit near 1.

**Confirmed — the ladder holds at 4 agents.** Seed 2 nearly reproduces the ideal ceiling across
the whole range: monotone, cleanly separated, tracking true DOF. The image-trained embedding
recovers the full stratified structure from adjacency alone. High-DOF compresses (8 -> ~6.9)
because an 8-D volume needs a radius unreachable with finite points (curse of dimensionality),
not a model failure — the *ideal* geometry compresses identically. A **control cell is a junction**:
'all-none' with a0 on its control cell reads ~2, where the surrounding fully-frozen bulk is 0-dim
(dead) — a genuine dimension bump where a stuck agent can switch movement modes.

### Embedding-NN neighbourhood (dropping the graph) — negative result

Question: can local dimension be read from the embedding's OWN nearest neighbours, without the
BFS legal ball? Sampling ruled out a global uniform bank first (1e6 uniform states give 0
neighbours within radius 3 of a DOF2 probe; an 8-D local ball can't be filled uniformly), so the
pool is a per-probe local jitter of ALL agents (+ mixed knobs), legality never consulted, and
embedding distance alone picks the k-NN ball (W=6, M=60000, k=3000).

| true DOF | BFS ball | emb-NN ball |
|---------:|---------:|------------:|
| 1 | 1.19 | 8.02 |
| 2 | 2.19 | 8.21 |
| 3 | 3.36 | 8.17 |
| 4 | 3.89 | 8.50 |
| 5 | 5.17 | 8.84 |
| 6 | 5.66 | 8.34 |
| 7 | 5.00 | 8.15 |
| 8 | 6.19 | 8.43 |
| junction | 2.08 | 8.71 |

**Embedding-NN reads ~8 flat for every state; the strata are invisible to it.** The gate lives in
the *graph* (which moves cost one step), not in the embedding's local Euclidean metric. The image
encoder maps a small pixel move of *any* agent — legal or not — to a small embedding move, and the
isometry loss only supervised *legal* moves, so displacing a frozen agent was never pushed apart.
The embedding is thus locally ~isometric to the full 8-D position space (4 agents x 2 axes), and
Euclidean k-NN recovers that ambient 8 everywhere. Reading a low stratum dimension requires the
legal-move structure (BFS); it cannot be recovered from embedding neighbours alone.

Follow-up if emb-NN is wanted: add a training term that pushes *illegal* local perturbations apart
(so non-adjacent neighbours become geometrically far), making the gate part of the embedding's
local metric rather than only its adjacency. Model checkpoint saved at factored_vis/gated4_seed1.pt
for retuning without retraining.

### Making the EMBEDDING stratified — 5 hypotheses, criticism, and an ideal pre-check

Goal: not "BFS-on-the-graph recovers dimension" (that measures the graph) but "the embedding's OWN
local geometry is stratified", so a graph-free embedding-NN reads local dim = DOF. Key fork for an
illegal move (e.g. displacing a stuck agent): REPEL it far (injective glued strata, but a generic
jitter pool has ~0 on-stratum neighbours -> unmeasurable) vs COLLAPSE it to the same point (quotient
immovable axes -> the local manifold's extent IS the movable axes -> emb-NN reads DOF).

Ideal pre-check (no training): graph-free emb-NN on a generic jitter pool, using idealised
distances. COLLAPSE distance -> 2.35/2.84/3.96/4.93/5.87/6.99/8.07 across DOF 2..8 (tracks DOF).
REPEL distance -> 18.8/12.2/9.5/7.8/7.3/5.6/8.1 (garbage, non-monotone). So collapse is the
mechanism; repulsion stratifies but stays unmeasurable graph-free.

Five tests (image G=24, 4 agents unless noted):
- E1  illegal-repulsion hinge (margin, image)        -> expect large legal/illegal gap, emb-NN fails
- E7  illegal-COLLAPSE loss (image)                  -> expect emb-NN tracks DOF
- E5-learned  factored encoder, knob-gated (learned gate + collapse loss)
- E5-hard     factored encoder, hardwired gate (zeroes immovable axes)  -> upper bound
- E2  knob-guided emb-NN sampling on the E1 model (jitter only knob-allowed axes)
Dropped: InfoNCE (redundant with E1, loses absolute metric); per-anchor reachability negatives
(BFS-per-step too slow, E1 is its tractable proxy).

Early signal: factored + hardwired gate reads the ladder via graph-free emb-NN at 400 steps
(2.75/3.99/4.86/6.45/7.46/8.70 for DOF 2..7), legal neighbours ~1.2 vs illegal 2-10. Full runs
in progress (E1 112169, E7 112171, factored 112172).

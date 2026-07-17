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

| head | Spearman(D, geodesic) | baseline | detour signature | verdict |
|------|----------------------:|---------:|-----------------:|:-------:|
| `cross_scalar` (B3, free scalar) | 0.13 | 0.66 | -0.01 | **FAIL** |
| `self_norm` (attention + norm readout) | **0.86** | 0.68 | **0.90** | **PASS** |
| `cross_reg` (B3 + triangle/identity knobs) | see `probe/results/` | — | — | (cluster) |

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

_(filled after the multi-seed run completes)_

## Cluster confirmation (ciirc-old-cluster, GPU)

- Synced to `~/cognitiveMapsPlusPlus` on `ciirc-old-cluster` (user `hulajan1`).
- Job submitted via `probe/run_cluster.sbatch` (partition `gpu`, 1x GPU), runs all
  three heads at seed 0; JSONs land in `probe/results/`.
- _(filled after the job completes)_

## Reproduce

```bash
# local (CPU is fine)
python probe/bridged_tori_probe.py --head self_norm   --steps 4000   # PASS
python probe/bridged_tori_probe.py --head cross_scalar --steps 4000   # FAIL (B3 as specified)

# cluster
sbatch probe/run_cluster.sbatch   # from ~/cognitiveMapsPlusPlus
```

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

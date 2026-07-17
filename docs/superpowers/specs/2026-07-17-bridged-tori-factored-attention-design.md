# Bridged Tori — Factored Latent + Attention Distance (Probe)

Date: 2026-07-17
Branch: `bridged-tori-factored-attention`

## Goal

Learn a distance `D(A,B)` over states of a two-torus graph such that `D` matches
the true shortest-path (geodesic) distance, where crossing between tori is only
possible through a single bridge node. Two states with identical position on
different tori must be far apart by the *detour* cost through the bridge, not by a
constant. The distance is computed by a learned attention head over a **factored**
latent, and it must reflect the routing constraint even though supervision is
**local only** (one-step transitions); the global geodesic should *emerge*.

This is an idealized "cognitive map" of a stratified state space (cf. repo README
touching-tori result), extended with actions and an attention-based metric.

## Graph — `bridged_tori`

- Two 15x15 wraparound tori, 225 nodes each = **450 states**.
  State = `(pos in 0..224, torus in {x, y})`.
- Node id convention: `pos = row*15 + col`, moves are 4-neighbor with wraparound.
- **Bridge:** one undirected edge `(150, x) <-> (150, y)`. The tori are otherwise
  disconnected, so every cross-torus path routes through node 150.
- **6 actions:**
  - `up`, `down`, `left`, `right`: grid moves with wraparound (always defined).
  - `next`: at `(150, x) -> (150, y)`; self-loop (stay) everywhere else.
  - `prev`: at `(150, y) -> (150, x)`; self-loop (stay) everywhere else.
- Ground-truth geodesic: exact, via networkx BFS on the 450-node graph.

## Data

- Enumerate all `450 * 6 = 2700` transitions `(state, action, next_state)`,
  including self-loops. In-memory, no path-JSON, no Hydra (probe only).

## Model

- **Factored embedding:** shared position table `E_pos[225]` reused by *both*
  tori + torus table `E_id[2]`. State -> 2 tokens `[t_pos, t_id]` of dim `d_model`.
- **Dynamics net (action enters here):** `f(t_pos, t_id, action) -> (t_pos', t_id')`
  predicts next-state tokens. Keeps the distance head a pure state-state metric so
  `D` can be read out directly for the geodesic check.
- **B3 distance head `D(u,v)`:** input tokens `[t_pos^u, t_id^u, t_pos^v, t_id^v]`
  plus a learned `[DIST]` query token; small transformer encoder (1-2 layers);
  read the `[DIST]` output -> Linear -> `softplus` -> scalar >= 0.
  Symmetrized: `D(u,v) = 0.5*(D_raw(u,v) + D_raw(v,u))`.
  No routing/metric prior baked in (purest test that attention discovers the detour).

## Loss (A1 — local supervision only, geodesic emerges)

- **Dynamics consistency:** `D(f(t^A, a), t^B) ~ 0` for every real transition.
- **Scale anchor:** `D(A, B) ~ 1` for one-step neighbors (non-self-loop transitions).
- **Repulsion (anti-collapse):** random pairs pushed apart via softplus (repo-style).
- **Optional knobs** (off by default; enable only if B3 will not converge, since a
  learned scalar has no metric guarantees): `D(x,x) -> 0` penalty, mild triangle
  inequality regularizer.

## Baseline

- Plain shared `nn.Embedding` + fixed L-p distance, trained with the same local +
  repulsion loss (the repo's current contrastive model), on the same graph. No
  factoring, no attention, no dynamics net.

## Kill criterion (set before running)

PASS if both:
1. **Geodesic fit:** Spearman(`D`, true geodesic) over sampled state pairs beats the
   baseline by a clear margin.
2. **Detour signature:** `D((p,x),(p,y))` grows with `torus_geo(p, 150)` (positive
   rank correlation), rather than being roughly constant across `p`.

Otherwise: pivot the head (try B2 gated / B1 hub-routing, or enable the metric knobs).

## Build order

1. Probe first: one standalone script (`probe/bridged_tori_probe.py`) building the
   graph, transitions, both models, training, and the kill-criterion eval.
2. Run locally (CPU is fine; graph is tiny) to get signal.
3. Run on `ciirc-old-cluster`.
4. Productionize into `generate/graph_types/`, a dataset class, and Hydra configs
   **only after** the probe shows signal (separate plan).

## Outcome (2026-07-17, run overnight)

See `probe/RESULTS.md` for the full write-up. Summary:

- **B3 as literally specified (`cross_scalar`, free scalar cross-attention distance)
  FAILS.** The distance head has no metric prior, so once the pointwise constraints
  (neighbor=1, random pushed to `offset`) tighten, the repulsion term overfits and
  the geodesic *grading* collapses: Spearman peaks ~0.56 at step 400 then decays to
  ~0.13 by step 4000. Detour never appears. This is the risk flagged when B3 was
  chosen.
- **Fix — metric-prior attention readout (`self_norm`): PASS.** Keep the factored
  latent and attention, but read distance as `||e_u - e_v||_p` where `e` is an
  attention-produced (self-attention over the 2 factor tokens) state embedding. The
  norm supplies triangle inequality + identity, which the free scalar lacked.
  Result (seed 0): Spearman(D, geodesic) = 0.86 vs baseline 0.68; **detour signature
  Spearman = 0.90** (the route-through-bridge structure emerges from local-only
  supervision, confirming the A1 claim). Trajectory is stable, no collapse.
- `cross_reg` (B3 + triangle/identity soft regularizers) is the intermediate test of
  whether the *pure* scalar head can be salvaged with the reserved metric knobs.

Design decision recorded: the attention head must carry a metric prior. `self_norm`
is the recommended head to productionize.

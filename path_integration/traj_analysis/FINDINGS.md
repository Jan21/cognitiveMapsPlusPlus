# Trajectory analysis — what the recall-flow distance model does inside

Model: recall-flow integrator, checkpoint `dgx_D20n5.pt` (n=5 factors, cycles C_6, CHAIN guards: factor i can move
only when factor i-1 sits at position 0; trained on geodesic distance <= 20). Analysis = load checkpoint, run the
flow with per-factor trajectory logging on ~1500 (start,goal) pairs. CPU only. Script: `analyze_traj.py`.

## Setup recap
Total predicted distance = softplus(scale) * sum_t sum_i ||dz_i(t)||, i.e. a SUM over factors of each factor's
integrated displacement c_i = softplus(scale) * sum_t ||dz_i(t)||. Question: does c_i reflect the distance factor i
actually has to traverse?

## Findings (run 1, dgx_D20n5)

**F1 — Total cost = geodesic (sanity).** corr(sum_i c_i, true total geodesic) = 1.00, MAE = 0.11 moves. (plot H1 right)

**F2 — Per-factor cost reflects the REAL per-factor traversal, not the naive cycle distance.**
  - corr(c_i, per-factor CYCLE distance d_i=min-arc) = 0.13  (NO)
  - corr(c_i, per-factor GEODESIC-PATH moves)         = 0.94  (YES)
  The two differ because of the guards: an internal factor often must MOVE to position 0 to unlock its descendants
  (chain), then move on, so it takes more steps than its own cycle arc. The model charges that real per-factor
  movement. Evidence: at cycle-distance 0 (factor's start already = its goal) the per-factor cost still ranges 0..6
  (it still moves, to unlock others). The LEAF factor (factor 4, nothing depends on it) only moves for itself, so its
  cost DOES track its cycle distance (corr 0.97); internal factors 0-3 do not (corr ~0). (plot H1 left)
  => ANSWER to the user's hypothesis: yes, the per-factor displacements reflect the per-factor distances that must be
     traversed — specifically the guard-aware geodesic-path moves, NOT the raw cycle distance.

**F3 — The flow JUMPS; it is an amortized oracle, not a step-by-step navigator.**
  Cumulative displacement fraction: ~37% in step 1, ~92% by step 3, ~100% by step 5 (of T_train=6; test budget 200).
  All the "work" happens in the first ~3 steps, then the flow PARKS (adds ~0 more) — which is exactly why budget-free
  extrapolation works. The distance is encoded in the MAGNITUDE of the jump, not in a number of steps. (plot H2 jumpwalk)
  Consistent with the earlier flow-not-navigator result.

**F4 — Decoded position trajectories snap to the goal in ~3 steps, with brief transients.**
  Decoding each factor's token to its nearest position: a 2-4 step transient (some overshoot, several factors dip
  through position 0 = the unlock cell) then a hard snap to the goal position, held for the remaining ~36 steps. So it
  does not walk the cycle; it commits almost immediately. (plot H2 decode_examples)

**F5 — Position embeddings are NOT a clean metric circle.**
  PCA of the 6 position embeddings is scattered, corr(embedding-distance, cycle-distance) = 0.38. So the model does
  not compute distance as Euclidean-on-a-circle; the flow dynamics compute it. (plot H2 posemb)

## Emerging mechanistic picture
The model is an amortized per-factor distance oracle: it snaps each factor's representation toward the goal in ~3
steps, and the SIZE of each factor's jump is calibrated to that factor's true guarded-path traversal (own cycle
distance + any extra moves needed to unlock descendants). Summed over factors this equals the geodesic. It does not
simulate the path; it predicts the per-factor path-length directly.

## Next (run 2+)
- per-factor c_i vs path-moves broken out per factor; decompose c_i into own-distance + unlock-overhead.
- do the first-3-step transients ROUTE through unlock positions in a guard-meaningful way (factor dips to 0 exactly
  when it must unlock a descendant)?
- robustness across checkpoints (other n, image input, DAG guards); recall/arrive ablations.

## Findings (run 2, dgx_D20n5, deeper) -- CONFIRMED + new
F2 refined: per-factor c_i vs its own PATH-moves, broken out PER FACTOR = [0.99,0.98,0.97,0.98,0.97] (all near-perfect).
  vs own CYCLE-dist = [0.00,0.04,0.01,0.00,0.97] -> internal factors decoupled from cycle dist, leaf tracks it.
  Guard overhead (path-moves - cycle-arc) mean per factor = [1.43,1.42,1.36,1.26,0.0]: internal factors take ~1.3-1.4
  EXTRA moves to unlock descendants; leaf 0. Cost tracks overhead too (corr 0.78). Total r=1.0 MAE=0.095.
F6 (NEW, striking) -- the JUMP transient ROUTES THROUGH THE UNLOCK CELL when the guard needs it:
  P(factor visits position 0 during its first ~5 transient steps | it must unlock a descendant) = 0.96
  P(... | it does NOT need to unlock)                                                            = 0.36
  So the flow is not a blind jump: its brief transient passes through the unlock position exactly when the chain guard
  requires it, then snaps to the goal. (This model DOES encode guard-routing internally, unlike the earlier compose flow.)
MECHANISM (updated): amortized per-factor distance oracle. In ~3 steps each factor snaps to its goal; the jump MAGNITUDE
is calibrated to that factor own guarded-path traversal (own arc + unlock overhead), and the transient visits the unlock
cell when a descendant must be freed. Sum over factors = geodesic. Position embeddings non-metric (0.38) -> distance is
computed by the flow dynamics, not embedding geometry.

## Run 3 (robustness). IMAGE input entangles the per-factor decomposition.
w11_img_n5 (image,n5): total r=0.89 MAE=1.14; per-factor c_i vs path-moves = 0.24 (weak) [0.77,0.41,0.32,0.46,0.65].
w10_dagimg (image,dag,n4): total r=0.99; per-factor = 0.43 (weak) [0.69,0.09,0.05,0.22].
=> vs the clean INDEX model (0.94), IMAGE models achieve total accuracy but DONT cleanly decompose per-factor -- the

## Run 3 (robustness). IMAGE input entangles the per-factor decomposition.
w11_img_n5 (image,n5): total r=0.89 MAE=1.14; per-factor c_i vs path-moves = 0.24 (weak) [0.77,0.41,0.32,0.46,0.65].
w10_dagimg (image,dag,n4): total r=0.99; per-factor = 0.43 (weak) [0.69,0.09,0.05,0.22].
=> vs the clean INDEX model (0.94), IMAGE models achieve total accuracy but do NOT cleanly decompose per-factor; the
metric is entangled across factors (matches image-input-shared-canvas). NOTE: the decode-based unlock probe (visit0) is
INVALID for image models -- they encode via a CNN, not the pos embeddings the decoder assumes -- so visit0=0 is an
artifact there. The per-factor COST metric does not use decode and is valid. Clean-index robustness running next.

## Run 4 (index robustness) -- KEY: two internal solutions (clean vs entangled), same total accuracy.
All predict total geodesic perfectly (r=1.0), but per-factor decomposition varies:
  dgx_D20n5  : c_i vs path-moves 0.94  unlock-routing 0.96/0.36  -> CLEAN
  w10_n5D24  : 0.96  unlock 0.98/0.41                            -> CLEAN
  w10_n5D18  : 0.47  unlock 0.49/0.45 (none)                     -> ENTANGLED
  w10_n6D14  : 0.27  unlock 0.48/0.39 (none)                     -> ENTANGLED
=> Getting the TOTAL right does NOT imply a clean per-factor decomposition. Two mechanisms exist:
   (1) DISENTANGLED: each factor cost = its true guarded-path traversal, and the transient routes through the unlock
       cell (visit0 0.96-0.98 when a descendant must be freed).
   (2) ENTANGLED: total exact but per-factor costs redistributed; NO unlock-routing signal.
The unlock-routing gap (needs vs not) cleanly discriminates: ~0.6 gap = clean, ~0.05 = entangled.
OPEN: what determines which? seed? D? Ttest? convergence? -> testing seeds of the same config next.

## Run 5 (seed test, n5 D18) -- clean/entangled is SEED-DEPENDENT; and clean-cost vs unlock-routing are SEPARABLE.
Same config (n5 D18), 3 seeds:
  s1: total 1.00, per-factor 0.43 (entangled),   unlock 0.69/0.53 (weak)
  s2: total 1.00, per-factor 0.97 (CLEAN),        unlock 0.98/0.52 (routes)
  s3: total 1.00, per-factor 0.99 (CLEAN),        unlock 0.27/0.22 (does NOT route)
=> (a) whether the per-factor decomposition is clean is a SEED/training-run property, not fixed by config.
   (b) "clean per-factor cost" and "transient routes through the unlock cell" are SEPARABLE: s3 assigns correct
       per-factor costs WITHOUT its transient visiting the unlock position. So there are >=3 internal variants at the
       same perfect total accuracy: entangled / clean-no-route / clean-and-route.

## Run 6 (mechanism vs capability). Clean-ness does NOT strongly drive extrapolation.
Across 7 index checkpoints: corr(per-factor clean-ness, extrapolation MAE beyond train D) = -0.32 (WEAK).
An ENTANGLED model (w10_n5D18, pf 0.47) extrapolates as well (far-MAE 0.11) as CLEAN ones. So the clean per-factor
decomposition + unlock-routing are REAL but NOT load-bearing for the headline capability: the model predicts the
total distance and extrapolates flatly to ~2xD whether or not it decomposes cleanly per-factor. Table+plot:
traj_analysis/clean_summary.json, clean_vs_extrap.png.
CONSOLIDATED ANSWER to the user's hypothesis: per-factor cost CAN reflect the true per-factor guarded-path traversal
(clean models, corr 0.94-0.99, incl. guard-overhead), but this is a SEED-dependent internal property, not universal
and not required for accuracy. The flow is an amortized oracle (jumps in ~3 steps); clean models' transients route
through the unlock cell (0.96) but that too is separable and non-essential. Image input entangles the decomposition.

## Run 7 (Wave-1 checkpoints, D=6 early) -- clean decomposition needs training maturity; seed-variable among mature.
All w1_* (D6, early wave): per-factor clean-ness 0.11-0.28 (entangled) AND poor extrapolation (far-MAE 5-8, flat ~8-13).
=> undertrained/small-range models are BOTH entangled AND weak extrapolators.
Well-trained models (D18-24) all extrapolate well (far-MAE 0.1-0.5); clean-ness VARIES by seed (0.27-0.99).
FULL PICTURE: (1) competence (total geodesic + flat extrapolation to ~2xD) needs enough training/D. (2) Among competent
models the INTERNAL per-factor decomposition ranges from entangled to clean, seed-dependent. (3) Clean-ness is NOT
load-bearing for competence (competent-entangled models extrapolate as well as competent-clean). The clean models are
the ones where the user's hypothesis holds: per-factor cost = true guarded-path traversal (0.94-0.99, incl unlock
overhead), transient routes through the unlock cell. Prevalence among mature n5 models ~ half (4 clean / 3 entangled here).

## Run 8 (order + calibration, clean checkpoints) -- refinement: spatial routing, NOT temporal ordering.
Calibration: c_i vs path-moves slope = 0.81 (dgx_D20n5) / 0.89 (w11_n5D18_s2); per-factor 0.76-0.99. So each factor
cost is a near-linear count of its true path-moves (slightly under 1:1).
Topological ORDER: parent reaches the unlock cell BEFORE the child first moves only 17-25% of the time; median step-gap = 0.
=> The transient is NOT a temporally-ordered sequential simulation. The parent visiting the unlock cell and the child
moving happen in the SAME ~3-step jump. So the earlier unlock-routing (0.96 visit-rate) is SPATIAL (the token passes
through the unlock position) but NOT temporal -- the flow amortizes the whole guarded path into one simultaneous jump
whose geometry routes tokens through unlock cells; it does not re-enact the path step by step. This is consistent with
the amortized-oracle picture and slightly walks back any "compressed sequential simulation" reading.

## Run 9 (entangled mechanism) -- the two solutions fully characterized.
Per-factor cost correlation with (own path-moves) vs (total geodesic):
  CLEAN (D24, D18s3):      own path-moves 0.94-0.99   total 0.45-0.59   -> each factor computes ITS OWN traversal.
  ENTANGLED (D18, D18s1):  own path-moves 0.25-0.82   total 0.32-0.74   -> cost SMEARED: each factor cost tracks the
                                                                            GLOBAL total about as much as its own moves.
So the entangled solution distributes/shares the cost across factors (partly reflecting the overall distance), giving
an exact SUM without a clean per-factor attribution. The clean solution assigns each factor exactly its own traversal.
Both reach total r=1.0 and flat extrapolation. Which one a training run lands in is seed-dependent (see run 5).

## INVESTIGATION SUMMARY (F1-F10)
The recall-flow distance model is an AMORTIZED per-factor distance oracle:
 - Total predicted distance = sum of per-factor displacements = true geodesic (r=1.0, ~0.1 move err), always.
 - It JUMPS: ~90% of internal movement in the first ~3 steps, then parks (why it extrapolates budget-free).
 - In CLEAN models (~half of well-trained n5 runs), each factor displacement = its true guarded-path traversal
   (corr 0.94-0.99, calibration slope ~0.85, includes ~1.3 extra unlock moves for internal factors). The transient
   SPATIALLY routes tokens through the unlock cell when a descendant must be freed (0.96 vs 0.36) -- but NOT in temporal
   order (parent-before-child only 17-25%): it is one simultaneous amortized jump, not a step-by-step re-enactment.
 - In ENTANGLED models the per-factor cost is smeared (tracks the global total); total still exact.
 - Distance is computed by the flow DYNAMICS, not embedding geometry (pos embeddings non-metric, r=0.38).
 - Clean-ness is SEED-DEPENDENT, needs training maturity, and is NOT load-bearing (entangled models extrapolate as well).
 - IMAGE input entangles the per-factor decomposition further (0.24 vs 0.94 index).
ANSWER to the user hypothesis: YES, per-factor cost reflects the true per-factor guarded traversal -- in the clean
solutions (a common but not universal, seed-dependent internal structure). Report: traj_analysis/traj_mechanism_report.html

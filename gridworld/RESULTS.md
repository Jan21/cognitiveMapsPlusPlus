# Gridworld geodesic probe -- results

Keys-&-doors guarded gridworld (fixed map), factors [x,y,keybits] read the top-down
rendered image; recall flow + arrive gate; cost=integrated displacement -> BFS geodesic;
budget-free fixed Ttest. flat = largest dist with by_d MAE<1.5.

## Wave gw / gw2d (fixed map): geodesic learned near-perfectly in-range; flat-extrap LIMITED (~1.5xD); D-lever extends.
  xattn 8x8 D6 T22:  near 0.04  flat_d9  (d7=0.2; grows: d12=3.6 d19=7.8)
  xattn 8x8 D8 T22:  near 0.05  flat_d11-14 (gw2d_0 d19=1.4 best; seed variance)
  xattn 8x8 D8 T16:  near 0.04  flat_d10  (tight Ttest ~same)
  xattn 8x8 D10 T28: near 0.12  flat_d13 (d19=3.8 -- D-lever extends far range)
  xattn 10x10 K2 D8: near 0.08  flat_d11 (2 keys/doors WORK; md32 grows far)
  convspatial D8:    near 0.05  flat_d11-13 (~= xattn)
  convpool D8:       near 0.09  flat_d12 (slightly worse far, d19=5.6)
  recall=0 D8:       near 0.38  flat_d14 (near ~8x WORSE than recall=1 -> recall helps in-range a lot;
                                          oddly far plateaus ~2.0 vs recall's growth -> recall/over-integration interact)
FINDINGS:
  (1) Gridworld geodesic (with guarded doors) is LEARNED near-perfectly in-range: near 0.04-0.12, BETTER than abstract factors.
  (2) Budget-free flat-extrapolation WORKS but is MORE LIMITED than abstract factors: flat ~1.3-1.5xD then grows.
      Spatial geodesic (walls/detours/key-fetch) is harder to extrapolate than product-graph geodesic.
  (3) D-LEVER still extends the flat range (D10 far << D6 far). Pushing D (Wave gw3) to test full-range.
  (4) ALL 3 READOUTS work; xattn ~= convspatial > convpool. Convolutional factor encoding is viable.
  (5) RECALL helps in-range accuracy strongly (0.05 vs 0.38). K2 (2 guarded doors) works.
  PENDING: random-layout (gw2) = generalization to UNSEEN maps (the big test), still running.

## Wave gw2/gw3/gw2d: D-LEVER SOLVES fixed-map extrapolation (FULL RANGE); random-layout GENERALIZES to unseen maps.
FIXED-MAP -- D-lever gives FULL-RANGE flat (earlier ~1.5xD limit was just D<<diameter):
  8x8 (maxd19): D10 near0.05 flat_d19 FULL; D12 near0.01 flat_d19 FULL; D14 near0.03 flat_d19 FULL.
  convspatial D12: near0.33 flat_d19 FULL (conv readout also full-range).
  12x12 (maxd31): D10 flat_d15 (needs higher D, ~diameter); 10x10-K2 (maxd32): D8 flat_d11 (needs higher D).
  => RULE: train D ~ graph DIAMETER -> full-range flat, near 0.01-0.08. Same D-lever as abstract factors. EXTRAP SOLVED.
RANDOM-LAYOUT (fresh map per instance, held-out maps at test = generalization to UNSEEN geometry):
  xattn 8x8 D8: near 0.14 flat_d10 (GENERALIZES to unseen maps in-range! reads walls/doors/agent from pixels).
  convspatial: near 0.26 ; convpool: near 0.96 (READOUT MATTERS on random: xattn >> convpool; attention needed for spatial generalization).
  random 10x10 D10 near0.20; random 10x10 K2 D10 near0.25 (2 doors + unseen maps works in-range).
  recall0 near0.16 vs recall1 0.14 (recall small effect on random, unlike fixed where recall0=0.38 vs 0.05).
  random EXTRAP still short (flat_d10-12) because D8-10 << diameter(29-52) -> apply D-lever on random (Wave gw5).
HEADLINE: (1) D-lever (D~diameter) -> FULL-RANGE flat gridworld geodesic, near ~0.01-0.08. (2) Model GENERALIZES across
  UNSEEN random maps (near 0.14, xattn), reading geometry from pixels. (3) xattn best readout for generalization.

## Wave gw3/gw4: extrapolation fix = STEPS + D-LEVER (not anytime/capacity). Confirmed at scale (12x12).
  8x8 D8 90k: near 0.03 flat_d19 FULL  <- MORE STEPS alone fixes D8 (was flat_d11-14 at 30-40k = UNDERTRAINING).
  12x12 D22 Tt48: near 0.09 flat_d30 FULL (maxd31)  <- D-lever at scale: D~diameter -> full range on big grid.
  12x12 D14: flat_d22 (d31=5.3, partial) -- D14 < diameter31, consistent w/ D~diameter rule.
  gw4 levers @ D8 60k: anytime(Tmin4 T22) flat_d12; layers2 flat_d12 -- ~= D8/60k baseline. NOT a big help.
    => ANYTIME + CAPACITY are MINOR; the real fix is ENOUGH STEPS (90k) and/or D~DIAMETER. Both give full-range.
  convspatial D12: full-range but near 0.33 (worse near than xattn 0.01) -- xattn best readout.
ANSWER to "improve flat extrapolation": train LONGER (90k) and/or with D ~ graph DIAMETER. Either -> full-range flat
  gridworld geodesic, near 0.01-0.09. Same two levers as abstract factors. Fancy levers (anytime/layers) unneeded.

## Wave gw5: CAPSTONE -- D-lever gives FULL-RANGE flat on UNSEEN random maps.
  random xattn D24 8x8: near 0.39 flat_d29 FULL (maxd30). Generalizes across UNSEEN maps AND full-range extrapolates.
  random xattn D20: near 0.26 flat_d27 (nearly full). random convspatial D20: near 0.84 flat_d16 (xattn > conv on random).
  gw4 combos: layers2+90k -> d19=0.3; anytime+layers2 -> d19=0.8 (help somewhat, but D/steps are the clean levers).
COMPLETE GRIDWORLD STORY: keys-&-doors guarded gridworld, factors [x,y,keybits] cross-attend rendered image, recall flow.
  (1) fixed map: full-range flat via D~diameter OR 90k steps (8x8 near0.01, 12x12 D22 near0.09).
  (2) UNSEEN random maps: generalizes (near0.14) + full-range flat at D~diameter (near0.39). Reads geometry from pixels.
  (3) xattn best readout (needed for spatial generalization; convpool fails 0.96); recall load-bearing on fixed; K2 works.
  Reported in docs/recall_scaling.html Result 4 + chart. Two levers (steps, D~diameter) same as abstract factors.

## Wave gw6/gw7/gw7d/gw8: reliability + harder guards + scale.
  RANDOM D24 generalization RELIABLE across seeds1-5: near 0.35/0.38/0.39/0.45 (+s0 0.39), flat_d24-26 (~full, md29-31). Stable capstone.
  SEQ-GUARD (door_i needs keys 0..i): fixed seq-K3 D12 near 0.06 in-range (learnable); flat_d16 (D12<<diameter36 -> needs D~diameter for full-range).
  10x10 random: D20 near0.34 flat_d23, D18 near0.30 flat_d22 (md36-37); bigger grids need higher D for full-range.
  READOUT on random high-D: xattn 0.35-0.45 (best) > convspatial 0.62-1.19 (seed-variable) > convpool (fails). conv catches up SOME at high D but xattn wins.
  convpool FIXED D12: near0.04 flat_d19 FULL (conv fine full-range on FIXED; only fails on random=generalization).
  D10/D12 fixed seed1: full-range (reliable).

## Wave gw6/gw7/gw8: D-lever scales to 16x16; K-scaling = diameter-bound; reliability continues.
  16x16 fixed K1 D30: near 0.09 flat_d43 FULL (md43). D-LEVER SCALES to big grids (full-range).
  12x12 random D26: near 0.36 flat_d32 (md50, partial -- D26<diameter50). random generalizes at 12x12.
  random D24 s5: near 0.36 flat_d27 FULL. Reliability continues (seeds 1-5 all near 0.35-0.45).
  K-scaling: more doors -> bigger diameter -> needs proportionally higher D. K2 random 10x10 D20 near0.55 (md56 partial);
    K3 random 10x10 D16 near0.67 (md71 partial); K3 fixed 10x10 near0.10-0.11 in-range. Guard depth learnable; full-range needs D~diameter.
  convspatial random 10x10 D20: near 0.58 (xattn 0.34 better). xattn best on random confirmed at scale.

## COMPOSITIONAL (comp_probe.py): does the model compute distance for NOVEL mechanic combos?
Setup: 5 mechanics as automata (key/button/twokey/portal/slide), SUBSET present per instance; model reads
present mechanics from image (cross-attn); trained on some presence-combos, tested on HELD-OUT combos.
--factored 1 = additive per-factor cost (sum ||dz_i||); --factored 0 = joint MLP readout baseline.

EASY split (27 train combos, 4 held out): COMPOSES WELL. factored 100k heldout_mae 0.62 (train 0.17).
  -> when the model sees enough combos, it generalizes to novel ones cleanly.
HARD split (train = singletons + 3 pairs = 8 subsets; 23 HELD-OUT combos): partial composition, and SURPRISE:
  factored heldout 2.05/2.66/1.74 (train ~0.13); JOINT heldout 1.61/1.33/1.05 (train ~0.2).
  => JOINT BASELINE COMPOSES BETTER than the additive-cost FACTORED (3 seeds, consistent).
  HYPOTHESIS FALSIFIED: the additive per-factor cost does NOT aid compositional generalization here; it slightly HURTS.
  Reason: even "independent" rooms share the base corridor traversal, so geodesic(combo) != exact sum of singleton costs;
  the rigid additive bias mis-sums, while the flexible joint readout interpolates novel combos better.
  Per-combo: pairs generalize best (~1-1.5), the all-5 combo worst (factored 2.8-3.9, joint 1.7-2.7). More components = harder.
TAKEAWAY: (1) compositional generalization to novel mechanic combos WORKS given enough training combos (easy: 0.62).
  (2) Forcing composition from mostly-singletons is hard for both; the factored/additive structure is NOT the win here
  (joint >= factored). Interacting mode (--mode cross, geodesic=MIN not sum) results pending (cluster SSH was down).

## COMPOSITIONAL -- interacting vs independent (the boundary). ALL results in:
INDEPENDENT rooms (geodesic = SUM of per-room costs):
  easy split: factored heldout 0.43-0.68, joint 0.63-0.74. COMPOSES (novel combos ~= train).
  hard split: factored 1.7-2.7, joint 1.0-1.6 (partial; joint > factored).
INTERACTING cross (geodesic = MIN over crossings, shared obstacle):
  easy split: factored 2.47-2.67, joint 2.51 (train 0.25) -> ~10x gap. DOES NOT COMPOSE (even easy).
  hard split: factored 4.35-4.84, joint 3.48-4.17 (train 0.06-0.11). Barely composes. joint > factored.
  -> fits training combos (train low) but CANNOT generalize the INTERACTION to unseen combos.
FINDING (clean boundary): compositional generalization to novel mechanic combos WORKS when the combined
  distance is a SUM of independent parts (independent rooms: 0.6), and FAILS when components INTERACT
  (min over crossings: 2.5-4.8) -- the interaction (which crossing wins) cannot be inferred from mechanics
  seen only separately. Factored/additive cost is NOT the driver anywhere (joint >= factored).

## Compositional seed-3 confirmation: cross(interacting) hard 3.80/3.94, easy 2.58/2.59 (factored/joint ~= identical).
  Robust: interacting FAILS to compose (2.5-3.9), factored ~= joint (no additive advantage). Medium-split curve pending (comp2).

## Compositional FIX attempts 1-2 on interacting (cross) FAILED. Escalating to aux supervision.
  disentangle+sum: heldout 5.98 (WORSE than baseline 3.85). disentangle+mixed-masked: 3.98 (=baseline).
  mixmask-only: 3.56 (marginal). None near train (0.1-0.4). Disentangle HURTS; structured agg doesn't fix.
  Root cause: the flow's per-factor cost c_i is NOT calibrated to mechanism i's actual solo-crossing distance,
  so masked-softmin(c_i) has nothing meaningful to minimize. FIX = AUX SUPERVISION: force c_i -> solo_i
  (distance via ONLY mechanism i), learnable from singletons; then softmin(c_i)=min solo=interacting geodesic.

## Compositional: MEDIUM curve + TRAINMOST control => interacting failure is FUNDAMENTAL, not data.
  ROOMS (independent, sum): easy 0.6, MEDIUM heldout 1.09-1.32, hard partial. Composes across the curve.
  CROSS (interacting, min): easy 2.5, MEDIUM 3.48-3.79, hard 4-5. FAILS across the curve. factored~=joint.
  TRAINMOST CONTROL (train ~all 28 combos, hold out only 3): cross heldout 2.58-2.86 -- STILL FAILS.
    => NOT data-limited: even seeing every OTHER combo, held-out combos fail. Model fits any TRAIN combo (0.02-0.28)
       but cannot generalize the MIN to an unseen combo. Interacting min-composition = FUNDAMENTAL failure of this flow.
  Fix ablations (cross-hard): disentangle+sum 7.24 (HURTS), disentangle+mixed 4.8-5.3, mixmask 3.96, aux+solomin 4.4-5.
    NONE works. Last test = --indep (per-factor independent flow) running on cluster (113197); if it fails too -> boundary confirmed.

## COMPOSITIONAL — FINAL VERDICT: interacting (MIN) composition is a real BOUNDARY.
  indep (per-factor INDEPENDENT flow, the 6th/last fix): cross-hard heldout 4.0-4.5 (train 0.2-0.8) -- FAILS like the rest.
  ALL 6 fixes fail (disentangle+mixed/sum, mixmask, aux+solomin, indep). + trainmost control fundamental (2.6 w/ ~all combos).
  CONCLUSION: SUM-structured (independent) components COMPOSE to novel combos (0.6-1.1). MIN-structured (interacting,
  shortest-of-alternatives) does NOT -- a single-integrated-trajectory distance flow cannot represent a min over
  alternatives never seen together. Boundary is fundamental, not data/optimization. STOPPED forcing fixes.
  Next-if-wanted: explicit per-mechanism distance predictors + hard min (composes by construction, gives up single flow).

## COMPOSITIONAL — CORRECTED TASK: INTERACTING COMPOSES (the "boundary" was the task bug).
On the well-posed cross task (goal=POSITION, geodesic=clean min over crossings, correct aux targets):
  HEADS (per-mechanism distance regressor + disentangle + aux->solo + softmin): cross-HARD heldout 0.98-1.47,
    EASY 0.36, TRAINMOST 0.50 (was 2.6 on ill-posed!). COMPOSES. (@20-30k steps, still improving to 55-60k.)
  aux+solomin FLOW +disentangle: heldout 1.9 (partial -- flow weaker than heads).
  BASELINE sum-flow (no aux): heldout 5.48 -- fails (a sum-flow cannot represent a MIN over crossings).
=> INTERACTING components DO compose to novel combos, given: (1) well-posed min task, (2) per-mechanism DISENTANGLED
   distance estimation (heads or independent flow), (3) AUX supervision to per-mechanism solo costs, (4) softmin aggregation.
   The earlier 6-fix "fundamental boundary" was ENTIRELY the ill-posed goal-as-state task (bits entangled mechanics,
   aux targets wrong in 44%). Verification (min(solo)==geodesic?) caught it before a false conclusion.

## Compositional interacting CONVERGED finals (corrected task):
  flow-solomin+aux: EASY heldout 0.44, TRAINMOST 0.48 (control -- was 2.6 on ill-posed task) -> COMPOSES.
    HARD (singletons-only) 2.25 (partial; disentangle/heads help here).
  baseline sum-flow (no aux): HARD 5.0 -- FAILS (sum cannot represent min). Confirms aux+softmin is what enables it.
  heads (per-mechanism regressor+disentangle+aux) hard: ~1.0 (finals landing). BEST on the hardest split.
  => Interacting components COMPOSE to novel combos. Easy/trainmost 0.44-0.48; hard(singletons) ~1-2 (approach-dependent).

## Compositional interacting -- HEADS converged finals (corrected task): COMPOSES ALL SPLITS.
  heads (per-mech regressor + disentangle + aux) HARD: heldout 1.03 (aux2.0) / 1.38 (aux1.0); EASY 0.28.
  flow-solomin+aux: EASY 0.43, TRAINMOST 0.48, HARD 2.25 (partial). baseline sum HARD 5.01 (fails).
  FINAL: interacting (MIN) components compose to novel combos. HEADS best (hard ~1.0, easy 0.28); flow-solomin
  works easy/trainmost (0.4-0.5) partial hard (2.25); a plain sum-flow cannot (5.0). Independent(SUM) composes 0.6.

## Compositional ROBUSTNESS (conf, corrected task): reproducible across seeds.
  heads hard: seeds 0/2/3 -> heldout 1.36 / 1.64 / 1.66 (train ~0.06). REPRODUCIBLE (composes ~1.4-1.7).
  heads medium: 1.21. INDEP-flow (per-factor independent flow) hard: 1.20 -- ALSO composes (flow version that works).
  flow-solomin JOINT (disentangle, no indep): 2.17 -- partial (needs per-factor independence: heads or indep-flow).
  => Interacting composition is ROBUST across seeds with heads OR per-factor-independent flow (~1.2-1.7 hard).
     Key ingredient = PER-FACTOR INDEPENDENCE (each mechanism scored without the others) + aux-to-solo + softmin.

## DEPENDENCY-DAG (dag_probe.py): recall-flow COMPOSES to NOVEL dependency chains. (@20-28k, converging)
  Forced-corridor chains (unique path -> distance = clean SUM, no min). Test = held-out chains (novel dependency compositions).
  L2 nheld4: train 0.14 heldout 0.28. L2 nheld6 (incl portal>key): 0.35/0.46. L2 s1: 0.12/0.30. L2 s2: 0.23/0.42.
  L3 (depth-3) nheld8: train 0.20 heldout 0.20 (!). L3 nheld12: 0.31/0.34.
  => heldout ~= train, both LOW (<0.5). Compositional generalization over DEPENDENCY STRUCTURE works: the whole-grid
     recall-flow reads a NOVEL chain (e.g. portal->key never seen) and computes its distance. Even depth-3 DAGs compose.
     No per-mechanism structure needed -- the unique-path SUM regime composes (as expected) AND generalizes to novel DAGs.
  User's insight validated: unique path (forced corridor) avoids the min; DAG-of-dependencies + held-out chains =
     a clean, strong compositional-generalization test that PASSES.

## DAG chains CONVERGED: L3 heldout 0.12 (nheld12) / 0.23 (nheld8) -- composes down to train error. L2 0.27-0.33.
  Confirms: recall-flow composes novel dependency chains, depth-2 and depth-3, reproducibly. STRESS(L4/hold-most)+TREES(twolock) pending.

## DAG stress + trees: TREES compose; HOLD-OUT-MOST is the frontier (coverage threshold).
  TREES (twolock branching) L2 nheld6: heldout 0.44-0.56 (3 seeds) -- branching composes, ~= chains.
  HOLD-OUT-MOST (train FEW chain-types): L2 nheld18(train~7) heldout 1.25 (partial); L2 nheld12(train~4) heldout 3.59-3.80 (FAILS, train 0.10).
  => Frontier is TRAINING COVERAGE, not depth/branching: composition needs enough distinct chains to learn each link's
     function; from a handful of chains it cannot infer the primitives -> fails. Depth(L2/L3) + trees do NOT break it.
  Pending: depth-4 (dags L4), L3 hold-out-most.

## DAG frontier refined: coverage threshold ~= ABSOLUTE #train-chains (~15-20), not depth/fraction.
  L2 train~4 heldout 3.8 (fail); train~7 -> 1.25 (partial); train~19 -> 0.44-0.56 (composes).
  L3 nheld48 train~16-77 -> 0.23-0.27 (composes even hold-out-most, because #train >= threshold).
  TREES: L3 nheld100 train~25 -> 0.49; L3 nheld15 train~110 -> 0.30 (compose).
  => Need ~15-20+ DISTINCT training chains to learn the per-link primitives; below that fails. Depth(L2/L3)+trees OK given coverage.
  LAUNCHED length-generalization sweep (dagL 113245: train len<=Ltrain, test LONGER chains, Lmax 3-6) -- the key axis.

## Length-gen EARLY (undertrained 16-28k): STRUGGLING. train<=2 test3 heldout 4.25; test3-4 6.25; train<=3 test4-5 9.56 (train 0.3-0.7).
  Longer test chains -> higher error. Looks like DISTANCE-EXTRAP limit (longer chain=much bigger distance, far past trained range).
  Testing D-lever analog: train len CLOSER to test (Ltrain4 test5, +1 gap) -- does small-gap length-gen work? (dagL2 sweep)

## DAG scaling axes (separate): #MECHANISMS scales GREAT; LENGTH fails.
  #MECHANISMS (vocab nreach+nlock, fixed L2): nv4 heldout 0.33, nv6 0.14, nv8 0.10, nv12 0.09 (train 0.08).
    -> composition IMPROVES with more mechanism types (more coverage + cleaner per-mechanism funcs). Scales.
  LENGTH-GEN (train len<=Ltrain, test longer): Ltrain2->Lmax3 heldout 4.3-8.1; ->Lmax4 len3=3.3 len4=8.5 (train 0.1-0.3).
    -> FAILS: no extrapolation to LONGER chains, error GROWS with test length. The whole-grid recall-flow (fixed reasoning
       tokens) cannot extend to more dependency links. Closer-gap probes (Ltrain4->Lmax5) running (dagL2). Likely needs a
       DEPTH-RECURRENT architecture (process chain link-by-link -> more links=more steps) to length-generalize.
  VERDICT: mechanism-count axis scales; length/depth axis is the hard one (structural, needs recurrence over chain depth).

## DAG axes UPDATE: #mechanisms scales to L3; length fails ALL gaps (even +1 link).
  VOCAB: L2 nv4-12 heldout 0.09-0.33; L3 nv6 0.09, nv8 0.12. Mechanism-count scales L2+L3 (heldout ~0.1).
  LENGTH: Ltr3->L4 (+1 link) len4=5.3 FAILS; Ltr3->L5 len4=5.7 len5=13.3; all gaps fail, error grows w/ length.
    => whole-grid recall-flow structurally cant add even ONE dependency link. Length needs DEPTH-RECURRENT model.
  IMPLEMENTED --arch scan (per-region shared cost head -> sum over region-slots -> should length-generalize). Testing.

## LENGTH-GEN SOLVED by --arch scan (depth-recurrent per-region sum)!
  scan train<=2 test len3-4: heldout 1.27 @ONLY 4k steps (vs whole-grid flow 6-9 at convergence). len4 chain=1.30.
  => processing region-by-region with a SHARED per-region cost head SUMS -> naturally extends to more links -> length-generalizes.
  Launched scanL sweep: train<=2 test up to len6/8 (3-4x training length), + Ltrain3. Does it extrapolate FAR in depth?

## LENGTH-GEN SOLVED (scan) + FACTORS DEBUG clarifies everything.
SCAN grid length-gen (train len<=Ltrain, test LONGER): Ltr2->L8 heldout 0.45 (len3=0.3..len8=0.6), Ltr2->L6 0.41,
  Ltr3->L6 0.37. REPRODUCIBLE (seeds). Trained on 2-link chains -> predicts 8-link chain distance at ~0.6 (4x depth!).
  vs whole-grid flow FAILED (5-9). => per-REGION shared cost head + SUM generalizes to more links.
FACTORS-ONLY debug (tokens, no grid; gridworld/factor_probe.py):
  COMBO-gen (unseen mechanism combinations, slots): sum heldout 1.1, MIN mode 0.01 (perfect). Pool also ~1.3.
    => composition to unseen combos is EASY with token input. The grid's interacting-MIN difficulty was IMAGE-READING
       (localizing each crossing's cost from pixels), NOT the min-composition (trivial in factors).
  LENGTH-gen (seq): per-element SUM handles structure (heldout 3-5, scale=big synthetic costs); POOL FAILS (16.5).
    => aggregation must be a per-ELEMENT SUM, not a fixed-capacity pool. Confirms scan principle.
VERDICT: length/depth generalization SOLVED by depth-recurrent per-element-sum (scan). Combo generalization works.
  Next: verify scan also composes to unseen COMBINATIONS (combine axes), then move to image settings.

## COMBINE AXES (scanC): EVERYTHING STACKS. scan solves dependency-DAG generalization across all axes.
  LENGTH + MECHANISMS: scan vocab8 Ltr2->L8 heldout 0.24 (L8=0.6); vocab12 Ltr2->L8 0.33; vocab8 Ltr3->L8 0.29.
    -> length-gen HOLDS (even improves) with more mechanisms. len-gen to 8 links (4x depth) + vocab8/12.
  UNSEEN COMBINATIONS (scan, fixed L): vocab12 L3 nheld60 heldout 0.15=train; vocab16 L2 nheld40 0.11=train.
    -> scan ALSO composes to unseen mechanism combinations (heldout=train).
  => The depth-recurrent per-region-SUM (scan) generalizes across LENGTH (4x depth) + #MECHANISMS (16+) +
     NOVEL COMBINATIONS, all simultaneously, heldout~=train. Full dependency-DAG compositional generalization SOLVED.

## Factored 3-axis push (library size / instance count / length) + base-constant fix  [2026-07]

Harness: `gridworld/factor_probe.py`. Mechanisms as TOKENS (no grid). target = base(3.0) + sum/min of (C_type+param).
Axes: (a) LIBRARY size K = #types; (b) COUNT-gen = train |present|<=Ctrain, test MORE present (`--split count`);
(c) LENGTH = seq repr, train len<=Ltrain, test longer.

### Library-size axis (combo composition scales with K) -- WORKS
Unseen presence-COMBINATIONS, sum arch: K6 heldout 1.20, K10 0.54, K12 0.77. MIN mode K10 = 0.01.
=> Composing novel combinations holds as the mechanism library grows (K6->K12). Min is trivially clean.

### Count-gen & Length-gen "failure" was a BASE-CONSTANT ARTIFACT (not a composition limit)
Plain sum (`c.sum(-1)`, target = base + sum) FAILS to extrapolate to more elements than trained:
  COUNT K8 Ctr2: 3.6-4.1 | K16: 5.2 | K32: 4.9 | Ctr3: 2.8 ; LENGTH K8 Ltr2->6: 4.6-5.4 ; POOL(mean): 14+
Root cause: the constant base(3.0) gets folded into per-token costs (~base/N each) and summed N times, so at
test N (more present/longer) it overshoots by ~(N - Ntrain_avg)*offset ~= 4-5. Confirmed by MIN mode being clean
(0.01): adding base to every token then taking min still = base+min, count-independent; summing multiplies it.

FIX (`--globalbase 1`): add base ONCE via a scalar param, per-token head outputs cost only. Then per-element SUM
count- AND length-generalizes cleanly:
| axis                              | globalbase=0 | globalbase=1 |
|-----------------------------------|--------------|--------------|
| COUNT K8 (train<=2 -> test 3-6)   | 3.62         | **0.23**     |
| LENGTH K8 (train<=2 -> test 6)    | 5.36         | **0.001**    |
Length is essentially exact (0.001). Same lesson as the interacting-min false boundary: root-cause the negative
before declaring a wall -- the "count/length extrapolation limit" was a target-encoding artifact.
Repro: `python gridworld/factor_probe.py --repr seq --arch sum --globalbase 1 --K 8 --Ltrain 2 --Lmax 6 --steps 8000`

### GRID scan length-push (dag_probe --arch scan): 8x training depth
Train len-2 chains, test longer, depth-recurrent per-region SUM head:
  L2->12 vocab8 HELDOUT 0.23 (maxd 142) | L2->16 vocab8 HELDOUT 0.23-0.25 (maxd 165-175, **8x depth**)
  L2->14 vocab12 0.26 | L2->16 Ltr3 0.45 | in-range vocab24/32 0.10-0.23
=> The grid scan reads a novel corridor and sums region costs out to 8x the trained chain length.

### CONFIRMED at scale (globalbase=1, 40-45k steps) -- SUM generalizes, POOL is the real limit
COUNT-gen (train |present|<=2, test more): K8 0.002 (2 seeds 0.001-0.002) | K16 0.005 | K32 0.003 | K64 **0.002**
  => scales to a 64-type library with NO degradation. Ctr3 0.22.
LENGTH-gen (train len<=2, test longer): Lmax6 0.04 | Lmax10 0.05 | Lmax14 0.001 | Lmax20 0.000 | Lmax30 **0.18**
  (15x training depth) | K16 Lmax8 0.000 | K32 Lmax12 0.10 | Ltr3->12 0.29.
POOL (mean) count-gen: **14.4** (unchanged) -- base-fix cannot help; mean-pool discards the count, so a
fixed-capacity readout genuinely cannot represent a variable-count sum. SUM (per-element + base-once) is the fix.
Net: composing to MORE elements than trained (count OR length) is EXACT in the clean factored setting once the
additive constant is represented once; the only architecture that fails is the one that structurally can't sum.

### scanE: grid scan depth break-point -- reaches 14x training depth (no break yet)
Train len-2 chains, depth-recurrent per-region SUM, pushed hard:
  L2->20 vocab8 HELD 0.91 (maxd 208) | L2->24 vocab8 HELD **0.34** (maxd 238, 12x) | L2->28 vocab8 HELD **0.35** (maxd 251, **14x depth**)
  L2->20 vocab20 HELD 1.12 -- MORE mechanism types hurts more than more depth.
=> No clear depth break even at 14x (maxd 251). The limiting axis is vocabulary (type count), not chain length.

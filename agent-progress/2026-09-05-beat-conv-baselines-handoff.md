# Handoff: beat the pair-concat convolutional baselines (open research)

Date: 2026-09-05. Owner: next agent. Status: OPEN. This is a research mission, not a
task list: form hypotheses, run cheap probes with pre-registered kill criteria, keep
what survives, write everything down (append results to this file or a sibling).

## Mission

Two beds, two targets. Beat them while keeping ONE hard constraint:

> The distance readout must be an INTEGRATION OF LATENT MOTION: distance is the
> accumulated movement of latent embeddings across compute iterations
> (softplus-scale times the sum over iterations of the norm of the embedding delta,
> as in the current Integ head). Everything else about the architecture is negotiable.

That readout is the paper's identity (novelty verified by survey; enables the
interpretability results and greedy navigation wins). What we defend is the readout,
not the current encoder, slots, or token layout.

### Target 1: Switchyard at scale (S11 and S13, then S17/S19)

Bed: `distance_model/switchyard.py` + `scale_bench.py`, headline config transferred
(683 maps, poolq 6800, 160k steps, map split, held-out-map test_corr, 2 seeds).
Env flags: S11 `--G 11 --ngate 3 --nlever 2 --nchute 1 --Rmax 36 --bfsmax 38360`,
S13 `--G 13 ... --Rmax 44 --bfsmax 72903` (S15/S17/S19 in `scale_probe.json`).
Runs on ciirc (`~/swbench`, plain python3, Volta100/A40; NEVER GTX1080Ti).

Numbers to beat (test_corr, s0/s1), from `paper_data/scale_campaign.md`:

| rung | coat64 (ENEMY) | image integ | factored integ | gcurr integ |
|------|----------------|-------------|----------------|-------------|
| S9   | .935 / .942    | .894        | .952 / .938    | .936 / .942 |
| S11  | .947 / .951    | .912 (leo)  | .944 / .936    | .918 / .848 |
| S13  | .961 / .964    | .911 (leo)  | .936 / .933    | not run     |

Read the shape: coat CLIMBS with grid size, factored integ DECLINES, image integ flat
and low, the S9 curriculum win did not transfer to S11. Success = an
integration-readout model at or above coat64 on S11 AND S13 (2 seeds, same site).

### Target 2: CRTR Sokoban, search regime

Bed: `delpi-lab/crtr_bench/` (trainer `crtr_delpi.py`), data + eval on Leonardo
(`~/cmpp/CRTR`, venv `~/cmpp/crtr_venv`, account EUHPC_B38_121, outputs in
`/leonardo_scratch/large/userexternal/jhula000/scale_out/`). Eval = their solver,
1000 boards, solved-rate at node budgets (their paper Fig. 2 protocol; exact numbers
are our repro of their shipped checkpoints).

| model | no-search | search @1000 nodes |
|-------|-----------|--------------------|
| their supervised (pair-concat CE bins, ENEMY) | .224 | **.782** |
| DeLPI f3x3/T8 gap-trained | **.385** | .750 |
| their crtr | .310 | .708 |

Success = integration-readout model above .782 @1000. In flight: clean-BFS-label
f3x3 (corr .974) through their solver (Leonardo job 56054784); check that first,
the .03 gap may already be label noise.

## Why the enemies win (evidence, not speculation)

1. Both winning baselines are PAIR-CONCAT networks: state and goal processed JOINTLY
   from layer 0 (their supervised: channel-concat pair classifier; Chrestien CoAt:
   pair-concat + input re-concat at every block + attention). Per-state embedding
   models lose everywhere (CRTR embedding+L2: .79-.82 switchyard, .708 their bed).
   Our own jointmix probe agrees: IQE .861 -> .924 at S7 when the pair is mixed
   jointly before the metric.
2. Binding is real but SOLVED-ish and not the whole story: factored integ (perfect
   binding) still loses to coat at S13. So at scale the deficit is in the mechanism
   or its budget, not only perception.
3. MAE columns: their CE-argmax readout denoises one-sided label noise (mode vs
   mean); never compare MAE on inflated labels without a bias column (see
   delpi-lab/journal/09, bias-check section).

## Hypothesis backlog (ranked; each with a cheap probe)

H1. ITERATION BUDGET: T=4 was tuned at S7 (diam ~20); S13/S15 diameters are 55-70.
    Probe RUNNING: ciirc job 131955 (`t8scale.sbatch`, factored T8 at S13/S15,
    kill: T8 <= T4 + .005). If T8 helps, sweep T with diameter.

H2. PER-PIXEL INTEGRATION, NO BINDING (user's idea): drop slots entirely; give EVERY
    grid cell its own evolving embedding (the pcnn feature map already provides
    this); run the recurrent passes on the full pixel-token grid (or a strided/
    downsampled version if 361 tokens is too slow) and read out distance as the sum
    over ALL pixels of accumulated latent motion. Nothing needs to learn to bind;
    static cells simply should not move. The readout constraint is satisfied
    per-pixel. Probe: `--readout pixels`-style token path + Integ accumulator over
    pixel tokens at S11, 80k steps, vs coat64's .947. Watch cost: attention over 121+
    tokens x T passes; layers can be cheap (1-2) since convs do local mixing.

H3. STATIC LAYOUT AS CONTEXT + DYNAMIC SLOTS (user's idea): the layout (walls,
    gates, levers, plate, chute) is FIXED per map; only worker/crate/gate-bits
    change. Split the representation: a static context encoding (per-map, computed
    once, e.g. conv features of the layout channels) that CONDITIONS the dynamics
    (via cross-attention or FiLM), plus a small set of dynamic tokens (worker,
    crate, gate state) whose embeddings evolve and whose motion is integrated. Slots
    then never re-discover the layout; they only track the movers, and they are
    aware of the layout through the conditioning. Probe: at S11, dynamic tokens from
    the two entity channels (worker/crate positions are single lit pixels; gather
    their features directly, no attention binding needed) + static context tokens
    frozen across T passes; integrate only the dynamic tokens' motion.

H4. JOINT PAIR PROCESSING BEFORE INTEGRATION: coat-style front (pair-concat conv
    stack with input re-concat) producing tokens for BOTH states jointly, then the
    recurrent integration passes + motion readout on top. The S9 `--reinject` arm
    (.937/.925) is step one of this; extend it with pair-concat (currently the two
    states are encoded separately, goal re-injected only inside the transformer).
    This is the most direct "steal what works, keep the readout" move.

H5. RANGE/CAPACITY SCALING: Rmax grows 24 -> 70 across the ladder while d=256,
    layers=3 stay fixed; the accumulator must represent longer sums. Probes: d 384,
    layers 4, and lr/schedule retune at S13 (one knob at a time, kill at +.005).

H6. CURRICULUM, FIXED: gcurr failed at S11 with a G-4 small phase (.918/.848).
    Variants worth one probe each: multi-stage (7 -> 9 -> 11 progressive), longer
    small phase, or curriculum COMBINED with H3/H4 (the S9 win suggests curricula
    help exactly when binding is the bottleneck; at S11+ the bottleneck moved).

H7. Sokoban transfer: whatever wins on switchyard, port to `crtr_delpi.py` (it
    already has f3x3/tgtchan/dihedral front ends and clean-BFS labels with synthetic
    goals; beware the dataset lens corruption, journal/09). Then their solver.

## Methodology (non-negotiable, learned the hard way)

- Probe first: smallest run that can kill the idea; pre-register the kill criterion
  in the sbatch header comment. 80k steps and 1 seed for probes; 160k and 2 seeds
  only for survivors.
- Same-site anchors only (Leonardo vs ciirc drift is .027 on identical configs).
- Never GTX1080Ti (sm_61 crash); never GPUs others are using; Leonardo account is
  EUHPC_B38_121 (AIFAC expired); Leonardo login needs the user's `leonardo-login`.
- No remote heredocs for quoted content (scp sbatch files); exit-code-gated DONE
  markers; `set -e`.
- Write results into `paper_data/scale_campaign.md` (switchyard) or
  `delpi-lab/journal/` (Sokoban beds), commit with the session trailer, push.
- Untuned-baseline objection: coat runs at lr 1e-3 untuned and still wins; one tuned
  coat run at S13 would make the comparison airtight, do it if time allows.

## In flight right now (do not duplicate)

- ciirc 131813/14/15: scaleup S11-S19 (gcurr/coat64/factored + img at S17/S19),
  finals trickling into `~/swbench/scaleup*_*.out`.
- ciirc 131955: H1 probe (factored T8 at S13/S15).
- ciirc 131812: in-loop L* DeLPI on the rank bed (interim 48/50 @1000).
- Leonardo 56054784: clean-BFS f3x3 through their solver (Target 2 update).
- Leonardo 55837833: matched-budget retrains of their three models.
- OSU: their shipped + 36h-L* checkpoints under the budgeted A* protocol.

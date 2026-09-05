# Switchyard scale campaign (2026-08-29/30)

Scaffold: `distance_model/scale_bench.py` (+ `--bfsmax` in switchyard.py). Tuned 683/167k
headline configs transferred unchanged; only env geometry, Rmax (scaled by the S7
Rmax/diameter ratio), bfsmax (reachable-set calibrated) vary. Probe file:
`distance_model/scale_probe.json`. Raw outs mirrored in `distance_model/scale_results/`
(ciirc A40 / dgx-osu A100 / Leonardo A100); aggregate in `scale_results.json`.

## Ladders

- SIZE S9-S15: grid grows, complexity fixed (3 gates / 2 levers / 1 chute), --steps 160000
  (the S9 probe showed integ still climbing at 80k: 0.855 -> 0.894, plateau ~140k).
- JOINT J9-J15: grid AND complexity grow (4 gates, 3-4 levers, 2-3 chutes), 80k steps
  (2x steps at J11 was falsified: 0.858 -> 0.862).
- Factored arm on the joint ladder: same integ/dhead trunks fed clean per-factor tokens
  (binding removed), 80k.

## Size ladder (Leonardo, 160k, 2 seeds, held-out-map corr)

| model  | S9    | S11   | S13   | S15   |
|--------|-------|-------|-------|-------|
| integ  | .921  | .912  | .911  | .929  |
| iqe    | .864  | .904  | .883  | .898  |
| scalar | .830  | .887  | .839  | .886  |
| sym    | .822  | .827  | .835  | .844  |
| dhead  | .801  | .776  | .842  | .830  |
| mrn    | .802  | .795  | .804  | .819  |

Grid size alone does not degrade the integrator (S15 is its best rung, seeds within
0.002); the margin over iqe compresses vs S7 (+0.01..+0.06) but never flips. Caveats:
scalar/dhead seed spreads up to 0.11; Leonardo S9 integ (0.921) sits 0.027 above the
identical-config ciirc probe run (0.894) = cross-site drift, same class as the earlier
provenance findings.

## Joint ladder (image 80k vs factored 80k, seeds noted)

| J-scale | integ img      | best baseline img | integ fac   | dhead fac   | binding tax (integ) |
|---------|----------------|-------------------|-------------|-------------|---------------------|
| J9      | .813+-.042 (2s)| iqe .818          | .912 (2s)   | .900 (2s)   | ~.10                |
| J11     | .827+-.032 (2s)| iqe .831          | .912 (2s)   | .853 (2s)   | ~.09                |
| J13     | .809+-.043 (2s)| iqe .826          | .897 (2s)   | .891 (2s)   | ~.09                |
| J15     | .805+-.027 (2s)| iqe .834          | .885 (2s)   | .860 (2s)   | ~.08                |

FINAL (both seeds everywhere): image integ ties iqe within noise on EVERY joint cell
(.80-.83 vs iqe's stable .82-.83) with 3-8x the seed variance (J9 .771/.855,
J11 .862/.794, J13 .852/.766, J15 .831/.778); iqe is the most seed-stable image model
at joint complexity. The earlier "J11 integ leads" read was a seed-0 artifact. Binding
tax on integ is ~.08-.10 uniformly; on iqe it is near zero relative (iqe-fac J11 .797
was BELOW iqe-img .831, untuned-head caveat).

- On IMAGE input integ leads no joint cell; on FACTORED it leads every one.
- Binding tax tracks entity DENSITY, not grid size: J9 (12 entities on 9x9) is the worst
  image cell (0.771) despite the smallest grid.
- Image-side seed variance is large at J13/J15 (integ 0.852/0.766, 0.831/0.778): binding
  quality is seed-unstable at scale.
- Factored readout advantage (integ - dhead): +.01/+.06/+.01/+.03 - real but modest,
  strongest at J11 (seed-consistent).
- J15 mrn seed 1: nan (known MRN instability; latentnorm off in tuned config).
- Missing (10h window sacrifices): scalar-factored beyond J11 s0; a few J-image seed-1
  cells still queued on ciirc at time of writing.

## Dead-end probes (all pre-registered kill criteria)

- T=6 at J11: +0.004 (killed). T=8: 0.744, unstable (killed).
- slots 24 at J11: 0.791 vs 0.862 at 16 (more slots HURT binding).
- 2x steps at J11: +0.003 (killed; contrast with S9 where 2x steps gave +0.039).
- cotsup per-pass waypoint-reconstruction aux (pin waypoint k to pass k): 0.795/0.788 vs
  0.862 - actively hurts; conflicts with the measured spatial-not-temporal routing, and
  J11 pairs average only ~1 enabling state (38% have none).

## Checkpoint-CoT replication (cot_switchyard.py, factored, map split, seed 0, 80k)

Port of cot_distance.py: chains = [start, enabling states (effective-gate-mask change
points on a BFS shortest path), goal]; teacher-forced next-waypoint + stop; distance =
summed contextual per-factor increments calibrated on the geodesic; free-run eval.

| bed | freerun corr | gtdist corr | stop acc | tf acc by nway (0/1/2/3) |
|-----|--------------|-------------|----------|--------------------------|
| S7  | .806         | .863        | ~1.0     | .99/.89/.77/.58          |
| J11 | .686         | .888        | 1.0      | .98/.83/.64/.54          |

Replication verdict: stop + calibration replicate; the NEW break (vs the original's
single fixed env) is the cross-map transition at nway>=2, trained on only 13-26% of
pairs.

Fix probe (cotfix, J11): stratified chains (frac>=2: .26 -> .37) + all-states transition
supervision (~10x rows) -> freerun .686 -> .772 (+.086); +wseg per-segment calibration
HURT (.723, no gtdist gain). Transition acc at nway>=2 moved only .64->.70 / .54->.59.
KILL CRITERION FAILED (needed tf@2-3 >= .85, freerun >= .85). Conclusion: the
data-starvation lesson does NOT transfer; predicting the next enabling state on an
UNSEEN map is a genuine composition problem (wiring -> which lever/gate matters next),
not a data problem. The image port (--enc image, implemented and smoke-tested in
cot_switchyard.py, spatial reconstruction heads) was NOT launched per the sequenced
protocol. Next candidates if resumed: bigger/deeper transition trunk, wiring-aware
attention structure, or the hybrid (CoT skeleton + amortized integ as segment metric,
whose fixed-mask segments the size ladder shows integ handles at .92+).

## Where things ran

ciirc amd-1 A40 (J-image), dgx-osu A100 0/2/4 (factored arm), Leonardo boost_usr_prod
(size ladder 48 jobs, CoT). Leonardo env: ~/cmpp/distance_model, cineca-ai/4.3.0,
account EUHPC_B38_121.

## External-architecture transplants (2026-09-05, ciirc 131625/131626/131637)

Faithful ports consuming the pureimage render, trained through the bed's smooth-L1 path
(--extonly crtr|coat, --extw scales widths; published width = 64). crtr = CRTR LNConvNet
(BatchNorm residual trunk, GAP, per-state embedding, distance = L2; symmetric by
construction). coat = Chrestien CoAt (state+goal channel concat, 7 conv blocks with input
re-concat, 4 attention-augmented blocks, GAP -> dense -> scalar). lr 1e-3 for all
(untuned, unlike our per-model-tuned baselines).

S9 bed (160k, map split, test_corr s0/s1):
| model  | params | corr        | mae         |
| coat64 | 1.79M  | .935 / .942 | 1.81 / 1.70 |
| coat32 | 555k   | .900 / .899 | 2.24 / 2.21 |
| coat16 | 170k   | .885 / .947 | 2.50 / 1.73 |
| crtr64 | 307k   | .808 / .820 | 3.59 / 3.43 |
| crtr32 | 79k    | .797 / .801 | 3.64 / 3.60 |
| crtr16 | 21k    | .792 / .793 | 3.67 / 3.65 |
| refs: integ .894 (ciirc same-site) / .921 (leo); iqe .864; scalar .830 |

7x7 headline bed, top rungs (80k, best_corr s0/s1 vs 3-seed campaign means):
| rung | coat16      | integ | scalar | iqe  |
| L4   | .883 / .885 | .935  | .884   | .840 |
| L5   | .871 / .864 | ~.92  | .86-.88| .86-.89 |
| L6   | .824 / .829 | .837  | .788   | .781 |

Reading: CoAt's advantage is scale-dependent. At 7x7 it is mid-pack (scalar-level L4,
below integ everywhere, closest at L6). At S9 coat64 beats integ cleanly (.94 vs .894
same-site) and even coat16's lucky seed does (.947), with 0.06 seed spread at small
width. CRTR transplant plateaus at scalar level; symmetric L2 + global pooling cost it
the detour structure. Open question: whether integ's S9 gap closes with a bigger trunk,
or CoAt's pair-conditioned attention genuinely scales better with grid size.

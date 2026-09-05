# Two additional CRTR tasks: Lights Out and Digit Jumper

User authorized choosing any two additional paper benchmarks and comparing the
successful joint latent-motion method against supervised controls, in parallel,
without using occupied GPUs. This extends, not replaces, the Sokoban campaign.

## Design and verified limits

Preserve JointPixelInteg width 64, T=8, two attention heads and its sole integrated
latent-change readout. No analytical distance solver appears in model inference.
Exact oracles supply labels only: GF(2) inverse/minimum press count for Lights,
directed reverse BFS for Digit.

Public CRTR revision `426c3f11f485e5fa10909e9e3055f30c516306d4` releases only
Sokoban and Rubik environments/checkpoints. The official supplement contains an
older appendix, not missing source code. These new tasks are reconstructed,
label-matched experiments, NOT reproductions of the authors' checkpoints or
their multi-million-update/full-paper training budget.

Lights Out uses a 7x7 binary grid, cardinal-neighbor-plus-self toggles, no wrap.
The final paper documents 49 trajectory actions but does not specify experimental
grid size; 7x7 is an explicit reconstruction assumption. Figure 6 is illustrative
5x5 (also for Digit, despite its documented 20x20 experiment). Rank at 7x7 is 49:
all binary boards are reachable and the unique GF(2) press vector gives exact
distance. Training pairs are independently chosen positions along 49 random
presses from all-off; mixed gaps prevent parity-only sampling. Validation/test
hold out identical unordered endpoint pairs, not all equivalent XOR queries.
The 1,000 search cases are uniform binary boards to all-off, a harder/different
distribution than trajectory-pair direct evaluation.

Digit Jumper uses the paper's 20x20 grid and digits 1..6. A move jumps exactly the
source-cell digit in a cardinal direction. Edges are directed. Invalid moves are
self-loops during search. Boards use the cited puzzlegen IID-digit generation
with reachability rejection, ported to NumPy RNG; this is NOT the final paper's
unspecified path-first sampler. Training positions are sampled uniformly among
cells that can reach bottom-right, including off-path cells. D4-canonical board
hashes are disjoint across train/validation/test. Search uses top-left starts on
the 1,000 final-test boards. Agent-marked digits are encoded as 7..12; base digits
remain 1..6. This categorical representation is a documented reconstruction.
There is no Digit temporal-gap bank; all initial supervision is exact distance.

## Matched arms and evaluation

Ten full runs, each 80,000 Adam updates, batch 128, seeds 0 and 1, no augmentation:

| Task | Model | Learning rate | Source/configuration |
| --- | --- | --- | --- |
| Lights | Joint latent motion | 1e-4 | width 64, T8, heads 2 |
| Lights | Supervised CNN | 1e-4 | released LNConvNet, width 64/depth 8, 50 CE bins |
| Lights | Supervised dense sensitivity | 1e-4 | released LNDenseNet, 8x512, 50 CE bins |
| Digit | Joint latent motion | 3e-4 | width 64, T8, heads 2 |
| Digit | Supervised MLP | 3e-4 | released LNDenseNet, 8x512, 400 CE bins |

The final paper specifies a Lights CNN; its older supplement reports a stronger
dense baseline, hence both controls. Lights CNN width 64 and its exact released
architecture are assumptions, not recovered task-specific configuration. The
paper mentions stronger supervised Lights performance with larger LR/batches
but does not supply those settings. These initial matched settings are not a
claim to have exhausted baseline tuning. Exact supervision is stronger than the
paper's temporal-gap labels and is explicitly labeled in every artifact name.

All models receive the same frozen minibatch stream per seed (trace SHA256 saved).
Each task has 1,000,000 training pairs, 20,000 validation pairs, 20,000 test pairs,
and 1,000 search cases. Digit has 25,000 training boards and 1,000 boards per heldout
split. Validation is logged every 10,000 updates. Final weights are saved before
test inference; test and search results must not guide model selection.

Direct metrics: MAE, RMSE, bias, Pearson and tie-aware Spearman; supervised reports
both argmax and expectation. Search uses supervised argmax and joint scalar
distance with the untouched CRTR BestFSSolver/TrivialPolicy. Lights retains top
10 actions; Digit retains all four; greedy retains one. Report original `nodes`
and strict `nodes < budget` solved fractions separately from expanded nodes.
Cap 6,000, 1,000 cases. Explicit already-solved roots are handled as terminal
before calling the original solver, whose implementation omits that check.

## Safety, deployment and progress

Remote root: `/home/hulajan1/swbench/crtr_extra_20260905`.
Read-only dependencies: `/home/hulajan1/swbench/sokoban_ar_20260905/venv` and `crtr`.
Do not modify existing Switchyard or Sokoban frozen sources. Direct dgx-ciirc
workers still reserve locks even when cards look idle; new work goes through
Slurm on typed compute GPUs only. Resolve the assigned card to UUID, lock it,
reject display cards, any existing compute process, memory above 20 MiB or
nonzero utilization; fail closed on unknown telemetry. Never kill foreign work.

CPU preparation array **132034** completed both tasks successfully (Lights 21 s,
Digit 111 s). Sources are frozen under `prep_src/SHA256SUMS`. Generated files and
source hashes are in each `banks/<env>/manifest.json`. Both train banks contain
exactly 1,000,000 pairs. Lights test mean distance 10.883, search mean 24.456;
Digit test mean 6.390, search mean 9.513. These are label statistics, not results.

GPU smoke runs must use `smoke_test.npz` (byte-identical to validation, separate
path), and `smoke_cases.npz` (first two validation pairs, arbitrary stored goals).
No final-test inference is permitted during the smoke phase. Smokes test 40
updates, final checkpoint reload, both solver modes, two cases and cap 64.

Full workers allow 9 h training, 1 h greedy and 13 h search inside 24 h allocation.
Search timeout/failure leaves per-case partial records and a failed job, never a
complete aggregate. Training failure prevents all search; no automatic resume or
same-prefix overwrite is supported. Search may be costly because the original
solver scores successors individually. Do not call partial results complete.

Successful GPU smokes: Digit array **132037** (joint/MLP, 53 s/34 s), Lights
array **132043** (joint/CNN/dense, 59 s/42 s/38 s), all COMPLETED exit 0.
Every smoke completed 40 training updates, final checkpoint reload, and both
original-solver modes on two validation-derived cases. The real Digit joint and
MLP sampling SHA256 matched exactly.

Full arrays submitted:
- **132046**, Digit tasks 0..3, four V100s on **dgx-5**:
  joint s0, supervised s0, joint s1, supervised s1.
- **132050**, Lights tasks 0..5, six V100s on **dgx-3**:
  joint s0, supervised CNN s0, joint s1, supervised CNN s1, dense s0, dense s1.

Artifact prefixes are `artifacts/<env>_<kind>_exact_s<seed>`; training logs are
`logs/worker_<array>_<task>.log`. Separate `_<mode>.log` files hold search output;
`artifacts/<prefix>_<mode>.cases.jsonl` preserves completed per-case records.
The successful frozen GPU snapshot is `src/SHA256SUMS`; do not edit it.

Original A40 Lights smoke array **132036** safely failed before ANY training:
telemetry repeatedly showed 1 MiB memory and 3–5% utilization. Read-only Slurm
audit **132042** independently observed 5% with no listed compute PID; cause
unknown. No threshold was relaxed, no foreign work was stopped, and the complete
unchanged experiment was moved to free typed V100s. Interactive read-only probe
132041 failed with a Slurm job-credential error; no computation ran.

Verification: **42 focused tests + 70 subtests passed**, with CRTR_TEST_ROOT set to
the real local release so the actual-original-solver test did not skip. This
includes data generation/manifest/smoke isolation, directed BFS and GF(2) oracles,
checkpoint roundtrips, readout invariants, shared GPU guards and Sokoban regression
checks. Independent final code review found no concrete blocker. `bash -n`
passed for both new launchers. Original remote networks.py, search/solver.py and
search/goal_builder.py SHA256s match the locally tested release byte for byte.

Only smokes have completed; no full benchmark comparison or win is known yet.
Local source changes only; no git push authorized in this task.

## Sources

- Final paper: https://papers.nips.cc/paper_files/paper/2025/file/9d75de47462ffe77addaa7b985fc6d8e-Paper-Conference.pdf
- Older supplement: https://papers.nips.cc/paper_files/paper/2025/file/9d75de47462ffe77addaa7b985fc6d8e-Supplemental-Conference.zip
- Release: https://github.com/Princeton-RL/CRTR
- Digit reference: https://github.com/martius-lab/puzzlegen/blob/1a5ff909b80526fde6fb1045cfab9a42291d3c85/puzzlegen/digit_jump.py

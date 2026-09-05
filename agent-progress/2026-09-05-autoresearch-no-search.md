# Direct-distance autoresearch, 2026-09-05

User request: test ideas in parallel on ciirc-old-cluster and dgx-ciirc, use only
unoccupied GPUs, and first beat CNN baselines on MAE or correlation without search.
The integration-of-latent-motion readout remains mandatory.

## Design and execution plan

- Preserve the original Switchyard training maps, BFS labels, map split, 683 maps,
  6800 pool queries, batch 128, Adam, and 80k-step probe budget.
- Use an independent validation map bank for adaptive discovery. The original
  test maps are reserved for 160k confirmation with two seeds. Historical 160k
  coat numbers are context, not an 80k comparison threshold.
- Run matched coat64 anchors on each participating hardware/site and scale.
- Probe joint pair-convolution plus spatial latent integration, static-layout
  context plus moving tokens, the existing per-pixel transformer path, and
  iteration/capacity/learning-rate variants. Never use a scalar prediction head
  for a candidate integration model.
- Cache exact deterministic pool results, including post-build NumPy RNG state;
  cache keys include actual map structures and sampling configuration. Build
  labels on CPUs before occupying GPUs where possible.
- Keep remote source snapshots isolated from existing runs. Slurm owns cluster
  allocations; direct DGX workers pin UUIDs, lock, and recheck occupancy before
  each trial. Exclude display cards and GTX1080Ti. Never stop another run.
- Start a bounded 24-hour campaign. First screen at 80k; promote a candidate if
  its final validation correlation exceeds matched coat64 by .005 OR its MAE is
  at least 3% lower. Record both metrics and their tradeoff. Follow up the best
  near miss with a predeclared one-knob change; stop unpromising branches.
- Confirm surviving configurations at S11 and S13, 160k, seeds 0 and 1, with
  matched coat64 anchors. A win needs two-seed evidence, not a best-test snapshot.
- Save full configs, logs, predictions, checkpoint files, decisions, errors, and
  a machine-readable leaderboard. Report Switchyard and Sokoban separately.

Implementation: parent owns trainer integration, cache, controller, deployment;
parallel agents own joint-pixel and static-context model modules; a third agent
audits baseline/data/evaluation comparability. Check actual latent trajectories
reconstruct the reported distances and gradients are finite before training.

## Initial audit

- dgx-ciirc: eight V100 32GB compute GPUs free, no compute processes at audit.
- ciirc-old-cluster: amd-2 eight A40s free; additional free V100/A40 allocations
  elsewhere. Existing scaleup/T8 jobs are active and must not be duplicated.
- Existing trainer has no validation split; interim evaluation runs in training
  mode. New campaign selects fixed-budget validation endpoints only.
- Local unrelated worktree changes existed before this session and are preserved.

## Live results

Campaign initialized 2026-09-05 17:55 Europe/Prague; 24-hour deadline is
2026-09-06 17:55 Europe/Prague. All 18 initial trials are in the durable queue.
First full-budget validation results are recorded below; no confirmed baseline win yet.

Remote root on both requested hosts:
`/home/hulajan1/swbench/ar_20260905_direct`

- DGX direct workers: eight V100 UUIDs; worker PIDs recorded in
  `dgx_launch.json` and live `state.json`. Initial discovery rung S11.
- Old cluster: eight A40 workers on amd-2, Slurm array **131967**.
  Initial discovery rung S13. Array 131959 failed before any training because
  PyTorch returned UUIDs without the `GPU-` prefix; launcher normalization fixed it.
- Source snapshot is in `src/`, with SHA256 hashes in `state.json`.
  Workers verify these before each new trial. Do not edit that snapshot in place.
- Per-trial logs are in `logs/`; checkpoints and prediction/map-ID arrays are in
  `artifacts/`. Decision records, exact commands, GPU/host/PID, timeouts and metrics
  are in `state.json`. Names labeled `test_*` in validation trials refer to the
  independent validation bank; `eval_bank` distinguishes it explicitly.
- CPU-only label preparation uses four processes and exact Numba BFS; 683 maps
  and 6800 queries preserved. S11 training/evaluation pools contain roughly 238k
  pairs. Cache includes post-build RNG state and full map fingerprints.
- GPU occupancy is checked before each trial; Slurm provides cluster allocations,
  direct workers lock UUIDs. GNU timeout bounds every child independently of worker
  survival, and the child inherits its GPU lock. No existing jobs were stopped.
- If a worker is killed, its bounded child retains the lock, then exits. The task
  can remain marked running; recovery deliberately refuses a duplicate. Inspect
  the GPU, log and worker/child state before reconciling such an orphan manually.

Inspect progress from either host:

```bash
python3 /home/hulajan1/swbench/ar_20260905_direct/src/autoresearch_loop.py status \
  --root /home/hulajan1/swbench/ar_20260905_direct
```

To stop this campaign, create `/home/hulajan1/swbench/ar_20260905_direct/STOP`.
Workers then stop only their own children. Do not kill other jobs or clear the
shared cache. This control is documented, not executed.

Verification: 32 focused CPU tests pass, covering exact BFS/pool/RNG equivalence,
strict motion-readout reconstruction, gradients, queue selection, grid/seed matching,
GPU telemetry, source fingerprints and child lifetime. Three BFS tests also pass
on the actual remote NumPy2.0.2/Numba0.60 environment. Four 200-step GPU smokes
(joint, context, coat64, full-pixel) completed with finite results, no skipped steps,
saved checkpoints and predictions. CPU checkpoint reloads reproduced predictions
bitwise. Additional owned-process crash checks confirmed independent timeout and
GPU-lock retention. Smoke results are not research evidence.

The current loop is deliberately bounded: 80k discovery, at most two mutations
for a near miss, then 160k historical confirmation of up to two survivors on both
rungs and both seeds with matched coat anchors on the same hardware group. A
confirmation failure does not trigger tuning on the historical test data.

CRTR Sokoban data, shipped checkpoints and BFS LUT were not found on either host.
This first loop therefore targets Switchyard only; no solver or search run was
launched. A later Sokoban direct-distance comparison must repair corrupted recorded
goals in the legacy evaluation by using solved goals and report signed bias.

Verified existing 160k results (separate from this new campaign):
S11 coat final corr .946/.949, MAE 2.195/2.040; S13 coat .960/.963,
MAE 2.324/2.181; S13 curriculum .938/.973, MAE 2.761/1.741. The handoff
mistook coat best_corr for final test_corr. Full correction is recorded in
`paper_data/scale_campaign.md`.

Session: 01a07217-b16e-7253-8efe-aa5cdd112637.


## Durable supervision and first completed probes

Both supervisors are running outside the frozen src directory:
DGX PID 1685018, ciirc-old-cluster login PID 4160716. The latter submitted
replacement Slurm array 131979 after earlier workers finished or declined active
GPU telemetry. Arrays 131967 and 131975 are tracked as earlier allocations.
The supervisors use locks, strict idle checks, source verification, bounded
replacement counts and the campaign deadline. They never kill processes or alter
experiment state. Their logs are logs/supervisor-{dgx,slurm}.jsonl. Uncertain sbatch
outcomes block further submissions until reconciled, preventing oversubscription.

At the 2026-09-05 18:19 Europe/Prague snapshot: 5 completed, 9 training, 4 pending.
All completed trials had zero skipped updates. Fixed-80k validation endpoints:

| hardware/rung | candidate | correlation | MAE |
|---|---|---:|---:|
| DGX V100 / S11 | joint64T4 | .912667 | 2.767 |
| DGX V100 / S11 | joint96T4untied | .916668 | 2.734 |
| A40 / S13 | joint64T4 | .925451 | 3.174 |
| A40 / S13 | joint64T8 | .929814 | 2.919 |
| A40 / S13 | joint96T4untied | .927210 | 3.072 |

Matched fixed-budget coat anchors were still running at this snapshot. These rows
are not historical-test results and are not evidence of a CNN win. Source data:
paper_data/autoresearch_20260905_initial.json; the remote state.json remains live.
The context128T4 S11 checkpoint at 20k had corr .950 / MAE 2.181, a preliminary
signal only; selection will use its final endpoint.

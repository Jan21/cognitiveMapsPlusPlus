# Sokoban joint-pixel transfer — 2026-09-05

## Approved scope and design

User requested testing the promising Switchyard joint pixel integration model on
Sokoban against the supervised model that won search. Transfer the architecture,
not the Switchyard weights: width64, tied T8, two attention heads, seven tile
channels. Preserve distance = positive global scale times the sum of norms of
actual latent changes, with no scalar decoder.

Compare both direct distance quality and the original CRTR solver. Use the shipped
supervised checkpoint as a historical anchor and retrain the exact supervised CNN
on the same clean BFS labels, minibatches, augmentation and update budget as ours.
The clean-label comparison is explicitly oracle-supervised; it is not an equal-label
comparison to the original gap-trained checkpoint.

## Execution plan

- [x] Prepare frozen train/validation/test banks from the existing Leonardo data
  and BFS LUTs. Synthetic solved goals; discard corrupt/padded states; validation
  board-disjoint from train, including D4 symmetries; exclude test and solver boards
  from training. Save provenance, hashes and inclusion counts.
- [x] Test the flat-tile joint adapter, exact supervised loading, tie-aware metrics,
  checkpoint round trips and final-only test evaluation. Implement in
  `distance_model/sokoban_joint.py` with its focused tests.
- [x] Test the original-solver adapter in
  `distance_model/value_estimator_sokoban_joint.py`. Preserve search n_actions12,
  greedy n_actions1, max_tree_size6000 and the historical strict generated-node
  reporting threshold nodes<1000; do not silently substitute expanded nodes.
- [x] CPU tests, fresh free-GPU smoke, original supervised direct evaluation.
- [x] Launch matched joint and supervised clean-label runs, seeds0/1, 80k updates
  and batch128, lr1e-3, warmup2k, clip1, joint D4 augmentation. Validation every10k;
  fixed final checkpoint/test, not best-test selection. Add heads0 seed0 ablation
  only on available GPUs. Also run paired joint/supervised gap-label controls,
  seed0, on the identical frozen two-million-pair gap bank. These use a separate
  gap validation bank and the same final clean-distance test as the BFS arms.
- [ ] Original solver evaluation of final joint checkpoints plus original anchor:
  1000 fixed boards, greedy and search. Report all budget curves separately from
  direct MAE/Pearson/Spearman, with clean-label training caveat.

## Safety / artifacts

Remote root `/home/hulajan1/swbench/sokoban_ar_20260905` on shared CIIRC NFS.
Use fresh typed Volta100 Slurm allocations on free old-cluster GPUs; verify UUID,
non-display identity and zero occupancy before training. DGX Switchyard workers
retain their idle reservations. Never edit the frozen Switchyard source or kill
any existing job. No git push authorized.

Source CRTR revision on Leonardo: `426c3f11f485e5fa10909e9e3055f30c516306d4`.
Existing training trajectories: 2.3GB; test83MB; train LUT106MB; test LUT966KB.
Historical supervised search78.2% uses generated nodes<1000, not a 1000-node cap.

## Verified preparation / launch state

CPU preparation job **132023** completed successfully on node-03.
GPU smoke array **132024** completed both tasks with exit0, including 40 training
steps, checkpoint evaluation, and both actual Sokoban solver modes at the smoke
cap. Full comparison array **132026** is running all eight tasks on dgx-2.
Every worker checked a fresh non-display V100 allocation with zero utilization,
negligible memory and no compute processes. Training first-step heartbeats and
the original checkpoint direct result were verified on 2026-09-05 around19:18 Prague.

Frozen banks: 4,671,291 BFS training pairs, 20,000 validation pairs, 20,000 final
test pairs across all 1,000 test trajectories; 2,000,000 gap-training pairs and
20,000 gap-validation pairs. Excluded117 train trajectories whose geometry
overlaps official test/solver boards; rejected603,774 malformed training LUT
rows, zero malformed test rows. All16 sampled positive BFS distances independently
recomputed exactly (8 train +8 test). Train LUT all/all_fix arrays were confirmed
element-identical across5,528,099 entries.

Smoke uses a byte-identical copy of VALIDATION as its diagnostic test, never the
official final test. Solver smoke:2 boards and64-node tree cap, explicitly tagged
smoke-only. Full solver protocol:1,000 boards,6000-node cap, report nodes<1000.

Exact shipped supervised checkpoint loaded successfully into the original
LNConvNet on CPU: output shape(2,150), finite. Lean Sokoban runner imports verified
in the isolated CIIRC environment. It pins CRTR namespace packages because the
shared Python environment otherwise imports an unrelated Magic_Words/search.py;
the original solver, environment, board generator and solve job remain unchanged.

All campaign source/launchers frozen in remote src/SHA256SUMS; every worker checks
them before GPU work. Reproducible local manifest: distance_model/sokoban_source.sha256.
Environment: root/venv inherits installed torch but isolates added gin/JAX/jumanji
dependencies. JAX is CPU-only; PyTorch is UUID-pinned to the allocated GPU.

Expected full array task map:
0 joint_bfs_s0;1 supervised_bfs_s0;2 joint_bfs_s1;3 supervised_bfs_s1;
4 joint_gap_s0;5 supervised_gap_s0;6 joint_noattn_bfs_s0;7 supervised_shipped.
Each training arm gets80k updates,bs128; then greedy and search run on its FINAL
checkpoint. Training timeout8h; greedy1h; search13h; total allocation24h. Output
prefixes refuse overwrite. Any failure stops that worker; no blanket retry,
no foreign-job termination, no ambiguous resubmission.

Results: artifacts/<name>.json direct metrics, .pt final checkpoint,
.predictions.npz paired predictions, .validation.jsonl validation curves.
Solver: artifacts/<name>_<mode>/result.json and logs/<name>_<mode>.log.
Generated-node and expanded-node budgets are separately recorded.

## Initial measured anchor (not a new-model result)

Shipped supervised checkpoint on the corrected 20,000-pair final test:
argmax MAE **3.4808**, Pearson **0.9633360712**, Spearman **0.9742095584**,
bias+2.6582, RMSE5.172785. Same logits softmax expectation:MAE6.304511,
Pearson0.9574502171, bias+6.074846. Original checkpoint remains gap-trained;
the eval-only command's --targets=bfs names the evaluation bank, not retraining.
Historical search reference78.2% is being reproduced afresh by task7.

Verification: **60 focused tests +710 subtests passed**; both GPU smokes completed.
Final joint/baseline comparison is pending. No Sokoban improvement is claimed yet.

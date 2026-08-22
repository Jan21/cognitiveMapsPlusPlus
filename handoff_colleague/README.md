# Switchyard checkpoint + prediction handoff

Code to load these: `switchyard.py` + `traj_switchyard.py` from the DELPI repo, branch
`image-based-experiments` (github.com/Jan21/DELPI).

## Important provenance note

The trainer never saved weights during the original campaigns, so these are **same-seed
retrainings on ciirc-old-cluster A40s** (training is fully deterministic given seed + GPU arch +
torch stack, but the originals ran on different hardware, so scores drift by the run-to-run noise
floor). Recorded vs. these checkpoints (held-out corr):

| checkpoint | recorded | this ckpt | bed | model |
|---|---|---|---|---|
| idv_map_s1 | 0.963 | 0.904 | 683 maps / 167k pairs | integrator T=4, L5 map split |
| int600_map_s2 | 0.961 | 0.917 | 600 / 201k | integrator T=4 |
| int600_map_s3 | 0.965 | 0.956 | 600 / 201k | integrator T=4 |
| dh683_lr5e4_s0 | 0.810 | 0.684 | 683 / 167k | decode-head trunk T=4 |
| dh683_lr5e4_s1 | 0.815 | 0.817 | 683 / 167k | decode-head trunk T=4 |

The decode-head s0 swing is its documented instability, not a bug.

## checkpoints/

`torch.load(path, weights_only=False)` gives `{state_dict, args, model_name, result}`.
`args` is the full flag namespace: rebuild the model with the same `switchyard.py` and those args,
then `load_state_dict`. Easiest is the built-in analysis mode, which also regenerates the exact
held-out pool and dumps per-pass slot trajectories:

```
python3 switchyard.py --train <original flags from args> --loadckpt <ckpt.pt> --dumptraj 4096 --save out
```

(the exact original flag set is in `ck["args"]`; result json is in `ck["result"]`)

## predictions/  (calibration plots)

One npz per model, `d_true` + `d_pred` (float32) per held-out pair, same pair order per bed.
Covers the five checkpoints above plus the tuned 683-bed baseline winners:
iqe (0.891 this run), scalar (0.859), sym (0.789), mrn (0.752).

## analysis/

Output of `traj_switchyard.py` on each checkpoint (walk-vs-distance, per-slot cost vs per-entity
BFS moves, lever-visitation). P1 headline: integrator walk~d 0.90-0.95, decode-head trunk 0.60-0.67
(image analogue of the symbolic 0.95 vs 0.40).

## slotdiag

Nothing here: per-slot binding diagnostics are already in the RESULT lines of the runs
(DELPI `results/raw_runs/`, field `slotdiag`).

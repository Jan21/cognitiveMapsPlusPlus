# Integration-based distance model — coupling gridworld

A recall-flow integrator trained to predict the **BFS geodesic distance** between two states of a coupling
gridworld, evaluated for **distance accuracy** and **generalization to held-out constraint configurations**.
Input can be factored or a rendered image. Self-contained: everything is in `integ_distance.py`.

## The task

A state is `[positions (N), mobility_key, link_key]`:
- **positions** — N agents on a `G×G` grid.
- **mobility_key** — each gated agent (1..N-1) is *free* / *H-only* / *V-only* / *locked* (base-4 packing).
- **link_key** — a bitmask over agent pairs; **linked agents move as one rigid group**.

Coupling makes constraints interact: linking two free agents turns them into one shared 2-DOF unit, so the reachable
set (and the geodesic distances) depend on the joint `(mobility, link)` configuration.

The model is trained only to reproduce the true graph distance:

```
loss = smooth_L1( model(s, g),  BFS_distance(s, g) )
```

No other supervision.

## Model — recall-flow integrator

Each state is encoded into `N+1` tokens (N positions + 1 constraint token). The integrator estimates distance by
simulating movement and accumulating its cost over `T` weight-shared transformer steps, re-injecting the goal and
start each step (**recall**):

```
tok = [emb(s) | emb(g) | emb(s)]
repeat T times:
    z    = Transformer(tok)
    cost += Σ_state_tokens ‖ z − tok ‖          # path length this step
    tok  = [z_state | emb(g) | emb(s)]          # re-inject goal + start
distance = softplus(scale) · cost
```

## Input formats (`--enc`)

| `--enc` | agent position read from | note |
|---|---|---|
| `factored` | direct per-agent embedding | reference (positions handed clean) |
| `bmask` | a lossless `G×G` canvas (deterministic recovery) | image representation, no binding |
| `marker` | learned attention over the canvas | full image: learned agent binding |

The canvas is **additive-lossless**: a cell holds the sum of the embeddings of the agents present, so agent `i` is
always recoverable (bit `i`), and learned attention can bind agents on its own.

## Metrics (accuracy + generalization)

Each run reports, for the evaluation partition and (when a split is set) the training partition:

- **`dist_mae`** — mean absolute error between predicted and true distance.
- **`dist_corr`** — correlation between predicted and true distance.
- **`px_auc`** — P(distance to a legal 1-step move < distance to an illegal one). Ranking sanity.

`test` = held-out configurations (generalization); `train` = in-distribution (accuracy).

## Generalization splits (`--heldout`)

| split | train | test | type |
|---|---|---|---|
| `combo` | most `(mobility,link)` combos | held-out joint combos, same DOF range | interpolation |
| `links2` | ≤ 1 active link | ≥ 2 links (unseen multi-link structure) | hard structural extrapolation |
| `dofhi` / `dofhi2` | DOF ≤ 6 / ≤ 5 | DOF ≥ 7 / ≥ 6 (unseen levels) | value extrapolation |

## Run

```bash
# factored, held-out combos
python integ_distance.py --enc factored --heldout combo --steps 40000 --seed 0

# image (learned binding), hard structural split (needs more steps to learn binding)
python integ_distance.py --enc marker --heldout links2 --steps 80000 --seed 0
```

Batch: `run_best.sbatch` sweeps the best configs (factored / bmask / marker × splits).

## Results

See `RESULTS.md` (populated by `run_best.sbatch`).

# Our scalar baseline: definition and training hyper-parameters

Model: `our_scalar_baseline.py` (verbatim copy of `class Scalar` + its `Block`/`Pool` from
`distance_model/switchyard.py`, selected with `--scalaronly`). Encode start and goal **separately**
(shared perception -> own transformer -> pool -> vector), then a 3-layer MLP on the concatenation.
No recurrence, no start-goal interaction before the final MLP: the no-inductive-bias control.
NOT the same model as `scalar_mlp` in `switchyard_walls.py` (that one runs the integrator's joint
recurrence with a decode head; its analogue of ours is `concat_mlp`). See the docstring for the table.

## Tuned configurations used in the final results (phase B′)
Both tuned on the full switchyard (L5, unseen-map split, seed 0, 80k steps) over: encoder/tokens
{12 slots (1×1), all pixels 1×1+coords, all pixels 3×3 CNN d2–4 w64/128 (+coords), fgpix (excluded later)}
× pooling {mean, flatten} × own transformer depth {0, 2, 4, 6} × width d {128, 256} × lr {5e-4, 1e-3, 2e-3};
then used unchanged at every ladder rung and on both splits, with and without the objectness input plane.

### Image-only setting (the headline tables)
| variant | command flags (after the shared prefix) | params | L5 tuning corr |
|---|---|---|---|
| unconstrained best | `--readout pixels --cnnk 3 --coordconv 1 --cnndepth 4 --cnnw 128 --basepool flat --baselayers 4 --d 128 --lr 1e-3 --scalaronly` | 2.79 M | 0.734 |
| param-matched best | `--readout xattn --slots 12 --cnnk 1 --basepool mean --baselayers 4 --d 128 --lr 1e-3 --scalaronly` | 0.76 M | 0.695 |
Shared prefix: `python3 switchyard.py --train --enc pureimage --heads 4 --nmaps 200 --poolq 2000
--steps 80000 --bs 128 --gradclip 1.0 --warmup 2000 --split map|wire --seed {0,1,2} [rung flags]`;
each also run with `--objch 1` (the "O" variants; best at L5: scalarMO 0.730).

### Factored (symbolic) setting
`python3 switchyard.py --train --enc factored --scalaronly --baselayers 2 --d 128 --heads 4 --lr 1e-3
--gradclip 1.0 --nmaps 200 --poolq 2000 --steps 80000 --split map --seed {0,1,2}` -> L5 corr 0.796 ± 0.004,
MAE 2.96 (tuned over depth {2,4,6} × lr {5e-4,1e-3,2e-3}; depth 2 / lr 1e-3 best).
Also verified: the reference sheet's recipe (4 layers, lr 1e-3, `--seewalls 1`, 60k, up to 560 maps)
changes nothing: 0.789–0.799 (see `paper_data/baselines.md` §6).

## Everything swept for this baseline (so nothing was left on the table)
depth 0/2/4/6 (0 = pure CNN-flatten-MLP) · width 128/256 · pooling mean/flatten · token type slots /
all-pixels(1×1/3×3, CNN depth 2–4, width 64–128) · lr 5e-4/1e-3/2e-3 · ± coordinate planes · ± objectness
plane · ± LayerNorm on the latent (hurts) · 200/560 maps · 60k/80k steps · seewalls on/off.
Best of ALL of it at L5 (image): 0.734 corr / MAE 3.41 (map). Integrator, same protocol: 0.860 / 2.20.

# Training the CNN image integrator (best unsupervised image model)

This is the recipe for the model that won the image-input campaign on switchyard:
**recall-flow integrator reading a rendered scene through a CNN + cross-attention readout**,
trained with **no binding supervision** (no cell labels). It beats every metric/symmetric
baseline on the image input and reverses the earlier additive-canvas result (where the
integrator lost to MRN).

All commands run on the CIIRC A40 cluster. File: `distance_model/switchyard.py`.

---

## 1. Result being reproduced

Metric = `test_corr` (predicted vs true geodesic on the held-out split; higher = better).
Structural generalization, NOT length extrapolation (distances stay in the training range).

| model on image input | split map | split wire |
|---|:--:|:--:|
| **integ (CNN + xattn), unsupervised** | **0.81** | **0.84** |
| IQE (same encoder) | 0.76 | 0.70 |
| sym (same encoder) | 0.71 | 0.76 |
| MRN (same encoder) | nan | nan |

Reference points (same env/data/loss):
- Additive-canvas marker, unsupervised (`--enc marker --bindmode gather`): integ 0.73 → **loses** to MRN 0.78.
- Marker + privileged cell labels (`--enc marker --markeraux 1.0`): integ 0.88 (the ugly supervised ceiling).

So CNN+xattn is the fix: an **unsupervised** image encoder that puts the integrator back on top (0.81/0.84), ~0.03 under the supervised aux ceiling.

---

## 2. The winning command

```bash
python3 switchyard.py --train \
  --nobaseline \
  --enc image --readout xattn --cnnw 64 --cnndepth 2 \
  --d 128 --layers 4 --T 14 \
  --nmaps 200 --poolq 2000 --steps 80000 \
  --split map          # or: --split wire
```

Read `test_corr` from the printed `RESULT` json: field `integ.test_corr`.

To reproduce the baseline row for the same encoder, swap only the head flag (keep everything else identical):

```bash
# same encoder, different distance head
... --iqeonly  --enc image --readout xattn --cnnw 64 --cnndepth 2 ...   # IQE  0.76 / 0.70
... --symonly  --enc image --readout xattn --cnnw 64 --cnndepth 2 ...   # sym  0.71 / 0.76
... --mrnonly  --enc image --readout xattn --cnnw 64 --cnndepth 2 ...   # MRN  nan (numerical blow-up on CNN feats)
```

---

## 3. What each flag does (why it matters)

**Encoder (the part that matters for "image, unsupervised"):**
- `--enc image` — render the full scene to an 8-channel `G x G` image
  (wall / worker / crate / gate / gate-open / lever / plate / chute) and encode it with a small CNN.
  No per-entity index is fed to the distance head; the model must find the entities in pixels.
- `--readout xattn` — factor query tokens (worker, crate) **cross-attend** the CNN feature map
  to pull out their state. This is the load-bearing choice: `xattn` > `convspatial` ~ `convpool`.
  It ports the trick from `gridworld/gw_probe.py` (branch `image-input-stratified-dof`).
- `--cnnw 64` — CNN hidden width. 64 is the sweet spot; 32 works nearly as well, wider (128) does not help.
- `--cnndepth 2` — number of conv layers. **2 is best. Deeper HURTS** (depth 3 → ~0.66, depth 4 → ~0.72).
  Do not add conv layers.

**Distance head (ours):**
- `--nobaseline` — train only the integrator (recall-flow accumulator), skip the scalar baseline.
  distance = `softplus(scale) · Σ_t Σ_i ‖Δz_i(t)‖`, goal + start re-injected each of `T` steps.

**Backbone / optimization (held fixed across the whole comparison for fairness):**
- `--d 128` transformer width, `--layers 4`, `--T 14` recall steps.
- `--nmaps 200` distinct layouts, `--poolq 2000` query pool, `--steps 80000` (80k is needed;
  40k undertrains the image encoder).
- `--split map` = held-out layouts + wirings; `--split wire` = same layouts, wiring resampled
  (tests reading the wiring vs memorizing geometry). Run both.

**NOT used here** (deliberately): `--markeraux` (that is the supervised path), `--bellman`/`--bellwarm`
(those are length-gen stabilizers; this is structural generalization, distances in range).

---

## 4. What was tried and did NOT help (the plateau)

Unsupervised image ceiling is ~0.85. None of these levers beat the simple 2-conv w64:

| lever | tried | verdict |
|---|---|---|
| CNN depth | `--cnndepth 3`, `4` | **hurts** (0.66 / 0.72) |
| CNN width | `--cnnw 128` | no gain over 64 |
| attn heads | `--markerheads 8` | no gain |
| training length | `--steps 120000` | no gain |
| readout | `convspatial`, `convpool` | worse than `xattn` |

The only thing that beats 0.85 is adding privileged cell labels (`--markeraux`, supervised) → 0.88.
That gap (0.85 unsupervised vs 0.88 supervised) is the open item.

---

## 5. Raw outputs on the cluster

`~/cognitiveMapsPlusPlus/distance_model/*.out`, grep `^RESULT`:
- image campaign sweeps: `imgCNN_*.out`, `imgCNN2_*.out`, `imgCNN3_*.out` (depth/width/steps).
- additive-canvas + baseline comparison: `imgBase_*.out`, `bindMode_*.out`, `mkPush_*.out`.

Artifact (published summary with the two figures): **"Image Integrator"**
`https://claude.ai/code/artifact/0aa8f982-b7e8-49a8-9782-dbe4183d00b8`
Source HTML: `distance_model/image_model_campaign.html`.

Related: [`length_gen_results.md`](length_gen_results.md) section 3 (baseline table),
memory `length-extrapolation-campaign` point 10.

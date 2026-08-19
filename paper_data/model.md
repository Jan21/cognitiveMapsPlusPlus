# Model: the recall-flow integrator (image-only version used in the final results)

Recipe (`distance_model/switchyard.py`):
```
python3 switchyard.py --train --enc pureimage --cnnk 1 --cnnw 64 --cnndepth 2 --objch 1 --readout xattn --slots 16 \
  --d 256 --layers 3 --heads 4 --T 4 --lr 1e-3 --gradclip 1.0 --warmup 2000 --nmaps 200 --poolq 2000 --steps 80000 \
  --bs 128 --nobaseline --split map|wire --seed {0,1,2} [rung flags]
```
Parameters: 1.9 M (d 256). Small variant (`--slots 12 --d 128 --layers 4`): 0.56 M.

## Forward pass (one pair of states)
1. **Render** start and goal to 13 planes each (12 entity/wall planes + objectness plane). Shape (B, 13, 7, 7).
2. **Per-cell encoder**: `Conv1×1(13→64) → ReLU → Conv1×1(64→256) → ReLU` (no spatial mixing; each cell's vector is a
   function of that cell only). Flatten to 49 cell vectors, add a learned per-cell position embedding.
3. **Slots**: 16 learned query vectors, one-shot multi-head cross-attention (4 heads) over the 49 cells → 16 tokens of
   256 dims per state; add a learned slot-index embedding. Nothing steers the attention; slots specialise during
   training (entropy/entity diagnostics in `slotDiag` runs).
4. **Sequence**: `[enc(start)+role₀ ; enc(goal)+role₁ ; enc(start)+role₂]` = 48 tokens (goal and start-anchor are
   re-injected every pass: "recall").
5. **Recall loop**, T = 4 passes of one weight-shared block of 3 pre-norm transformer layers (d 256, 4 heads, FF 512,
   GELU):
```
cost = 0
for t in range(T):
    z    = Block(tok)
    cost = cost + (z[:, :16] − tok[:, :16]).norm(dim=-1).sum(-1)   # L2 displacement of each state token, summed
    tok  = [ z[:, :16] ; goal ; anchor ]
D = softplus(scale) · cost                                           # one learned positive scalar, no head
```
6. **Loss**: smooth-L1 between D and the BFS distance. Adam, lr 1e-3 with 2k-step linear warm-up, grad-norm clip 1.0,
   batch 128, 80k steps. Non-finite-loss steps are skipped (never triggered for the integrator).

## Design choices and where they came from (details in `ablations.md`)
- **1×1 instead of 3×3 encoder**: 3×3 convolution blurred per-cell identity (gate bits, lever identity) into neighbours;
  this was the failure of all image models at the first coupled rung (pinpoint study: L1½ "read static gate bits" is
  where 3×3 breaks). Per-pixel features recovered the symbolic level (L2: 0.76 → 0.90).
- **Safe optimiser** (lr 1e-3, clip, warm-up): at lr 2e-3 without clipping a slot sometimes never bound the worker and
  training plateaued (bimodal seeds, 0.73 vs 0.93). The recipe removed the bimodality for us and for the baselines.
- **16 slots, d 256, 3 layers**: slots ≥ number of entities (9–11 at L3+) mattered (+0.06 at L3); width over depth.
- **Objectness plane**: one extra input feature, given to every model; +0.01–0.03 for the integrator, +0.03 for IQE.
- **T = 4**: recurrence is unnecessary on uncoupled rungs (T 1 ≈ T 14) and pays from L2 on; T 4 is the sweet spot at
  the training budget used. Re-injection of goal/start (recall) is load-bearing at full complexity (without it training
  does not fit: train MAE 2.8).
- Rejected after testing: competitive iterative slot attention (Locatello-style: worse at every rung), reconstruction
  auxiliaries, attention-entropy sharpening, hard/straight-through attention, learned attention bias on lit cells (kept
  out of the final model on purpose), coordinate channels with slots (no gain), wire-path rendering, more maps, longer
  training, cosine decay, deeper block, more heads, 1×1-then-3×3 encoder, supervised binding auxiliaries (the supervised
  Hungarian ceiling on this encoder is *below* the unsupervised model), and the hand-built foreground-pixel tokenizer
  (same accuracy, but a detector written for this rendered world; reported only as an ablation).

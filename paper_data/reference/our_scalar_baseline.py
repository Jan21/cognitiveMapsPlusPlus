"""OUR scalar baseline, verbatim from distance_model/switchyard.py (commit dc6be6d line range 549-632).

This is the model called "scalar" in every table of ours (paper_data/results_main.md, baselines.md).
It is DELIBERATELY different from the reference folder's `scalar_mlp` readout:

  ours ("scalar", below)                 hers ("scalar_mlp" in switchyard_walls.py)
  ------------------------------------   -------------------------------------------------
  start and goal encoded SEPARATELY      joint sequence [state | goal | anchor | context]
  own transformer, applied ONCE          the integrator's weight-shared block, applied T times
  no interaction until the last MLP      full start-goal cross-attention every pass, goal re-injected
  MLP([f(s); f(g)]) -> distance          MLP(mean(final state tokens)) -> distance

Hers keeps the recurrent joint processing (= the integrator minus the displacement sum; in our code that
exact model is `--decodehead` on the integrator). Ours removes it: that is the point of a no-inductive-bias
control. Her file's `concat_mlp` readout is the analogue of ours.

The classes below close over the training-script context in switchyard.py:
  a       - argparse namespace (a.d width, a.heads, a.baselayers, a.basepool, a.latentnorm)
  Enc     - the shared perception (identical for every head): for the image setting,
            render -> [objectness plane] -> 1x1 conv stack -> 49 cell vectors -> 12 learned slot queries
            (one-shot cross-attention) -> 12 tokens; for the factored setting, one embedding token per
            entity. NTOK = number of tokens per state.
  Block   - stack of pre-norm TransformerEncoderLayers (l=0 -> identity).
Selected with `--scalaronly`. Trained by the same loop as every model (smooth-L1 to BFS distance).
"""
import torch, torch.nn as nn

class Block(nn.Module):
    def __init__(s, d, h, l):
        super().__init__()
        s.ls = nn.ModuleList([nn.TransformerEncoderLayer(d, h, 2 * d, dropout=0.0, activation="gelu",
                              batch_first=True, norm_first=True) for _ in range(max(0, l))])   # l=0 -> identity
    def forward(s, z):
        for l in s.ls: z = l(z)
        return z

class Pool(nn.Module):
    """baseline state pooling: mean over tokens (default) or FLATTEN all tokens -> Linear (keeps which-token-where)."""
    def __init__(s, d, ntok, basepool):
        super().__init__()
        s.flat = nn.Sequential(nn.Linear(ntok * d, 2 * d), nn.GELU(), nn.Linear(2 * d, d)) if basepool == "flat" else None
    def forward(s, z):
        return s.flat(z.flatten(1)) if s.flat is not None else z.mean(1)

class Scalar(nn.Module):
    """Encode each state separately -> pool -> MLP on the concatenation. No recurrence, no joint processing."""
    def __init__(s, Enc, d, heads, baselayers, ntok, basepool="mean", latentnorm=False):
        super().__init__()
        s.enc = Enc(d); s.mix = Block(d, heads, baselayers); s.pool = Pool(d, ntok, basepool)
        s.ln = nn.LayerNorm(d) if latentnorm else None
        s.head = nn.Sequential(nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, 2 * d), nn.GELU(), nn.Linear(2 * d, 1))
    def forward(s, x, g, m):
        hs = s.pool(s.mix(s.enc(x, m))); hg = s.pool(s.mix(s.enc(g, m)))
        if s.ln is not None: hs, hg = s.ln(hs), s.ln(hg)
        return s.head(torch.cat([hs, hg], 1)).squeeze(-1)

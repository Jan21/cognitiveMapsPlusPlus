"""Joint-pair spatial integration probes for switchyard hypotheses H2/H4.

The sole distance readout is

    softplus(scale) * sum_{update, row, col} ||z_next - z||_2.

Both images enter the convolutional encoder together. Every grid cell has an
evolving embedding, and raw pair features can be reinjected on every update.
There is no scalar decoder, additive distance bias, endpoint-distance term, or
input-dependent readout weight. The initial encoder's movement is not counted.

These are pair-conditioned computation trajectories, not guaranteed physical
paths or a metric over independently embedded states. In particular, symmetry,
zero self-distance, and the triangle inequality are not imposed. Counting all
cells introduces a grid-size dependence that the global scale must learn.
"""

from __future__ import annotations

import math
from numbers import Integral

import torch
from torch import nn
from torch.nn import functional as F


class _JointSpatialUpdate(nn.Module):
    """Produce one vector-valued spatial update, optionally with global attention."""

    def __init__(self, width, pair_channels, kernel_size, attention_heads):
        super().__init__()
        padding = kernel_size // 2
        self.norm = nn.GroupNorm(1, width)
        self.conv1 = nn.Conv2d(width + pair_channels, width, kernel_size, padding=padding)
        self.conv2 = nn.Conv2d(width, width, kernel_size, padding=padding)
        self.attention_heads = attention_heads
        if attention_heads:
            self.qkv = nn.Conv2d(width, 3 * width, 1)
            self.attention_out = nn.Conv2d(width, width, 1)

    def forward(self, state, pair):
        normalized = self.norm(state)
        inputs = torch.cat((normalized, pair), dim=1) if pair is not None else normalized
        hidden = F.silu(self.conv1(inputs))
        update = self.conv2(hidden)
        if self.attention_heads:
            batch, width, height, columns = hidden.shape
            qkv = self.qkv(hidden).reshape(
                batch, 3, self.attention_heads, width // self.attention_heads,
                height * columns,
            )
            query, key, value = (part.transpose(-1, -2) for part in qkv.unbind(dim=1))
            attended = F.scaled_dot_product_attention(query, key, value)
            attended = attended.transpose(-1, -2).reshape(batch, width, height, columns)
            update = update + self.attention_out(attended)
        return update


class JointPixelInteg(nn.Module):
    """A pair-concat conv encoder followed by accumulated spatial latent motion.

    Args:
        in_channels: Channels in EACH image; switchyard pureimage uses 10+nlever.
        width: Channels in every evolving cell embedding.
        T: Default number of recurrent iterations.
        kernel_size: Odd convolution kernel for encoder and recurrence.
        tied: Reuse the same update blocks across iterations. Untied creates T
            independent sets of blocks, enabling a feed-forward depth control.
        stride: Encoder's initial convolution stride; 1 keeps all input cells.
            Raw-pair reinjection uses adaptive average pooling to the latent grid
            when striding. The resulting lower-resolution cells are integrated.
        blocks: Residual spatial updates PER iteration. Each block's actual
            movement is accumulated separately, for T*blocks total increments.
        reinject: Concatenate the raw image pair at every spatial update.
        attention_heads: 0 disables attention; otherwise must divide width.
            Attention is part of the vector update, with no separate readout.
        step_scale: Fixed update multiplier before width normalization. Actual
            residual multiplier is step_scale/sqrt(width), keeping initial
            per-cell vector lengths roughly comparable between width probes.
        scale_init: Initial positive global distance multiplier, represented by
            a trainable unconstrained scalar followed by softplus.

    ``forward(a, b)`` consumes equal BCHW floating tensors and returns B values.
    ``Trun`` overrides T (including zero); tied updates can extrapolate beyond T.
    ``ret_states=True`` returns ``(distance, states)`` with the INITIAL state and
    every subsequent integrated state, each BCHW. Diagnostics retain the graph.
    """

    def __init__(
        self,
        in_channels,
        width=64,
        T=8,
        kernel_size=3,
        tied=True,
        stride=1,
        blocks=1,
        reinject=True,
        attention_heads=0,
        step_scale=0.1,
        scale_init=1.0,
    ):
        super().__init__()
        for name, value in (
            ("in_channels", in_channels), ("width", width), ("T", T),
            ("kernel_size", kernel_size), ("stride", stride), ("blocks", blocks),
        ):
            if not isinstance(value, Integral) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve the recurrent grid")
        if (not isinstance(attention_heads, Integral) or attention_heads < 0
                or (attention_heads and width % attention_heads)):
            raise ValueError("attention_heads must be 0 or a positive divisor of width")
        for name, value in (("step_scale", step_scale), ("scale_init", scale_init)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")

        self.in_channels = in_channels
        self.width = width
        self.T = T
        self.tied = tied
        self.stride = stride
        self.blocks = blocks
        self.reinject = reinject
        self.step_scale = step_scale / math.sqrt(width)
        padding = kernel_size // 2
        self.encoder = nn.Sequential(
            nn.Conv2d(2 * in_channels, width, kernel_size, stride=stride, padding=padding),
            nn.SiLU(),
            nn.Conv2d(width, width, kernel_size, padding=padding),
            nn.SiLU(),
        )
        self.updates = nn.ModuleList(
            nn.ModuleList(
                _JointSpatialUpdate(
                    width, 2 * in_channels if reinject else 0,
                    kernel_size, attention_heads,
                )
                for _ in range(blocks)
            )
            for _ in range(1 if tied else T)
        )
        # Stable inverse softplus also for very small/large positive scales.
        self.scale = nn.Parameter(torch.tensor(scale_init + math.log(-math.expm1(-scale_init))))

    def forward(self, a, b, Trun=None, ret_states=False):
        if a.ndim != 4 or b.ndim != 4 or a.shape != b.shape:
            raise ValueError("a and b must have identical BCHW shapes")
        if a.shape[1] != self.in_channels or min(a.shape[2:]) < 1:
            raise ValueError(f"expected {self.in_channels} channels and a nonempty spatial grid")
        if not a.is_floating_point() or not b.is_floating_point():
            raise ValueError("a and b must be floating-point image tensors")
        iterations = self.T if Trun is None else Trun
        if (not isinstance(iterations, Integral) or isinstance(iterations, bool)
                or iterations < 0):
            raise ValueError("Trun must be a nonnegative integer")
        if not self.tied and iterations > len(self.updates):
            raise ValueError("untied updates cannot run beyond their configured T")

        pair = torch.cat((a, b), dim=1)
        state = self.encoder(pair)
        context = None
        if self.reinject:
            context = pair if pair.shape[-2:] == state.shape[-2:] else F.adaptive_avg_pool2d(
                pair, state.shape[-2:]
            )
        # Sum norms in float32 under mixed precision, without adding epsilon:
        # exactly zero latent movement must produce exactly zero distance.
        accumulation_dtype = torch.float32 if state.dtype in (torch.float16, torch.bfloat16) else state.dtype
        cost = torch.zeros(a.shape[0], dtype=accumulation_dtype, device=a.device)
        states = [state] if ret_states else None
        for iteration in range(iterations):
            for update in self.updates[0 if self.tied else iteration]:
                next_state = state + self.step_scale * update(state, context)
                delta = next_state.to(accumulation_dtype) - state.to(accumulation_dtype)
                cost = cost + torch.linalg.vector_norm(delta, dim=1).sum(dim=(1, 2))
                state = next_state
                if ret_states:
                    states.append(state)
        distance = F.softplus(self.scale) * cost
        return (distance, states) if ret_states else distance


__all__ = ["JointPixelInteg"]

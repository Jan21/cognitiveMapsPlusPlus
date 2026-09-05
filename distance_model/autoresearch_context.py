"""H3: static Switchyard layout conditions a small recurrent set of dynamic tokens.

The inputs are the existing ``Enc._render_pure`` image pair, shape (B, C, G, G):
0 wall; 1 worker; 2 crate; 3 gate locations; 4 RAW gate-open bits; 5.. lever
locations plus wired gates; then plate plus wired gates and four chute directions.
Channels 1, 2 and 4 are dynamic. All others are static and include the wiring.
In particular channel 4 does not include the pressure-plate override: the model
must learn that dependency from the crate and plate/wiring channels.

``ContextInteg(...)(a, b)`` predicts the directed distance from a to b. Both images
must have the same layout and exactly ``ngate`` gate markers; no map IDs, distance
labels, or search are used. Worker/crate binding is explicit via the supplied
single-pixel entity channels. Gate tokens use the supplied gate-location channel,
ordered by cell index, so each bit remains bound to its gate location.

The layout CNN runs once per image pair. Its features and the original endpoint
tokens are fixed memories throughout the recurrence (but remain differentiable).
All source AND goal dynamic tokens evolve jointly with different endpoint roles.
Only their actual movement between complete recurrent passes is integrated:

    distance = softplus(scale) * sum_t sum_token ||z[t + 1] - z[t]||_2

There is no independent distance decoder or additive prior. ``ret_states=True``
returns the initial tokens followed by each pass's tokens for auditing this
identity. Static features are recomputed each forward, avoiding stale parameters
or map information leaking through a persistent cache. Like the existing Integ
model, nonnegativity is structural; zero self-distance and triangle inequalities
are not enforced.
"""

import torch
from torch import nn
from torch.nn import functional as F


class _LayoutEncoder(nn.Module):
    """Raw layout re-injection keeps narrow wiring channels visible to the CNN."""

    def __init__(self, in_channels, width, d, depth):
        super().__init__()
        self.convs = nn.ModuleList(
            nn.Conv2d(in_channels if i == 0 else width + in_channels,
                      d if i == depth - 1 else width, 3, padding=1)
            for i in range(depth)
        )

    def forward(self, layout):
        h = layout
        for index, conv in enumerate(self.convs):
            h = F.gelu(conv(h if index == 0 else torch.cat([h, layout], dim=1)))
        return h


class _ContextBlock(nn.Module):
    """Update dynamic tokens using each other and a fixed, preprojected memory."""

    def __init__(self, d, heads, step_size):
        super().__init__()
        self.heads = heads
        self.step_size = step_size
        self.self_norm = nn.LayerNorm(d)
        self.self_attn = nn.MultiheadAttention(d, heads, dropout=0.0, batch_first=True)
        self.query_norm = nn.LayerNorm(d)
        self.memory_norm = nn.LayerNorm(d)
        self.query = nn.Linear(d, d)
        self.key_value = nn.Linear(d, 2 * d)
        self.cross_out = nn.Linear(d, d)
        self.ff_norm = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def _split_heads(self, tensor):
        batch, tokens, d = tensor.shape
        return tensor.reshape(batch, tokens, self.heads, d // self.heads).transpose(1, 2)

    def prepare_memory(self, memory):
        key, value = self.key_value(self.memory_norm(memory)).chunk(2, dim=-1)
        return self._split_heads(key), self._split_heads(value)

    def forward(self, tokens, memory):
        normalized = self.self_norm(tokens)
        update = self.self_attn(normalized, normalized, normalized, need_weights=False)[0]
        tokens = tokens + self.step_size * update
        query = self._split_heads(self.query(self.query_norm(tokens)))
        key, value = memory
        update = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0)
        update = update.transpose(1, 2).reshape(tokens.shape)
        tokens = tokens + self.step_size * self.cross_out(update)
        return tokens + self.step_size * self.ff(self.ff_norm(tokens))


class ContextInteg(nn.Module):
    """Drop-in image-pair distance model for the isolated H3 autoresearch probe.

    ``d`` and ``layers`` control the dynamic recurrent stack, ``T`` its repeated
    applications; weights are shared across T. ``cnndepth`` and ``cnnw`` control
    the static layout CNN. ``step_size`` sets the residual update size, not a
    separate distance readout. Ten moving tokens are used for the default three
    gates: worker, crate and three gate states for each endpoint.
    """

    def __init__(self, in_channels=12, d=128, heads=4, layers=2, T=4,
                 cnnw=64, cnndepth=3, ngate=3, step_size=0.1):
        super().__init__()
        if in_channels < 5:
            raise ValueError("Images need wall, worker, crate, gate and gate-bit channels")
        if d < 1 or heads < 1 or d % heads:
            raise ValueError("d must be positive and divisible by heads")
        if layers < 1 or cnndepth < 1 or cnnw < 1 or T < 0 or ngate < 0:
            raise ValueError("Invalid depth, width, iteration count or gate count")
        if step_size <= 0:
            raise ValueError("step_size must be positive")
        self.in_channels = in_channels
        self.d = d
        self.T = T
        self.ngate = ngate
        self.register_buffer("static_channels", torch.tensor(
            [channel for channel in range(in_channels) if channel not in (1, 2, 4)],
            dtype=torch.long,
        ), persistent=False)
        self.context_encoder = _LayoutEncoder(in_channels - 3 + 2, cnnw, d, cnndepth)
        self.position = nn.Linear(2, d, bias=False)
        self.entity_type = nn.Embedding(3, d)  # worker / crate / gate; no gate-ID lookup
        self.endpoint_role = nn.Embedding(2, d)
        self.gate_bit = nn.Linear(1, d, bias=False)
        self.layout_role = nn.Parameter(torch.randn(d) * 0.02)
        self.blocks = nn.ModuleList(_ContextBlock(d, heads, step_size) for _ in range(layers))
        self.scale = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.entity_type.weight, std=0.02)
        nn.init.normal_(self.endpoint_role.weight, std=0.02)

    def _initial_tokens(self, image, context, gate_cells, role):
        batch, _, height, width = image.shape
        # Each entity input channel contains exactly one lit pixel in the renderer.
        entities = image[:, 1:3].flatten(2).argmax(dim=-1)
        cells = torch.cat([entities, gate_cells], dim=1)
        tokens = context.gather(1, cells[..., None].expand(-1, -1, self.d))
        types = torch.cat([
            torch.arange(2, device=image.device),
            torch.full((self.ngate,), 2, dtype=torch.long, device=image.device),
        ])
        tokens = tokens + self.entity_type(types)[None] + self.endpoint_role.weight[role]
        # Gate position and its raw bit are represented in the SAME token.
        if self.ngate:
            bit = image[:, 4].reshape(batch, height * width).gather(1, gate_cells)
            tokens = torch.cat([tokens[:, :2], tokens[:, 2:] + self.gate_bit(bit[..., None])], 1)
        return tokens

    def forward(self, a, b, Trun=None, ret_states=False):
        if a.ndim != 4 or a.shape != b.shape or a.shape[1] != self.in_channels:
            raise ValueError("a and b must have equal (B, in_channels, H, W) shapes")
        iterations = self.T if Trun is None else Trun
        if not isinstance(iterations, int) or iterations < 0:
            raise ValueError("Trun must be a nonnegative integer")
        batch, _, height, width = a.shape
        if self.ngate > height * width:
            raise ValueError("ngate exceeds the number of image cells")
        row, col = torch.meshgrid(
            torch.linspace(-1, 1, height, device=a.device, dtype=a.dtype),
            torch.linspace(-1, 1, width, device=a.device, dtype=a.dtype),
            indexing="ij",
        )
        coordinates = torch.stack([row, col], dim=0)[None].expand(batch, -1, -1, -1)
        layout = torch.cat([a.index_select(1, self.static_channels), coordinates], dim=1)
        context = self.context_encoder(layout).flatten(2).transpose(1, 2)
        context = context + self.position(coordinates.flatten(2).transpose(1, 2))
        gate_cells = a[:, 3].flatten(1).topk(self.ngate, dim=1).indices.sort(dim=1).values
        tokens = torch.cat([
            self._initial_tokens(a, context, gate_cells, role=0),
            self._initial_tokens(b, context, gate_cells, role=1),
        ], dim=1)
        # Endpoint recall: the original pair and layout stay fixed across passes.
        memory = torch.cat([context + self.layout_role, tokens], dim=1)
        memories = [block.prepare_memory(memory) for block in self.blocks]
        states = [tokens] if ret_states else None
        # FP32 summation also measures the actual rounded states under AMP.
        cost = torch.zeros(batch, device=a.device, dtype=torch.float32)
        for _ in range(iterations):
            previous = tokens
            for block, projected in zip(self.blocks, memories):
                tokens = block(tokens, projected)
            cost = cost + (tokens.float() - previous.float()).norm(dim=-1).sum(dim=-1)
            if ret_states:
                states.append(tokens)
        distance = F.softplus(self.scale.float()) * cost
        return (distance, states) if ret_states else distance

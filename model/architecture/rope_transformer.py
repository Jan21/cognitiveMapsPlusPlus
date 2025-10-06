import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) as introduced in RoFormer.
    Applies rotation to query and key vectors based on their absolute positions.
    """
    def __init__(self, d_model: int, max_seq_length: int = 128, base: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        self.base = base

        # Compute theta values for rotation
        # theta_i = base^(-2i/d) for i in [0, d/2)
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)

        # Precompute rotation matrices for efficiency
        self._precompute_freqs(max_seq_length)

    def _precompute_freqs(self, seq_len: int):
        """Precompute the rotation frequencies for positions 0 to seq_len-1"""
        t = torch.arange(seq_len, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        # Create complex representation for rotation
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :], persistent=False)
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :], persistent=False)

    def rotate_half(self, x):
        """Rotate half the hidden dims of the input."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def apply_rotary_pos_emb(self, x, cos, sin):
        """Apply rotary positional embedding to input tensor."""
        # x shape: (batch, n_heads, seq_len, head_dim)
        return (x * cos) + (self.rotate_half(x) * sin)

    def forward(self, q, k, seq_len=None):
        """
        Apply rotary embeddings to queries and keys.

        Args:
            q: Query tensor (batch, n_heads, seq_len, head_dim)
            k: Key tensor (batch, n_heads, seq_len, head_dim)
            seq_len: Optional sequence length (defaults to q.size(2))

        Returns:
            Rotated query and key tensors
        """
        if seq_len is None:
            seq_len = q.size(2)

        # Extend cached freqs if needed
        if seq_len > self.cos_cached.size(2):
            self._precompute_freqs(seq_len)

        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        q_embed = self.apply_rotary_pos_emb(q, cos, sin)
        k_embed = self.apply_rotary_pos_emb(k, cos, sin)

        return q_embed, k_embed


class RoPEMultiHeadAttention(nn.Module):
    """Multi-head attention with RoPE positional encoding"""
    def __init__(self, d_model: int, num_heads: int, max_seq_length: int = 128,
                 dropout: float = 0.1, rope_base: int = 10000):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_length, rope_base)
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)

    def forward(self, x, attention_mask=None, causal_mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            attention_mask: (batch_size, seq_len) - 1 for real tokens, 0 for padding
            causal_mask: (seq_len, seq_len) - boolean mask for causal attention
        """
        batch_size, seq_len, _ = x.shape

        # Project and reshape to (batch, n_heads, seq_len, head_dim)
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to queries and keys
        q, k = self.rope(q, k, seq_len)

        # Compute attention scores
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / self.scale


        # Apply padding mask if provided
        if attention_mask is not None:
            # attention_mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))

        # Softmax and dropout
        attn_probs = F.softmax(attn_weights, dim=-1)
        attn_probs = self.dropout(attn_probs)

        # Apply attention to values
        out = torch.matmul(attn_probs, v)

        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)

        return out


class TransformerBlock(nn.Module):
    """Transformer block with RoPE attention"""
    def __init__(self, d_model: int, num_heads: int, d_ff: int,
                 max_seq_length: int = 128, dropout: float = 0.1, rope_base: int = 10000):
        super().__init__()

        self.attention = RoPEMultiHeadAttention(d_model, num_heads, max_seq_length, dropout, rope_base)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x, attention_mask=None, causal_mask=None):
        # Self-attention with residual
        attn_out = self.attention(x, attention_mask, causal_mask)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


class RoPETransformer(nn.Module):
    """Transformer model with Rotary Position Embeddings (RoPE)"""
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 1024,
        max_seq_length: int = 128,
        dropout: float = 0.1,
        rope_base: int = 10000,
        **kwargs
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.mask_token_id = vocab_size - 1

        # Token embedding (no positional embedding needed - RoPE handles it)
        self.token_embedding = nn.Embedding(vocab_size, d_model)

        # Learnable vectors for first and last token indicators
        self.first_token_embedding = nn.Parameter(torch.zeros(1, 1, d_model))
        self.last_token_embedding = nn.Parameter(torch.zeros(1, 1, d_model))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, max_seq_length, dropout, rope_base)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

        # Initialize first and last token embeddings
        torch.nn.init.normal_(self.first_token_embedding, mean=0.0, std=0.02)
        torch.nn.init.normal_(self.last_token_embedding, mean=0.0, std=0.02)

    def forward(self, input_ids, target_ids=None, attention_mask=None):
        """
        Args:
            input_ids: (batch_size, seq_len)
            target_ids: (batch_size, seq_len) - Optional, for training
            attention_mask: (batch_size, seq_len) - Optional, 1 for real tokens, 0 for padding

        Returns:
            dict with 'logits' and optionally 'loss'
        """
        batch_size, seq_len = input_ids.shape

        # Embed tokens
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)

        # Add special embeddings to first and last tokens
        x[:, 0, :] = x[:, 0, :] + self.first_token_embedding
        x[:, -1, :] = x[:, -1, :] + self.last_token_embedding

        x = self.dropout(x)

        # Apply transformer blocks (non-causal, bidirectional attention)
        for block in self.blocks:
            x = block(x, attention_mask, causal_mask=None)

        x = self.norm(x)
        logits = self.output_projection(x)

        # Compute loss if targets provided
        result = {"logits": logits}
        if target_ids is not None:
            loss = self.get_loss(logits, target_ids, input_ids)
            result["loss"] = loss

        return result

    def get_loss(self, logits, targets, masked_input_ids):
        """
        Compute cross-entropy loss only on masked tokens

        Args:
            logits: (batch_size, seq_len, vocab_size)
            targets: (batch_size, seq_len)
            masked_input_ids: (batch_size, seq_len) - input with mask tokens

        Returns:
            loss: scalar tensor
        """
        mask_token_id = self.mask_token_id
        mask = (masked_input_ids == mask_token_id)

        # Set targets to -100 (ignore_index) where not masked
        targets = targets.clone()
        targets[~mask] = -100

        shift_logits = logits.view(-1, logits.size(-1))
        shift_labels = targets.view(-1)

        loss = F.cross_entropy(
            shift_logits,
            shift_labels,
            ignore_index=-100
        )
        return loss

    def generate_path(self, goal_state: int, start_state: int, max_length: int = 64,
                     temperature: float = 1.0, top_k: int = None) -> list:
        """Generate a path from start_state to goal_state using autoregressive sampling"""
        self.eval()
        device = next(self.parameters()).device
        eos_token = self.vocab_size - 1

        # Initialize with goal and start state
        input_sequence = torch.tensor([goal_state, start_state], dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                output = self(input_sequence, None)
                logits = output["logits"]

                # Get logits for the last position
                next_token_logits = logits[0, -1, :] / temperature

                # Apply top-k filtering if specified
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits[top_k_indices] = top_k_logits

                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

                # Add to sequence
                input_sequence = torch.cat([input_sequence, next_token.unsqueeze(0)], dim=1)

                # Check if EOS token generated
                if next_token.item() == eos_token:
                    break

        # Extract generated path (excluding goal and start tokens, including any EOS)
        full_sequence = input_sequence[0].cpu().tolist()

        # Remove goal (first token) and start (second token)
        # The remaining should be the continuation of the path
        generated_tokens = full_sequence[2:]

        # Remove EOS token if present
        if generated_tokens and generated_tokens[-1] == eos_token:
            generated_tokens = generated_tokens[:-1]

        return generated_tokens

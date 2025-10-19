"""
Several architectural variants for middle node prediction that address potential learning issues.

Each variant tries a different approach to improve training stability and learning capacity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class VariantV1_ResidualUpsample(nn.Module):
    """
    Variant 1: Proper residual connections with layer normalization

    Key improvements:
    - LayerNorm before each conv for training stability
    - Proper residual connections with projection when needed
    - Separate prediction heads for each level
    - Better initialization strategy
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_iterations: int = 3,
        upscale_depth: int = 5,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_iterations = num_iterations
        self.upscale_depth = upscale_depth

        # Embeddings
        self.node_embedding = nn.Embedding(vocab_size, d_model)

        # Convolutional layers - separate norm and conv for dimension handling
        self.layer_norms = nn.ModuleList([
            nn.ModuleList([nn.LayerNorm(d_model) for _ in range(num_iterations)])
            for _ in range(upscale_depth)
        ])

        self.conv_layers = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Conv1d(d_model, d_model, kernel_size=3, padding=0),
                    nn.GELU(),
                    nn.Dropout(dropout)
                )
                for _ in range(num_iterations)
            ])
            for _ in range(upscale_depth)
        ])

        # Separate output projection for each upscale level
        self.output_projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, vocab_size)
            )
            for _ in range(upscale_depth)
        ])

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        """Xavier/Glorot initialization for better gradient flow"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.Conv1d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def interleave(self, seq):
        """Insert means between every two elements"""
        seq_orig = seq.clone()
        x = seq_orig
        b, c, s = x.shape
        means = (x[..., :-1] + x[..., 1:]) / 2
        stacked = torch.stack([x[..., :-1], means], dim=-1)
        interleaved = stacked.reshape(b, c, -1)
        new_seq = torch.cat([interleaved, x[..., -1:]], dim=-1)
        return new_seq

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        data = batch
        batch_tensor = data['input']
        batch_size = batch_tensor.size(0)

        # Embed start and end nodes
        start_emb = self.node_embedding(batch_tensor[:, 0])
        end_emb = self.node_embedding(batch_tensor[:, -1])

        intermediate_preds = []
        input_emb = torch.stack([start_emb, end_emb], dim=-1)

        for level in range(self.upscale_depth):
            # Interleave
            input_emb = self.interleave(input_emb)

            # Apply conv layers with residual connections
            for i in range(self.num_iterations):
                # Apply LayerNorm: [batch, d_model, seq] -> [batch, seq, d_model] -> norm -> [batch, d_model, seq]
                normed = self.layer_norms[level][i](input_emb.transpose(1, 2)).transpose(1, 2)

                # Apply conv - note it reduces sequence length by 2
                conv_out = self.conv_layers[level][i](normed)

                # Residual connection: add to the middle part of the input
                input_emb = conv_out + input_emb[:, :, 1:-1]

                # Re-attach start and end embeddings
                input_emb = torch.cat([
                    start_emb.unsqueeze(-1),
                    input_emb,
                    end_emb.unsqueeze(-1)
                ], dim=-1)

            # Project to vocabulary for this level
            # Reshape: [batch, d_model, seq_len] -> [batch, seq_len, d_model]
            input_emb_transposed = input_emb.transpose(1, 2)

            # Apply output projection: [batch, seq_len, d_model] -> [batch, seq_len, vocab_size]
            logits = self.output_projections[0](input_emb_transposed)

            # Transpose to [batch, vocab_size, seq_len]
            logits = logits.transpose(1, 2)

            intermediate_preds.append(logits)

        # Use last level predictions as final output
        result = {"logits": intermediate_preds[-1]}

        # Compute hierarchical loss with proper weighting
        if all(key in data for key in range(self.upscale_depth)):
            total_loss = 0
            # Weight earlier levels less (they're easier)
            weights = [0.5 ** (self.upscale_depth - i - 1) for i in range(self.upscale_depth)]
            weights = [w / sum(weights) for w in weights]  # Normalize

            for i in range(self.upscale_depth):
                labels = data[i]
                logits = intermediate_preds[i]
                loss = F.cross_entropy(logits, labels, ignore_index=-100)
                total_loss += weights[i] * loss

            result["loss"] = total_loss

        return result


class VariantV2_TransformerBased(nn.Module):
    """
    Variant 2: Use transformer layers instead of convolutions

    Key improvements:
    - Self-attention to model long-range dependencies
    - Positional encodings for sequence position awareness
    - More powerful modeling capacity
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_iterations: int = 3,
        upscale_depth: int = 5,
        nhead: int = 4,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_iterations = num_iterations
        self.upscale_depth = upscale_depth

        # Embeddings
        self.node_embedding = nn.Embedding(vocab_size, d_model)

        # Transformer layers for each level
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                activation='gelu',
                batch_first=True,
                norm_first=True
            )
            for _ in range(upscale_depth)
        ])

        # Output projections
        self.output_projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, vocab_size)
            )
            for _ in range(upscale_depth)
        ])

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def interleave(self, seq):
        """Insert means between every two elements"""
        x = seq  # [batch, seq_len, d_model]
        b, s, d = x.shape
        means = (x[:, :-1, :] + x[:, 1:, :]) / 2
        stacked = torch.stack([x[:, :-1, :], means], dim=2)  # [batch, s-1, 2, d_model]
        interleaved = stacked.reshape(b, -1, d)  # [batch, (s-1)*2, d_model]
        new_seq = torch.cat([interleaved, x[:, -1:, :]], dim=1)
        return new_seq

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        data = batch
        batch_tensor = data['input']
        batch_size = batch_tensor.size(0)

        # Embed start and end nodes
        start_emb = self.node_embedding(batch_tensor[:, 0])
        end_emb = self.node_embedding(batch_tensor[:, -1])

        intermediate_preds = []
        # [batch, 2, d_model]
        input_emb = torch.stack([start_emb, end_emb], dim=1)

        for level in range(self.upscale_depth):
            # Interleave
            input_emb = self.interleave(input_emb)

            # Apply transformer
            for _ in range(self.num_iterations):
                input_emb = self.transformer_layers[level](input_emb)

            # Project to vocabulary
            logits = self.output_projections[level](input_emb)  # [batch, seq_len, vocab_size]
            logits = logits.transpose(1, 2)  # [batch, vocab_size, seq_len]

            intermediate_preds.append(logits)

        result = {"logits": intermediate_preds[-1]}

        # Compute loss
        if all(key in data for key in range(self.upscale_depth)):
            total_loss = 0
            for i in range(self.upscale_depth):
                labels = data[i]
                logits = intermediate_preds[i]
                loss = F.cross_entropy(logits, labels, ignore_index=-100)
                total_loss += loss

            result["loss"] = total_loss / self.upscale_depth

        return result


class VariantV3_UNetStyle(nn.Module):
    """
    Variant 3: U-Net style architecture with skip connections across levels

    Key improvements:
    - Skip connections from earlier levels to later levels
    - Preserves fine-grained information
    - Better gradient flow through the network
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_iterations: int = 3,
        upscale_depth: int = 5,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_iterations = num_iterations
        self.upscale_depth = upscale_depth

        # Embeddings
        self.node_embedding = nn.Embedding(vocab_size, d_model)

        # Encoder path - separate norm and conv
        self.encoder_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(upscale_depth)])
        self.encoder_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.GELU(),
                nn.Dropout(dropout)
            )
            for _ in range(upscale_depth)
        ])

        # Skip connection projections (to match dimensions if needed)
        self.skip_projections = nn.ModuleList([
            nn.Identity()  # Can be replaced with projections if needed
            for _ in range(upscale_depth)
        ])

        # Output projections for each level
        self.output_projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model // 2),
                nn.GELU(),
                nn.Linear(d_model // 2, vocab_size)
            )
            for _ in range(upscale_depth)
        ])

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)

    def interleave(self, seq):
        """Insert means between every two elements"""
        x = seq
        b, c, s = x.shape
        means = (x[..., :-1] + x[..., 1:]) / 2
        stacked = torch.stack([x[..., :-1], means], dim=-1)
        interleaved = stacked.reshape(b, c, -1)
        new_seq = torch.cat([interleaved, x[..., -1:]], dim=-1)
        return new_seq

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        data = batch
        batch_tensor = data['input']
        batch_size = batch_tensor.size(0)

        # Embed start and end nodes
        start_emb = self.node_embedding(batch_tensor[:, 0])
        end_emb = self.node_embedding(batch_tensor[:, -1])

        intermediate_preds = []
        skip_connections = []  # Store features from each level

        input_emb = torch.stack([start_emb, end_emb], dim=-1)

        for level in range(self.upscale_depth):
            # Interleave
            input_emb = self.interleave(input_emb)

            # Process with encoder (norm then conv)
            normed = self.encoder_norms[level](input_emb.transpose(1, 2)).transpose(1, 2)
            processed = self.encoder_convs[level](normed)

            # Add skip connection from previous level if exists
            if level > 0 and skip_connections:
                # Need to match dimensions
                prev_skip = skip_connections[-1]
                # Upsample previous skip connection to match current size
                if prev_skip.shape[-1] != processed.shape[-1]:
                    # Simple linear interpolation
                    prev_skip = F.interpolate(prev_skip, size=processed.shape[-1], mode='linear', align_corners=False)
                processed = processed + 0.3 * self.skip_projections[level](prev_skip)

            # Residual connection
            input_emb = processed + input_emb

            # Store for skip connections
            skip_connections.append(input_emb.clone())

            # Project to vocabulary
            input_emb_transposed = input_emb.transpose(1, 2)
            logits = self.output_projections[level](input_emb_transposed)
            logits = logits.transpose(1, 2)

            intermediate_preds.append(logits)

        result = {"logits": intermediate_preds[-1]}

        # Compute loss with curriculum weighting
        if all(key in data for key in range(self.upscale_depth)):
            total_loss = 0
            # Progressive weights: earlier levels weighted more initially
            for i in range(self.upscale_depth):
                labels = data[i]
                logits = intermediate_preds[i]
                loss = F.cross_entropy(logits, labels, ignore_index=-100)
                # Equal weighting for now
                total_loss += loss

            result["loss"] = total_loss / self.upscale_depth

        return result


class VariantV4_SimpleProgressive(nn.Module):
    """
    Variant 4: Simplified progressive approach with shared parameters

    Key improvements:
    - Shared conv layers across levels (parameter efficiency)
    - Simpler architecture, easier to train
    - Strong regularization through parameter sharing
    - Larger hidden dimension for more capacity
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,  # Larger by default
        num_iterations: int = 2,
        upscale_depth: int = 5,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_iterations = num_iterations
        self.upscale_depth = upscale_depth

        # Embeddings with larger dimension
        self.node_embedding = nn.Embedding(vocab_size, d_model)

        # Shared processing layers (parameter efficient)
        self.layer_norm = nn.LayerNorm(d_model)
        self.shared_conv = nn.Sequential(
            nn.Conv1d(d_model, d_model * 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(d_model * 2, d_model, kernel_size=1),
            nn.Dropout(dropout)
        )

        # Level-specific output heads
        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, vocab_size)
            )
            for _ in range(upscale_depth)
        ])

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def interleave(self, seq):
        """Insert means between every two elements"""
        x = seq
        b, c, s = x.shape
        means = (x[..., :-1] + x[..., 1:]) / 2
        stacked = torch.stack([x[..., :-1], means], dim=-1)
        interleaved = stacked.reshape(b, c, -1)
        new_seq = torch.cat([interleaved, x[..., -1:]], dim=-1)
        return new_seq

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        data = batch
        batch_tensor = data['input']
        batch_size = batch_tensor.size(0)

        # Embed start and end nodes
        start_emb = self.node_embedding(batch_tensor[:, 0])
        end_emb = self.node_embedding(batch_tensor[:, -1])

        intermediate_preds = []
        input_emb = torch.stack([start_emb, end_emb], dim=-1)

        for level in range(self.upscale_depth):
            # Interleave
            input_emb = self.interleave(input_emb)

            # Apply shared conv multiple times with residuals
            for _ in range(self.num_iterations):
                # Apply layer norm after transpose
                normed = self.layer_norm(input_emb.transpose(1, 2)).transpose(1, 2)
                processed = self.shared_conv(normed)
                input_emb = input_emb + processed

            # Level-specific prediction
            input_emb_transposed = input_emb.transpose(1, 2)
            logits = self.output_heads[level](input_emb_transposed)
            logits = logits.transpose(1, 2)

            intermediate_preds.append(logits)

        result = {"logits": intermediate_preds[-1]}

        # Compute loss - focus on final levels more
        if all(key in data for key in range(self.upscale_depth)):
            total_loss = 0
            # Exponentially increasing weights for later levels
            weights = [2 ** i for i in range(self.upscale_depth)]
            weights = [w / sum(weights) for w in weights]

            for i in range(self.upscale_depth):
                labels = data[i]
                logits = intermediate_preds[i]
                loss = F.cross_entropy(logits, labels, ignore_index=-100)
                total_loss += weights[i] * loss

            result["loss"] = total_loss

        return result


class VariantV5_GRUBased(nn.Module):
    """
    Variant 5: Use GRU for sequential processing

    Key improvements:
    - GRU for better sequential modeling
    - Bidirectional processing for context from both directions
    - Attention mechanism for focusing on relevant parts
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_iterations: int = 3,
        upscale_depth: int = 5,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_iterations = num_iterations
        self.upscale_depth = upscale_depth

        # Embeddings
        self.node_embedding = nn.Embedding(vocab_size, d_model)

        # Bidirectional GRU for each level
        self.gru_layers = nn.ModuleList([
            nn.GRU(
                input_size=d_model,
                hidden_size=d_model // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
                dropout=0.0  # We add dropout separately
            )
            for _ in range(upscale_depth)
        ])

        # Attention layers
        self.attention_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.Tanh(),
                nn.Linear(d_model, 1)
            )
            for _ in range(upscale_depth)
        ])

        self.dropout = nn.Dropout(dropout)

        # Output projections
        self.output_projections = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, vocab_size)
            )
            for _ in range(upscale_depth)
        ])

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.xavier_uniform_(module.weight)
            elif isinstance(module, nn.GRU):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)

    def interleave(self, seq):
        """Insert means between every two elements"""
        x = seq  # [batch, seq_len, d_model]
        b, s, d = x.shape
        means = (x[:, :-1, :] + x[:, 1:, :]) / 2
        stacked = torch.stack([x[:, :-1, :], means], dim=2)
        interleaved = stacked.reshape(b, -1, d)
        new_seq = torch.cat([interleaved, x[:, -1:, :]], dim=1)
        return new_seq

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        data = batch
        batch_tensor = data['input']
        batch_size = batch_tensor.size(0)

        # Embed start and end nodes
        start_emb = self.node_embedding(batch_tensor[:, 0])
        end_emb = self.node_embedding(batch_tensor[:, -1])

        intermediate_preds = []
        input_emb = torch.stack([start_emb, end_emb], dim=1)

        for level in range(self.upscale_depth):
            # Interleave
            input_emb = self.interleave(input_emb)

            # Apply GRU
            for _ in range(self.num_iterations):
                gru_out, _ = self.gru_layers[level](input_emb)
                gru_out = self.dropout(gru_out)

                # Self-attention
                attn_weights = self.attention_layers[level](gru_out)
                attn_weights = F.softmax(attn_weights, dim=1)

                # Weighted combination
                attended = gru_out * attn_weights

                # Residual
                input_emb = input_emb + attended

            # Project to vocabulary
            logits = self.output_projections[level](input_emb)
            logits = logits.transpose(1, 2)

            intermediate_preds.append(logits)

        result = {"logits": intermediate_preds[-1]}

        # Compute loss
        if all(key in data for key in range(self.upscale_depth)):
            total_loss = 0
            for i in range(self.upscale_depth):
                labels = data[i]
                logits = intermediate_preds[i]
                loss = F.cross_entropy(logits, labels, ignore_index=-100)
                total_loss += loss

            result["loss"] = total_loss / self.upscale_depth

        return result

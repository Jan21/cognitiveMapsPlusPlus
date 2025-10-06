import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class Conv1dBlock(nn.Module):
    """Convolutional block with normalization and activation"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, channels, seq_len)
        x = self.conv(x)
        # Transpose for LayerNorm: (batch_size, seq_len, channels)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        # Transpose back: (batch_size, channels, seq_len)
        x = x.transpose(1, 2)
        return x


class DownBlock(nn.Module):
    """Downsampling block with two conv layers and pooling"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.conv1 = Conv1dBlock(in_channels, out_channels, kernel_size, dropout)
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, dropout)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        # Keep residual before pooling for skip connection
        x = self.conv1(x)
        x = self.conv2(x)
        skip = x
        x = self.pool(x)
        return x, skip


class UpBlock(nn.Module):
    """Upsampling block with skip connection"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.upsample = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        # Double in_channels because of skip connection concatenation
        self.conv1 = Conv1dBlock(out_channels * 2, out_channels, kernel_size, dropout)
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, dropout)

    def forward(self, x, skip):
        # Upsample
        x = self.upsample(x)

        # Handle size mismatch between upsampled and skip
        if x.size(2) != skip.size(2):
            # Pad or crop to match skip connection size
            diff = skip.size(2) - x.size(2)
            if diff > 0:
                x = F.pad(x, (diff // 2, diff - diff // 2))
            else:
                x = x[:, :, :skip.size(2)]

        # Concatenate skip connection
        x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)

        x = self.conv2(x)
        return x


class UNet1D(nn.Module):
    """U-Net architecture for 1D sequence modeling"""
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        num_layers: int = 3,
        kernel_size: int = 3,
        max_seq_length: int = 128,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.mask_token_id = vocab_size - 1
        # Token and positional embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)

        # Initial projection to first hidden dimension
        base_channels = d_model

        # Encoder (downsampling path)
        self.down_blocks = nn.ModuleList()
        in_channels = base_channels
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            self.down_blocks.append(DownBlock(in_channels, out_channels, kernel_size, dropout))
            in_channels = out_channels

        # Bottleneck
        bottleneck_channels = base_channels * (2 ** num_layers)
        self.bottleneck = nn.Sequential(
            Conv1dBlock(in_channels, bottleneck_channels, kernel_size, dropout),
            Conv1dBlock(bottleneck_channels, bottleneck_channels, kernel_size, dropout),
        )

        # Decoder (upsampling path)
        self.up_blocks = nn.ModuleList()
        in_channels = bottleneck_channels
        for i in range(num_layers - 1, -1, -1):
            out_channels = base_channels * (2 ** i)
            self.up_blocks.append(UpBlock(in_channels, out_channels, kernel_size, dropout))
            in_channels = out_channels

        # Output projection
        self.output_projection = nn.Linear(base_channels, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.last_token_proj = nn.Linear(d_model, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

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
        device = input_ids.device

        # Embed tokens
        token_embeds = self.token_embedding(input_ids) * math.sqrt(self.d_model)

        # Add positional embeddings
        #positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        #pos_embeds = self.position_embedding(positions)

        # Combine embeddings: (batch_size, seq_len, d_model)
        x = token_embeds #+ pos_embeds
        # Take the embedding of the last token in each sequence

        # Transpose for convolutions: (batch_size, d_model, seq_len)
        x = x.transpose(1, 2)

        # Encoder path with skip connections
        skips = []
        for down_block in self.down_blocks:
            x, skip = down_block(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder path with skip connections
        for up_block, skip in zip(self.up_blocks, reversed(skips)):
            x = up_block(x, skip)

        # Transpose back: (batch_size, seq_len, d_model)
        x = x.transpose(1, 2)

        # Project to vocabulary
        logits = self.output_projection(x)

        # Compute loss if targets provided
        result = {"logits": logits}
        if target_ids is not None:
            loss = self.get_loss(logits, target_ids, input_ids)
            result["loss"] = loss

        return result

    def get_loss(self, logits, targets, masked_input_ids):
        """
        Compute cross-entropy loss

        Args:
            logits: (batch_size, seq_len, vocab_size)
            targets: (batch_size, seq_len) with -100 for padding tokens

        Returns:
            loss: scalar tensor
        """
        # Reshape for loss computation
        # Ignore the first and last token in each sequence for loss computation
        # Only compute loss for masked tokens (mask token id is 402 in masked_input_ids)
        # We want to compute loss only for positions where masked_input_ids == 402
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

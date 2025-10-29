"""
Several architectural variants for middle node prediction that address potential learning issues.

Each variant tries a different approach to improve training stability and learning capacity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class Diffusion_ResidualUpsample(nn.Module):
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
        stacked = torch.stack([x[..., :-1], means], dim=-1).contiguous()
        interleaved = stacked.reshape(b, c, -1)
        new_seq = torch.cat([interleaved, x[..., -1:]], dim=-1).contiguous()
        return new_seq

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        data = batch
        batch_tensor = data['input_ids']
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
                input_emb_for_norm = input_emb.transpose(1, 2)
                normed_output = self.layer_norms[0][i](input_emb_for_norm)
                normed = normed_output.transpose(1, 2)

                # Apply conv - note it reduces sequence length by 2
                conv_out = self.conv_layers[0][i](normed)

                # Residual connection: add to the middle part of the input
                input_emb_slice = input_emb[:, :, 1:-1]
                input_emb = (conv_out + input_emb_slice)

                # Re-attach start and end embeddings
                input_emb = torch.cat([
                    start_emb.unsqueeze(-1),
                    input_emb,
                    end_emb.unsqueeze(-1)
                ], dim=-1)

            # Project to vocabulary for this level
            # Reshape: [batch, d_model, seq_len] -> [batch, seq_len, d_model]
            input_emb_transposed = input_emb.transpose(1, 2)

            # Apply output projection: [batch, seq_len, d_model] -> [batch, seq_len, .contiguous()

            # Transpose to [batch, vocab_size, seq_len].contiguous()
            logits = self.output_projections[0](input_emb_transposed).transpose(1, 2)
            intermediate_preds.append(logits)

        diffs = input_emb_transposed[:,:-1,:]-input_emb_transposed[:,1:,:]
        norm_diffs = diffs.norm(dim=-1)
        sum_norm_diffs = norm_diffs.sum(dim=-1)
        result = {"logits": intermediate_preds[-1].transpose(1, 2)}
        result["sum_norm_diffs"] = sum_norm_diffs
        return result


    def get_loss(self, result, data):
        lens = data['len']
        logits = result['logits'].transpose(1, 2)
        sum_norm_diffs = result['sum_norm_diffs']
        # Compute hierarchical loss with proper weighting
        if all(key in data for key in range(self.upscale_depth)):
            total_loss = 0
            # Weight earlier levels less (they're easier)
            weights = [0.5 ** (self.upscale_depth - i - 1) for i in range(self.upscale_depth)]
            weights = [w / sum(weights) for w in weights]  # Normalize

            # for i in range(self.upscale_depth):
            #     labels = data[i]
            #     logits = intermediate_preds[i]
            #     loss = F.cross_entropy(logits, labels, ignore_index=-100)
            #     total_loss += weights[i] * loss
            labels = data[4].contiguous()
            total_loss = F.cross_entropy(logits, labels, ignore_index=-100)
            lens = data['len'].float()
            mse_sum_norm_diffs = F.mse_loss(sum_norm_diffs.float(), lens.float())
            mse_sum_norm_diffs = 0
            result["loss"] = total_loss #+ mse_sum_norm_diffs
            result["mse_sum_norm_diffs"] = mse_sum_norm_diffs

        return result



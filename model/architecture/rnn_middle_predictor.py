import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class RNNMiddleNodePredictor(nn.Module):
    """
    RNN-based middle-node predictor.

    Architecture:
    1. Embed start and end nodes
    2. Concatenate embeddings to form initial input
    3. Run N recurrent updates on hidden state
    4. Project final hidden state to predict middle node ID

    Args:
        vocab_size: Size of node vocabulary
        d_model: Embedding and hidden state dimension
        num_iterations: Number of recurrent updates
        dropout: Dropout probability
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_iterations: int = 6,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_iterations = num_iterations

        # Embeddings for start and end nodes
        self.node_embedding = nn.Embedding(vocab_size, d_model)

        # RNN cell that takes concatenated input (2*d_model) and updates hidden state
        self.rnn_cell = nn.LSTMCell( d_model, d_model)

        # Output projection to predict middle node
        self.output_projection = nn.Linear(d_model, vocab_size)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with small random values"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, start_ids: torch.Tensor, end_ids: torch.Tensor,
                middle_ids: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the RNN.

        Args:
            start_ids: Start node IDs [batch_size]
            end_ids: End node IDs [batch_size]
            middle_ids: Ground truth middle node IDs [batch_size] (optional, for loss)

        Returns:
            Dictionary with 'logits' [batch_size, vocab_size] and optionally 'loss'
        """
        batch_size = start_ids.size(0)
        device = start_ids.device

        # Embed start and end nodes
        start_emb = self.node_embedding(start_ids)  # [batch_size, d_model]
        end_emb = self.node_embedding(end_ids)      # [batch_size, d_model]

        # Concatenate embeddings to form input
        input_emb = (start_emb + end_emb) / 2  # [batch_size, d_model]

        # Initialize hidden and cell states
        h = torch.zeros(batch_size, self.d_model, device=device)
        c = torch.zeros(batch_size, self.d_model, device=device)

        # Run recurrent updates
        for _ in range(1):
            h, c = self.rnn_cell(input_emb, (h, c))
            h = self.dropout(h)

        # Project to logits
        logits = self.output_projection(h)  # [batch_size, vocab_size]

        result = {"logits": logits}

        # Compute loss if targets provided
        if middle_ids is not None:
            loss = F.cross_entropy(logits, middle_ids)
            result["loss"] = loss

        return result

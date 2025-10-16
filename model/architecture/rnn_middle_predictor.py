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
        self.num_iterations = 1
        self.upscale_depth = 5
        # INSERT_YOUR_CODE
        self.conv_layers = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=3)
            for _ in range(self.upscale_depth)
        ])

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


    def interleave(self, seq):
        # Parallel (vectorized) insertion of means between every two elements along the last dimension
        seq_orig = seq.clone()
        x = seq_orig  # shape: (batch_size, in_channels, sequence_length)
        b,c,s = x.shape
        means = (x[..., :-1] + x[..., 1:]) / 2    # shape: (..., s-1)
        # Stack original elements (except last) and means for interleaving
        stacked = torch.stack([x[..., :-1], means], dim=-1)  # shape: (..., s-1, 2)
        # Reshape to (..., (s-1)*2) to interleave
        interleaved = stacked.reshape(b,c,-1)
        # Append the last element
        new_seq = torch.cat([interleaved, x[..., -1:]], dim=-1)
        return new_seq


    def forward(self, batch) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the RNN.

        Args:
            start_ids: Start node IDs [batch_size]
            end_ids: End node IDs [batch_size]
            middle_ids: Ground truth middle node IDs [batch_size] (optional, for loss)

        Returns:
            Dictionary with 'logits' [batch_size, vocab_size] and optionally 'loss'
        """
        data = batch
        batch = data['input']
        batch_size = batch.size(0)
        device = batch.device

        # Embed start and end nodes
        start_emb = self.node_embedding(batch[:,0])  # [batch_size, d_model]
        end_emb = self.node_embedding(batch[:,-1])      # [batch_size, d_model]
        # Concatenate start and end embeddings along the last dimension
        intermediate_preds = []
        input_emb = torch.stack([start_emb, end_emb], dim=-1)  # [batch_size, d_model * 2]
        for i in range(self.upscale_depth):
            # Interleave the input embeddings
            input_emb = self.interleave(input_emb)  # [batch_size, d_model * 2 * (sequence_length - 1)]
            # Apply the convolutional layer to the input embeddings
            for i in range(self.num_iterations):
                input_emb = self.conv_layers[i](input_emb) + input_emb[:,:,1:-1]
                input_emb = torch.cat([start_emb.unsqueeze(-1), input_emb, end_emb.unsqueeze(-1)], dim=-1).squeeze(-2)
            
        # Reshape input_emb to [batch_size, -1] if necessary before projecting
             # [batch_size, d_model * 2 * (sequence_length - 1) * 5]
            reshaped_input_emb = input_emb.reshape(-1, self.d_model)
            preds = self.output_projection(reshaped_input_emb).reshape(batch_size, self.vocab_size,-1)
            intermediate_preds.append(preds)
  # [batch_size, vocab_size]

        result = {"logits": preds}
        # Compute loss if targets provided
        loss = 0
        for i in range(self.upscale_depth):
            loss += F.cross_entropy(intermediate_preds[i], data[i])
        
        result["loss"] = loss

        return result

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import matmul
from typing import Dict, Any


class VertexToEdgeLayer(nn.Module):
    """Message passing from vertex nodes to edge nodes"""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.updater = nn.LSTM(d_model, d_model)

    def forward(self, adj_t, x_v, hidden):
        """
        Args:
            adj_t: Sparse adjacency matrix [num_edges, num_vertices]
            x_v: Vertex embeddings [num_vertices, d_model]
            hidden: Tuple of (h_e, c_e) for edge hidden states

        Returns:
            Updated [h_e, c_e]
        """
        # Aggregate messages from vertices to edges
        msg = matmul(adj_t, x_v)  # [num_edges, d_model]

        # Update with LSTM
        hidden = (hidden[0].unsqueeze(0), hidden[1].unsqueeze(0))
        msg, new_hidden = self.updater(msg.unsqueeze(0), hidden)
        return [new_hidden[0].squeeze(0), new_hidden[1].squeeze(0)]


class EdgeToVertexLayer(nn.Module):
    """Message passing from edge nodes to vertex nodes"""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.updater = nn.LSTM(d_model, d_model)

    def forward(self, adj_t, x_e, hidden, v_batch):
        """
        Args:
            adj_t: Sparse adjacency matrix [num_edges, num_vertices]
            x_e: Edge embeddings [num_edges, d_model]
            hidden: Tuple of (h_v, c_v) for vertex hidden states
            v_batch: Batch assignment for vertices

        Returns:
            Updated [h_v, c_v]
        """
        # Aggregate messages from edges to vertices
        msg = matmul(adj_t.t(), x_e)  # [num_vertices, d_model]

        # Current vertex hidden state
        x_v = hidden[0]

        # Update with LSTM
        hidden = (x_v.unsqueeze(0), hidden[1].unsqueeze(0))
        msg, new_hidden = self.updater(msg.unsqueeze(0), hidden)
        return [new_hidden[0].squeeze(0), new_hidden[1].squeeze(0)]


class BipartiteGNN(nn.Module):
    """
    Bipartite Graph Neural Network for shortest path prediction.

    Similar to GNN_SAT but for path prediction:
    - Vertex nodes (unknowns) = waypoints in the path
    - Edge nodes (clauses) = connections between consecutive vertices
    - Uses message passing with sparse adjacency matrices
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

        # Message passing layers
        self.vertex_to_edge = VertexToEdgeLayer(d_model)
        self.edge_to_vertex = EdgeToVertexLayer(d_model)

        # Output projection to predict vertex IDs
        self.output_projection = nn.Linear(d_model, vocab_size)

        # Initialization layers
        self.vertex_embedding = nn.Embedding(vocab_size + 1, d_model, padding_idx=vocab_size)
        self.edge_init = nn.Linear(1, d_model)

        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def init_embeddings(self, data, device):
        """
        Initialize embeddings for vertices and edges.

        Args:
            data: Batch with x_v, x_e, x_v_batch, x_e_batch
            device: Device to place tensors on

        Returns:
            vertex_hidden: (h_v, c_v)
            edge_hidden: (h_e, c_e)
        """
        # Map -1 (unknown) to vocab_size for embedding
        x_v_mapped = data.x_v.clone()
        x_v_mapped[x_v_mapped == -1] = self.vocab_size

        # Initialize vertex embeddings
        h_v = self.vertex_embedding(x_v_mapped.squeeze(-1))
        c_v = torch.zeros_like(h_v)

        # Initialize edge embeddings (simple initialization)
        init_ts = torch.ones(1, device=device)
        edge_init = self.edge_init(init_ts)
        h_e = edge_init.repeat(data.x_e.size(0), 1)
        c_e = torch.zeros_like(h_e)

        return [h_v, c_v], [h_e, c_e]

    def forward(self, data, num_iters=None):
        """
        Forward pass through the bipartite GNN.

        Args:
            data: Batch containing:
                - x_v: Vertex features [total_vertices, 1]
                - x_e: Edge features [total_edges, 2]
                - x_v_batch: Batch assignment for vertices
                - x_e_batch: Batch assignment for edges
                - adj_t: Sparse adjacency matrix [total_edges, total_vertices]
                - targets: Ground truth vertex IDs
                - target_mask: Which vertices to predict

        Returns:
            Dictionary with 'logits' and 'loss'
        """
        device = data.x_v.device
        num_iters = num_iters or self.num_iterations

        # Initialize embeddings
        vertex_hidden, edge_hidden = self.init_embeddings(data, device)

        # Get batch assignments and adjacency
        v_batch = data.x_v_batch
        adj_t = data.adj_t

        # Iterative message passing
        for _ in range(num_iters):
            # Vertex -> Edge
            edge_hidden = self.vertex_to_edge(adj_t, vertex_hidden[0], edge_hidden)
            edge_hidden[0] = self.dropout(edge_hidden[0])

            # Edge -> Vertex
            vertex_hidden = self.edge_to_vertex(adj_t, edge_hidden[0], vertex_hidden, v_batch)
            vertex_hidden[0] = self.dropout(vertex_hidden[0])

        # Predict vertex IDs
        logits = self.output_projection(vertex_hidden[0])

        result = {"logits": logits}

        # Compute loss if targets provided
        if hasattr(data, 'targets') and data.targets is not None:
            loss = self.compute_loss(logits, data.targets, data.target_mask)
            result["loss"] = loss

        return result

    def compute_loss(self, logits: torch.Tensor, targets: torch.Tensor,
                    target_mask: torch.Tensor) -> torch.Tensor:
        """Compute cross-entropy loss for predicted vertices."""
        targets_masked = torch.where(
            target_mask,
            targets,
            torch.full_like(targets, -100)
        )

        loss = F.cross_entropy(logits, targets_masked, ignore_index=-100)
        return loss

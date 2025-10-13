import torch
from typing import Dict, Any

try:
    from torch_geometric.data import Batch
    PYTORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    PYTORCH_GEOMETRIC_AVAILABLE = False
    Batch = None

from .base_lightningmodule import BasePathPredictionModule


class GNNPathPredictionModule(BasePathPredictionModule):
    """
    Lightning module for GNN-based middle-node prediction.

    This module predicts the middle node of a path given the start and end nodes.
    Uses a bipartite graph with:
    - 3 vertex nodes: start (fixed), middle (to predict), end (fixed)
    - 1 constraint node connecting all 3 vertices
    """

    def __init__(
        self,
        model_config: Dict[str, Any],
        vocab_size: int,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000,
        optimizer: str = "adamw",
        graph_type: str = "sphere",
        graph_path: str = "temp/sphere_graph.pkl",
    ):
        if not PYTORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "PyTorch Geometric is required for GNN models. "
                "Install it with: pip install torch-geometric"
            )

        super().__init__(
            model_config=model_config,
            vocab_size=vocab_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            optimizer=optimizer,
            graph_type=graph_type,
            graph_path=graph_path
        )

        # Mask token for unknown middle nodes
        self.mask_token_id = vocab_size - 1

    def forward(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward pass through the GNN model"""
        return self.model(batch)

    def training_step(self, batch: Batch, batch_idx: int):
        """
        Training step for middle-node prediction.

        The middle node is always masked during training.
        Start and end nodes are always visible.
        """
        # Clone batch and mask the middle node
        masked_batch = batch.clone()

        # Mask middle nodes (index 1 in each group of 3 vertices)
        for graph_idx in range(batch.num_graphs):
            v_mask = (batch.x_v_batch == graph_idx)
            graph_vertex_indices = torch.where(v_mask)[0]

            # Middle node is at index 1 (after start at 0, before end at 2)
            middle_vertex_idx = graph_vertex_indices[1]
            masked_batch.x_v[middle_vertex_idx, 0] = self.mask_token_id

        # Forward pass with masked middle nodes
        output = self.model(masked_batch)
        loss = output["loss"]

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        self._log_learning_rate()

        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        """
        Validation step for middle-node prediction.

        Mask the middle node and predict it using the model.
        """
        # Create a masked version with middle nodes masked
        masked_batch = batch.clone()

        for graph_idx in range(batch.num_graphs):
            v_mask = (batch.x_v_batch == graph_idx)
            graph_vertex_indices = torch.where(v_mask)[0]

            # Mask middle node (index 1)
            middle_vertex_idx = graph_vertex_indices[1]
            masked_batch.x_v[middle_vertex_idx, 0] = self.mask_token_id

        # Forward pass to get predictions
        output = self.model(masked_batch)
        logits = output["logits"]  # [total_vertices, vocab_size]

        # Get predictions (argmax)
        predictions = logits.argmax(dim=-1)  # [total_vertices]
        targets = batch.targets  # [total_vertices]

        # Compute metrics - only for middle nodes
        batch_size = batch.num_graphs
        total_correct = 0
        total_tokens = 0
        total_distance = 0.0

        for graph_idx in range(batch_size):
            v_mask = (batch.x_v_batch == graph_idx)
            graph_vertex_indices = torch.where(v_mask)[0]

            # Get start, middle, and end node IDs
            start_vertex_idx = graph_vertex_indices[0]
            middle_vertex_idx = graph_vertex_indices[1]
            end_vertex_idx = graph_vertex_indices[2]

            start_id = batch.x_v[start_vertex_idx, 0].item()
            end_id = batch.x_v[end_vertex_idx, 0].item()

            # Get prediction and target
            pred = predictions[middle_vertex_idx].item()
            target = targets[middle_vertex_idx].item()

            if pred == target:
                total_correct += 1
            total_tokens += 1

            # Compute distance from prediction to actual middle point
            actual_middle = self._find_middle_point(start_id, end_id)
            distance = self._compute_graph_distance(pred, actual_middle)
            total_distance += distance

        # Compute metrics
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0.0
        avg_distance = total_distance / batch_size if batch_size > 0 else 0.0

        self.log('val_exact_match', accuracy, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('val_distance_to_middle', avg_distance, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return accuracy

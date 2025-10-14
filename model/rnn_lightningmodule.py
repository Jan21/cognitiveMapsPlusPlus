import torch
from typing import Dict, Any

from .base_lightningmodule import BasePathPredictionModule
from .architecture.rnn_middle_predictor import RNNMiddleNodePredictor


class RNNMiddleNodeModule(BasePathPredictionModule):
    """
    Lightning module for RNN-based middle-node prediction.

    This module predicts the middle node of a path given the start and end nodes.
    Uses an RNN that:
    1. Takes concatenated embeddings of start and end nodes as input
    2. Runs several recurrent updates on the hidden state
    3. Projects the final hidden state to predict the middle node
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
        # Override model creation - use RNN architecture
        super().__init__(
            model_config=model_config,  # Empty, we'll create model manually
            vocab_size=vocab_size,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            optimizer=optimizer,
            graph_type=graph_type,
            graph_path=graph_path
        )

        # Create RNN model
        self.model = RNNMiddleNodePredictor(
            vocab_size=vocab_size,
            d_model=model_config.get('d_model', 128),
            num_iterations=model_config.get('num_iterations', 6),
            dropout=model_config.get('dropout', 0.1)
        )

    def forward(self, start_ids: torch.Tensor, end_ids: torch.Tensor,
                middle_ids: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Forward pass through the RNN model"""
        return self.model(start_ids, end_ids, middle_ids)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """
        Training step for middle-node prediction.

        Args:
            batch: Dictionary with 'start_id', 'end_id', 'middle_id'
        """
        start_ids = batch['start_id']
        end_ids = batch['end_id']
        middle_ids = batch['middle_id']

        # Forward pass
        output = self.model(start_ids, end_ids, middle_ids)
        loss = output["loss"]

        # Log metrics
        batch_size = start_ids.size(0)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self._log_learning_rate()

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """
        Validation step for middle-node prediction.

        Args:
            batch: Dictionary with 'start_id', 'end_id', 'middle_id'
        """
        start_ids = batch['start_id']
        end_ids = batch['end_id']
        middle_ids = batch['middle_id']

        # Forward pass
        output = self.model(start_ids, end_ids, middle_ids)
        logits = output["logits"]
        loss = output["loss"]

        # Get predictions (argmax)
        predictions = logits.argmax(dim=-1)

        # Compute accuracy
        correct = (predictions == middle_ids).sum().item()
        total = middle_ids.size(0)
        accuracy = correct / total

        # Compute distance to actual middle point
        batch_size = start_ids.size(0)
        total_distance = 0.0

        for i in range(batch_size):
            start_id = start_ids[i].item()
            end_id = end_ids[i].item()
            pred_id = predictions[i].item()

            # Find the actual middle point and compute distance
            actual_middle = self._find_middle_point2(start_id, end_id, pred_id)
            distance = self._compute_graph_distance(pred_id, actual_middle)
            total_distance += distance

        avg_distance = total_distance / batch_size if batch_size > 0 else 0.0

        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('val_exact_match', accuracy, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('val_distance_to_middle', avg_distance, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return accuracy

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Test step - same as validation"""
        return self.validation_step(batch, batch_idx)

import torch
from typing import Dict, Any

from .base_lightningmodule import BasePathPredictionModule
from .architecture.rnn_middle_predictor import RNNMiddleNodePredictor
from .architecture.variant_factory import create_model


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

        # Create model using variant factory or default to original
        variant = model_config.get('variant', 'original')
        if variant == 'original':
            # Use original implementation directly
            self.model = RNNMiddleNodePredictor(
                vocab_size=vocab_size,
                d_model=model_config.get('d_model', 128),
                num_iterations=model_config.get('num_iterations', 6),
                dropout=model_config.get('dropout', 0.1)
            )
        else:
            # Use variant factory
            self.model = create_model(
                variant=variant,
                vocab_size=vocab_size,
                d_model=model_config.get('d_model', 128),
                num_iterations=model_config.get('num_iterations', 3),
                upscale_depth=model_config.get('upscale_depth', 5),
                dropout=model_config.get('dropout', 0.1),
                nhead=model_config.get('nhead', 4)  # For transformer variant
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

        # Forward pass
        output = self.model(batch)
        loss = output["loss"]

        # Log metrics
        batch_size = batch['input'].size(0)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self._log_learning_rate()

        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """
        Validation step for middle-node prediction.

        Args:
            batch: Dictionary with 'start_id', 'end_id', 'middle_id'
        """


        # Forward pass
        output = self.model(batch)
        logits = output["logits"]
        loss = output["loss"]

        # Get predictions (argmax)
        predictions = logits.argmax(dim=-2)
        labels = batch[4]
        # Compute accuracy
        correct = (predictions == labels).sum().item()
        # INSERT_YOUR_CODE
        total = predictions.numel()
        accuracy = correct / total
        predictions[:,0] = batch['input'][:,0]
        predictions[:,-1] = batch['input'][:,-1]
        # INSERT_YOUR_CODE
        # Compute exact match accuracy: how many sequences match all labels exactly
        # predictions: [batch_size, seq_len], labels: [batch_size, seq_len]
        exact_matches = (predictions[:, 1:-1] == labels[:, 1:-1]).all(dim=-1).float()
        exact_match_accuracy = exact_matches.mean().item()



        # Compute distance to actual middle point
        batch_size = predictions.size(0)
        paths,validity = self._construct_and_validate_paths(predictions)
        lengths_gt = batch['len']
        total_diff = 0
        valid_count = 0.00000001
        for i in range(batch_size):
            if validity[i]:
                valid_count += 1
                path_length = len(paths[i])
                diff = abs(path_length - lengths_gt[i])
                total_diff += diff
        avg_diff = total_diff / valid_count
        validity_accuracy = sum(validity) / batch_size
        self.log('val_avg_diff', avg_diff, on_epoch=True, prog_bar=True, batch_size=valid_count)
        # Log metrics
        self.log('val_loss', loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('val_exact_match', exact_match_accuracy, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('val_validity', validity_accuracy, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return accuracy

    def _construct_and_validate_paths(self, predictions: torch.Tensor) -> tuple[list[list[int]], list[bool]]:
        """
        Construct paths from predictions by removing padding tokens and validate them.

        Args:
            predictions: Tensor of shape [batch_size, sequence_length] containing predicted node IDs
                        The padding token ID is vocab_size - 1

        Returns:
            tuple containing:
                - paths: List of paths, where each path is a list of node IDs (with padding removed)
                - is_valid: List of boolean values indicating whether each path is valid
        """
        pad_token_id = self.vocab_size - 1 # TODO fix this
        batch_size = predictions.size(0)

        paths = []
        is_valid = []

        for i in range(batch_size):
            # Get the prediction sequence for this example
            pred_seq = predictions[i]  # [sequence_length]

            # Remove padding tokens
            # Convert to list and filter out padding tokens
            path = [node_id.item() for node_id in pred_seq if node_id.item() != pad_token_id]

            # Validate the path using the graph
            valid = self._validate_path(path)

            paths.append(path)
            is_valid.append(valid)

        return paths, is_valid

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Test step - same as validation"""
        return self.validation_step(batch, batch_idx)

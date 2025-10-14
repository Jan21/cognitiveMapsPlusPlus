import pytorch_lightning as pl
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict, Any, Optional
import pickle
import networkx as nx

from .model_factory import create_model



class BasePathPredictionModule(pl.LightningModule):
    """
    Base Lightning module with shared functionality for all path prediction models.

    Handles:
    - Model creation
    - Graph loading
    - Optimizer and scheduler configuration
    - Common hyperparameters
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
        super().__init__()
        self.save_hyperparameters()

        # Model configuration
        self.model = create_model(model_config, vocab_size)
        self.vocab_size = vocab_size

        # Optimizer configuration
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.optimizer_name = optimizer

        # Load graph for path validation
        graph_path = graph_path.replace('.pkl', f'_{graph_type}.pkl')
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""

        # Default to AdamW
        optimizer = AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )

        # Warmup + cosine annealing scheduler
        def lr_lambda(step):
            if step < self.warmup_steps:
                # Linear warmup
                return step / self.warmup_steps
            else:
                # Cosine annealing after warmup with minimum lr of 0.1
                progress = (step - self.warmup_steps) / max(
                    1, self.trainer.estimated_stepping_batches - self.warmup_steps
                )
                cosine_factor = 0.5 * (1 + torch.cos(torch.tensor(torch.pi * min(progress, 1.0))))
                return 0.1 + 0.9 * cosine_factor

        scheduler = LambdaLR(optimizer, lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }

    def _log_learning_rate(self):
        """Helper to log current learning rate"""
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_step=True, on_epoch=False, prog_bar=False)

    def _validate_path(self, path: list) -> bool:
        """
        Validate if a path is valid in the graph.

        Args:
            path: List of vertex IDs

        Returns:
            True if all consecutive vertices are connected in the graph
        """
        if len(path) <= 1:
            return True

        for i in range(len(path) - 1):
            if not self.graph.has_edge(path[i], path[i + 1]):
                return False

        return True

    def _compute_graph_distance(self, node1: int, node2: int) -> float:
        """
        Compute the shortest path distance between two nodes in the graph.

        Args:
            node1: First node ID
            node2: Second node ID

        Returns:
            Shortest path distance (number of edges). Returns float('inf') if no path exists.
        """
        try:
            return nx.shortest_path_length(self.graph, node1, node2)
        except nx.NetworkXNoPath:
            return float('inf')

    def _find_middle_point(self, start: int, end: int, pred: int) -> int:
        """
        Find the middle point along the shortest path between start and end nodes.

        Args:
            start: Start node ID
            end: End node ID

        Returns:
            The node ID at the middle of the shortest path. If path length is even,
            returns the node just after the midpoint.
        """
        try:
            # Get all shortest paths between start and end
            all_paths = list(nx.all_shortest_paths(self.graph, start, end))
            if not all_paths:
                return start

            # For all shortest paths, collect all nodes that appear at a "middle" index
            candidate_set = set()
            for path in all_paths:
                mid_idx = len(path) // 2
                if not path:
                    continue
                candidate_set.add(path[mid_idx])
                if len(path) % 2 == 0 and mid_idx > 0:
                    candidate_set.add(path[mid_idx - 1])

            if not candidate_set:
                return start

            # Choose the candidate closest to pred
            min_dist = float('inf')
            closest_mid = None
            for node in candidate_set:
                try:
                    dist = nx.shortest_path_length(self.graph, node, pred)
                except nx.NetworkXNoPath:
                    dist = float('inf')
                if dist < min_dist:
                    min_dist = dist
                    closest_mid = node

            # Fallback if something went wrong (should not happen)
            if closest_mid is None:
                return start

            return closest_mid
        except nx.NetworkXNoPath:
            # If no path exists, return the start node
            return start


    def _find_middle_point2(self, start: int, end: int, pred: int) -> int:
        """
        Find the middle point on the shortest path between start and end nodes.

        Args:
            start: Start node ID
            end: End node ID

        Returns:
            The node ID at the middle of the shortest path (rounded down).
            If the path is length 1 or 2, returns start or end appropriately.
        """
        try:
            path = nx.shortest_path(self.graph, start, end)
            if not path:
                return start
            mid_idx = len(path) // 2
            return path[mid_idx]
        except nx.NetworkXNoPath:
            return start

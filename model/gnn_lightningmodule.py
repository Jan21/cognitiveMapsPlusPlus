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
    Lightning module for GNN-based diffusion path prediction.

    This module uses a diffusion process where vertex values are progressively
    masked and then unmasked through iterative denoising, using a bipartite
    graph where edges and vertices exchange messages through LSTM updates.
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

        # Diffusion-specific parameters
        self.mask_token_id = vocab_size - 1
        self.num_timesteps = 20
        self.scheduler = torch.linspace(
            1 / self.num_timesteps, 1, steps=self.num_timesteps, dtype=torch.float32, device='cuda'
        )

    def forward(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward pass through the GNN model"""
        return self.model(batch)

    def forward_noising(self, batch: Batch, t: torch.Tensor) -> Batch:
        """
        Apply noise to vertex values in the batch by masking them.

        First and last vertices are never masked (they are the boundary conditions).
        Intermediate vertices are masked with probability based on timestep t.

        Args:
            batch: PyTorch Geometric batch with x_v, x_e, x_v_batch, x_e_batch
            t: Timestep tensor [batch_size]

        Returns:
            Noised batch with some vertex features masked
        """
        # Clone the batch to avoid in-place modifications
        noised_batch = batch.clone()

        # Get mask probabilities for each graph in the batch
        mask_probs = self.scheduler[t]  # [batch_size]

        # Process each graph in the batch
        for graph_idx in range(batch.num_graphs):
            # Find vertices for this graph using x_v_batch
            v_mask = (batch.x_v_batch == graph_idx)
            num_vertices = v_mask.sum().item()

            # Get vertex indices for this graph
            graph_vertex_indices = torch.where(v_mask)[0]

            # Create mask: first and last vertices should NOT be masked
            should_mask = torch.ones(num_vertices, dtype=torch.bool, device=batch.x_v.device)
            should_mask[0] = False  # First vertex
            should_mask[-1] = False  # Last vertex

            # Bernoulli sampling to decide which vertices to mask
            will_mask = torch.bernoulli(
                torch.full((num_vertices,), mask_probs[graph_idx].item(), device=batch.x_v.device)
            ).to(dtype=torch.bool)

            # Only mask vertices that should be masked
            will_mask &= should_mask

            # Apply masking to vertex features
            for i, vertex_idx in enumerate(graph_vertex_indices):
                if will_mask[i]:
                    # Mask the vertex ID (first feature in x_v)
                    noised_batch.x_v[vertex_idx, 0] = self.mask_token_id

        return noised_batch

    def training_step(self, batch: Batch, batch_idx: int):
        """Training step with diffusion noise"""
        # Sample random timesteps for each graph in the batch
        t = torch.randint(
            0, len(self.scheduler),
            (batch.num_graphs,),
            device=batch.x_v.device
        )

        # Apply noise to the batch
        noised_batch = self.forward_noising(batch, t)

        # Forward pass with noised data
        output = self.model(noised_batch)
        loss = output["loss"]

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch.num_graphs)
        self._log_learning_rate()

        return loss

    def _get_sampling_timesteps(self, num_sampling_steps):
        """Get timesteps for sampling"""
        return torch.linspace(
            len(self.scheduler) - 1,
            len(self.scheduler) // num_sampling_steps,
            num_sampling_steps,
            dtype=torch.long,
        )

    def sample(self, batch: Batch, num_sampling_steps: int, temperature: float = 1.0) -> Batch:
        """
        Sample from the diffusion model by iteratively denoising.

        Args:
            batch: Initial batch with all intermediate vertices masked
            num_sampling_steps: Number of denoising steps
            temperature: Sampling temperature

        Returns:
            Denoised batch with predicted vertex values
        """
        # Clone batch to avoid modifying input
        x_batch = batch.clone()

        sampling_timesteps = self._get_sampling_timesteps(num_sampling_steps)
        relative_ts = self.scheduler[sampling_timesteps.to(self.scheduler.device)]
        relative_dts = get_timestep_step_sizes(relative_ts)

        for t_idx, (t, relative_t, relative_dt) in enumerate(zip(sampling_timesteps, relative_ts, relative_dts)):
            is_last_step = (t_idx == len(sampling_timesteps) - 1)

            # Forward pass to get predictions
            output = self.model(x_batch)
            logits = output["logits"]  # [total_vertices, vocab_size]

            # Sample from categorical distribution
            samples = torch.distributions.Categorical(
                logits=logits / temperature
            ).sample()  # [total_vertices]

            # Unmask threshold based on timestep
            unmask_threshold = relative_dt / relative_t

            # Process each graph
            for graph_idx in range(x_batch.num_graphs):
                v_mask = (x_batch.x_v_batch == graph_idx)
                num_vertices = v_mask.sum().item()
                graph_vertex_indices = torch.where(v_mask)[0]

                # Determine which vertices are currently masked
                was_masked = (x_batch.x_v[graph_vertex_indices, 0] == self.mask_token_id)

                # Decide which to unmask
                will_unmask = (
                    torch.rand(num_vertices, device=x_batch.x_v.device)
                    < unmask_threshold
                )
                will_unmask &= was_masked

                # Don't unmask first and last vertices (they should never be masked)
                will_unmask[0] = False
                will_unmask[-1] = False

                # Unmask vertices
                for i, vertex_idx in enumerate(graph_vertex_indices):
                    if will_unmask[i]:
                        x_batch.x_v[vertex_idx, 0] = samples[vertex_idx]

        return x_batch

    def validation_step(self, batch: Batch, batch_idx: int):
        """Validation step with full denoising"""
        # Create a noised version where all intermediate vertices are masked
        noised_batch = batch.clone()

        for graph_idx in range(batch.num_graphs):
            v_mask = (batch.x_v_batch == graph_idx)
            num_vertices = v_mask.sum().item()
            graph_vertex_indices = torch.where(v_mask)[0]

            # Mask all intermediate vertices
            for i, vertex_idx in enumerate(graph_vertex_indices):
                if i > 0 and i < num_vertices - 1:  # Not first or last
                    noised_batch.x_v[vertex_idx, 0] = self.mask_token_id

        # Denoise using sampling
        sampled_batch = self.sample(
            noised_batch,
            num_sampling_steps=self.num_timesteps,
            temperature=1.0
        )

        # Extract predictions and targets
        predictions = sampled_batch.x_v[:, 0].long()  # [total_vertices]
        targets = batch.targets  # [total_vertices]

        # Compute metrics per graph, ignoring first and last vertices
        batch_size = batch.num_graphs
        total_correct = 0
        total_tokens = 0
        exact_matches = 0

        for graph_idx in range(batch_size):
            v_mask = (batch.x_v_batch == graph_idx)
            num_vertices = v_mask.sum().item()

            graph_predictions = predictions[v_mask]
            graph_targets = targets[v_mask]

            # Only evaluate intermediate vertices (not first/last)
            if num_vertices > 2:
                pred = graph_predictions[1:-1]
                target = graph_targets[1:-1]

                correct = (pred == target)
                total_correct += correct.sum().item()
                total_tokens += len(pred)

                if correct.all():
                    exact_matches += 1

        # Compute metrics
        per_seq_acc = total_correct / total_tokens if total_tokens > 0 else 0.0
        exact_match = exact_matches / batch_size

        self.log('val_per_seq_acc', per_seq_acc, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.log('val_exact_match', exact_match, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return per_seq_acc


def get_timestep_step_sizes(timesteps: torch.Tensor) -> torch.Tensor:
    """Calculate step sizes between timesteps"""
    return -torch.diff(
        timesteps,
        append=torch.zeros([1], device=timesteps.device, dtype=timesteps.dtype),
    )

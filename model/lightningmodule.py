import torch
from typing import Dict, Any

from .base_lightningmodule import BasePathPredictionModule
from .metrics import NonGenerativeMetrics, GenerativeMetrics


class PathPredictionModule(BasePathPredictionModule):
    """Lightning module for autoregressive path prediction models"""

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
        loss: str = "cross_entropy",
    ):
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

        self.loss_name = loss

        # Initialize metrics
        self.path_metrics = NonGenerativeMetrics(self.graph, vocab_size, model_config.enabled_metrics)
        self.generative_metrics = None
        if hasattr(model_config, 'enabled_gen_metrics') and model_config.enabled_gen_metrics:
            self.generative_metrics = GenerativeMetrics(self.graph, vocab_size, model_config.enabled_gen_metrics)

    def forward(self, input_ids, target_ids):
        output = self.model(input_ids, target_ids)
        return output["logits"]

    def compute_metrics(self, logits, targets, input_ids):
        path_metrics = self.path_metrics.compute_metrics(logits, targets, input_ids)
        if self.generative_metrics:
            gen_metrics = self.generative_metrics.evaluate_generative(
                self.model, input_ids, num_samples=1, max_length=64, temperature=1.0
            )
            path_metrics.update(gen_metrics)
        return path_metrics

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        target_ids = batch['target_ids']

        output = self.model(input_ids, target_ids)
        loss = output["loss"]

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self._log_learning_rate()

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        target_ids = batch['target_ids']

        output = self.model(input_ids, target_ids)
        logits = output["logits"]
        loss = output["loss"]

        metrics = self.compute_metrics(logits, target_ids, input_ids)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        for metric_name, metric_value in metrics.items():
            self.log(f'val_{metric_name}', metric_value, on_epoch=True, prog_bar=True)

        return loss


class DiffusionPathPredictionModule(BasePathPredictionModule):
    """Lightning module for diffusion-based path prediction models"""

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
        loss: str = "cross_entropy",
    ):
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

        self.loss_name = loss
        self.mask_token_id = vocab_size - 1

        # Initialize metrics
        self.path_metrics = NonGenerativeMetrics(self.graph, vocab_size, model_config.enabled_metrics)
        self.generative_metrics = None
        if hasattr(model_config, 'enabled_gen_metrics') and model_config.enabled_gen_metrics:
            self.generative_metrics = GenerativeMetrics(self.graph, vocab_size, model_config.enabled_gen_metrics)

        # Diffusion scheduler
        self.num_timesteps = 20
        self.scheduler = torch.linspace(
            1 / self.num_timesteps, 1, steps=self.num_timesteps, dtype=torch.float32, device='cuda'
        )

    def forward(self, noised_x, input_ids):
        output = self.model(noised_x, input_ids)
        return output["logits"]

    def forward_noising(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Mask x (BL) depending on time step t (BL)."""
        mask_prob = self.scheduler[t].expand(-1, x.shape[1])
        will_mask = torch.bernoulli(mask_prob).to(dtype=torch.bool)

        should_noise = torch.ones_like(x, dtype=torch.bool, device=x.device)
        should_noise[:, 0] = False
        should_noise[:, -1] = False

        if should_noise is not None:
            will_mask &= should_noise

        noised_x = x.clone()
        noised_x[will_mask] = self.mask_token_id

        return noised_x

    def compute_metrics(self, logits, targets, input_ids):
        path_metrics = self.path_metrics.compute_metrics(logits, targets, input_ids)
        if self.generative_metrics:
            gen_metrics = self.generative_metrics.evaluate_generative(
                self.model, input_ids, num_samples=1, max_length=64, temperature=1.0
            )
            path_metrics.update(gen_metrics)
        return path_metrics

    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        target_ids = batch['target_ids']
        t = torch.randint(0, len(self.scheduler), [input_ids.size(0)], device=input_ids.device)

        noised_x = self.forward_noising(x=input_ids, t=t.unsqueeze(1))
        output = self.model(noised_x, input_ids)
        loss = output["loss"]

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self._log_learning_rate()

        return loss

    def _get_sampling_timesteps(self, num_sampling_steps, x):
        return torch.linspace(
            len(self.scheduler) - 1,
            len(self.scheduler) // num_sampling_steps,
            num_sampling_steps,
            device=x.device,
            dtype=torch.long,
        )

    def sample(
        self,
        num_sampling_steps: int,
        x: torch.Tensor,
        stochasticity: float = 0.0,
        temperature: float = 1.0,
    ):
        should_noise = torch.ones_like(x, dtype=torch.bool, device=x.device)
        should_noise[:, 0] = False
        should_noise[:, -1] = False

        sampling_timesteps = self._get_sampling_timesteps(num_sampling_steps, x)
        relative_ts = self.scheduler[sampling_timesteps]
        relative_dts = get_timestep_step_sizes(relative_ts)

        for t, relative_t, relative_dt in zip(sampling_timesteps, relative_ts, relative_dts):
            is_last_step = t == sampling_timesteps[-1]
            t = t.repeat(x.shape[0])

            logits = self(x, x.clone())
            samples = torch.distributions.Categorical(logits=logits / temperature).sample()

            unmask_threshold = relative_dt / relative_t
            if stochasticity != 0:
                unmask_threshold *= 1 + stochasticity * (1 - relative_t)

            was_masked = x == self.mask_token_id

            will_unmask = (
                torch.rand(x.shape[:2], device=unmask_threshold.device, dtype=unmask_threshold.dtype)
                < unmask_threshold
            )
            will_unmask &= was_masked

            if stochasticity != 0 and not is_last_step:
                remask_threshold = relative_dt * stochasticity
                will_remask = (
                    torch.rand(x.shape[:2], device=unmask_threshold.device, dtype=unmask_threshold.dtype)
                    < remask_threshold
                )
                will_remask &= ~was_masked
                if should_noise is not None:
                    will_remask &= should_noise
                x[will_remask] = self.mask_token_id

            x[will_unmask] = samples[will_unmask]

        return x

    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        target_ids = input_ids

        # Mask all tokens except first and last
        mask = torch.zeros_like(input_ids, dtype=torch.bool, device=input_ids.device)
        mask[:, 0] = True
        mask[:, -1] = True

        input_ids = input_ids.clone()
        input_ids[~mask] = self.mask_token_id
        sampled_input_ids = self.sample(
            num_sampling_steps=self.num_timesteps, x=input_ids, stochasticity=0, temperature=1.0
        )

        # Compute metrics ignoring first and last token
        pred = sampled_input_ids[:, 1:-1]
        target = target_ids[:, 1:-1]

        correct = (pred == target)
        per_seq_acc = correct.float().mean(dim=1)
        exact_match = correct.all(dim=1).float()

        self.log('val_per_seq_acc', per_seq_acc.mean(), on_epoch=True, prog_bar=True)
        self.log('val_exact_match', exact_match.mean(), on_epoch=True, prog_bar=True)

        return per_seq_acc.mean()


def get_timestep_step_sizes(timesteps: torch.Tensor) -> torch.Tensor:
    return -torch.diff(
        timesteps,
        append=torch.zeros([1], device=timesteps.device, dtype=timesteps.dtype),
    )

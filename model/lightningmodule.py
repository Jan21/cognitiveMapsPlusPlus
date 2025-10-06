import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
try:
    from muon import MuonWithAuxAdam
    MUON_AVAILABLE = True
except ImportError:
    MUON_AVAILABLE = False
from torch.optim.lr_scheduler import LinearLR
from typing import Dict, Any
import wandb
import pickle
import networkx as nx

from .architecture.model import TransformerModel
from .architecture.diffusion_model import DiffusionModel
from .architecture.past import PAST, PASTConfig

from .metrics import NonGenerativeMetrics, GenerativeMetrics
from .model_factory import create_model


class PathPredictionModule(pl.LightningModule):
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
        super().__init__()
        self.save_hyperparameters()
        
        self.model = create_model(model_config, vocab_size)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.optimizer_name = optimizer
        self.loss_name = loss
        
        graph_path = graph_path.replace('.pkl', f'_{graph_type}.pkl')
        # Load the graph for path validation
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        
        # Initialize metrics with configurable enabled metrics
        self.path_metrics = NonGenerativeMetrics(self.graph, vocab_size, model_config.enabled_metrics)
        self.generative_metrics = None
        if hasattr(model_config, 'enabled_gen_metrics') and model_config.enabled_gen_metrics:
            self.generative_metrics = GenerativeMetrics(self.graph, vocab_size, model_config.enabled_gen_metrics)
    
    def forward(self, input_ids, target_ids):
        output = self.model(input_ids, target_ids)
        return output["logits"]
    
    def compute_metrics(self, logits, targets, input_ids):
        # if current_epoch >= 20 and current_epoch % 10 == 0 and batch_idx % 10 == 0:
        #     gen_metrics = self.generative_metrics.evaluate_generative(self, batch, num_samples=1, max_length=64, temperature=1.0)
        #     self.log('val_gen_path_validity', gen_metrics['gen_path_validity'], on_epoch=True, prog_bar=False)
        #     self.log('val_gen_goal_accuracy', gen_metrics['gen_goal_accuracy'], on_epoch=True, prog_bar=False)
        #     self.log('val_gen_avg_path_length_diff', gen_metrics['gen_avg_path_length_diff'], on_epoch=True, prog_bar=False)
        path_metrics = self.path_metrics.compute_metrics(logits, targets, input_ids)
        if self.generative_metrics:
            gen_metrics = self.generative_metrics.evaluate_generative(self.model, input_ids, num_samples=1, max_length=64, temperature=1.0)
            path_metrics.update(gen_metrics)
        return path_metrics
    
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        target_ids = batch['target_ids']
        
        output = self.model(input_ids, target_ids)
        loss = output["loss"]
        
        # Log learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('learning_rate', current_lr, on_step=True, on_epoch=True, prog_bar=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        target_ids = batch['target_ids']
        #self.model.eval()
        # Use all-but-last mode for validation
        output = self.model(input_ids, target_ids)
        logits = output["logits"]
        loss = output["loss"]
        
        metrics = self.compute_metrics(logits, target_ids, input_ids)
        
        # Run generative evaluation only every 10 epochs and only after epoch 20
        current_epoch = self.current_epoch

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        
        # Log only the metrics that were computed
        for metric_name, metric_value in metrics.items():
            self.log(f'val_{metric_name}', metric_value, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        if self.optimizer_name.lower() == "muon":
            if not MUON_AVAILABLE:
                print("Warning: Muon optimizer not available. Falling back to AdamW.")
                optimizer = AdamW(
                    self.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay
                )
            else:
                try:
                    # Initialize distributed if not already done (required for Muon)
                    import torch.distributed as dist
                    if not dist.is_initialized():
                        import os
                        os.environ['MASTER_ADDR'] = 'localhost'
                        os.environ['MASTER_PORT'] = '12355'
                        os.environ['RANK'] = '0'
                        os.environ['WORLD_SIZE'] = '1'
                        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo', 
                                              rank=0, world_size=1)
                    
                    hidden_weights = [p for p in self.model.transformer.parameters() if p.ndim >= 2]
                    hidden_gains_biases = [p for p in self.model.transformer.parameters() if p.ndim < 2]
                    nonhidden_params = [*self.model.output_projection.parameters(), *self.model.embedding.parameters(), *self.model.pos_encoding.parameters()]
                    param_groups = [
                        dict(params=hidden_weights, use_muon=True,
                            lr=0.02, weight_decay=0.01),
                        dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
                            lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
                    ]
                    optimizer = MuonWithAuxAdam(param_groups)
                except Exception as e:
                    print(f"Warning: Failed to initialize Muon optimizer ({e}). Falling back to AdamW.")
                    optimizer = AdamW(
                        self.parameters(),
                        lr=self.learning_rate,
                        weight_decay=self.weight_decay
                    )
        else:
            # Default to AdamW
            optimizer = AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        
        # Simple warmup + cosine annealing scheduler - good default for language modeling
        def lr_lambda(step):
            if step < self.warmup_steps:
                # Linear warmup
                return step / self.warmup_steps
            else:
                # Cosine annealing after warmup with minimum lr of 0.1
                progress = (step - self.warmup_steps) / max(1, self.trainer.estimated_stepping_batches - self.warmup_steps)
                cosine_factor = 0.5 * (1 + torch.cos(torch.tensor(torch.pi * min(progress, 1.0))))
                return 0.1 + 0.9 * cosine_factor
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
    




class DiffusionPathPredictionModule(pl.LightningModule):
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
        super().__init__()
        self.save_hyperparameters()
        
        self.model = create_model(model_config, vocab_size)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.optimizer_name = optimizer
        self.loss_name = loss
        self.mask_token_id = vocab_size - 1

        
        graph_path = graph_path.replace('.pkl', f'_{graph_type}.pkl')
        # Load the graph for path validation
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        
        # Initialize metrics with configurable enabled metrics
        self.path_metrics = NonGenerativeMetrics(self.graph, vocab_size, model_config.enabled_metrics)
        self.generative_metrics = None
        if hasattr(model_config, 'enabled_gen_metrics') and model_config.enabled_gen_metrics:
            self.generative_metrics = GenerativeMetrics(self.graph, vocab_size, model_config.enabled_gen_metrics)

        self.num_timesteps = 20
        self.scheduler = torch.linspace(
            1 / self.num_timesteps, 1, steps=self.num_timesteps, dtype=torch.float32
        ,device='cuda')  # Probability path scheduler
    
    def forward(self, noised_x, input_ids):
        output = self.model(noised_x, input_ids)
        return output["logits"]
    


    def forward_noising(
        self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Mask x (BL) depending on time step t (BL)."""

        # t is the masking probability. t=0%: dont mask anything, t=100%: mask everything
        mask_prob = self.scheduler[t].expand(-1, x.shape[1])
        will_mask = torch.bernoulli(mask_prob).to(dtype=torch.bool)

        should_noise = torch.ones_like(x, dtype=torch.bool, device=x.device)
        should_noise[:, 0] = False
        should_noise[:, -1] = False
        # Don't mask tokens that should not be noised
        if should_noise is not None:
            will_mask &= should_noise

        noised_x = x.clone()
        noised_x[will_mask] = self.mask_token_id

        return noised_x


    def compute_metrics(self, logits, targets, input_ids):
        # if current_epoch >= 20 and current_epoch % 10 == 0 and batch_idx % 10 == 0:
        #     gen_metrics = self.generative_metrics.evaluate_generative(self, batch, num_samples=1, max_length=64, temperature=1.0)
        #     self.log('val_gen_path_validity', gen_metrics['gen_path_validity'], on_epoch=True, prog_bar=False)
        #     self.log('val_gen_goal_accuracy', gen_metrics['gen_goal_accuracy'], on_epoch=True, prog_bar=False)
        #     self.log('val_gen_avg_path_length_diff', gen_metrics['gen_avg_path_length_diff'], on_epoch=True, prog_bar=False)
        path_metrics = self.path_metrics.compute_metrics(logits, targets, input_ids)
        if self.generative_metrics:
            gen_metrics = self.generative_metrics.evaluate_generative(self.model, input_ids, num_samples=1, max_length=64, temperature=1.0)
            path_metrics.update(gen_metrics)
        return path_metrics
    
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        target_ids = batch['target_ids']
        t = torch.randint(0, len(self.scheduler), [input_ids.size(0)], device=input_ids.device)

        # noised_x: B L
        noised_x = self.forward_noising(
            x=input_ids, t=t.unsqueeze(1)
        )
        output = self.model(noised_x, input_ids)
        loss = output["loss"]
        
        # Log learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('learning_rate', current_lr, on_step=True, on_epoch=True, prog_bar=False)
        
        return loss
    
    def _get_sampling_timesteps(self, num_sampling_steps,x):
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
        x: torch.Tensor | None = None,
        stochasticity: float = 0.0,
        temperature: float = 1.0,
    ):

        # Create the integer timesteps and step sizes for the given num_sampling_steps
        # S
        should_noise = torch.ones_like(x, dtype=torch.bool, device=x.device)
        should_noise[:, 0] = False
        should_noise[:, -1] = False
        sampling_timesteps = self._get_sampling_timesteps(num_sampling_steps,x)
        relative_ts = self.scheduler[sampling_timesteps]
        relative_dts = get_timestep_step_sizes(relative_ts)

        for t, relative_t, relative_dt in zip(
            sampling_timesteps, relative_ts, relative_dts
        ):
            is_last_step = t == sampling_timesteps[-1]
            t = t.repeat(x.shape[0])
            assert t.shape == x.shape[:1], t.shape

            # B L V
            logits = self(x, x.clone())

            # B L
            samples = torch.distributions.Categorical(
                logits=logits / temperature
            ).sample()

            # B L
            # Chance to unmask proportional to
            # - step size: higher step size means higher chance
            # - timestep: lower timestep means higher chance (so in the end the chance is 100%)
            unmask_threshold = relative_dt / relative_t

            # With remasking, the unmasking probability is changed
            if stochasticity != 0:
                unmask_threshold *= 1 + stochasticity * (1 - relative_t)

            was_masked = x == self.mask_token_id

            # Unmask
            will_unmask = (
                torch.rand(
                    x.shape[:2],
                    device=unmask_threshold.device,
                    dtype=unmask_threshold.dtype,
                )
                < unmask_threshold
            )
            # Only unmask the tokens that were masked
            will_unmask &= was_masked

            # Remask when stochasticity is non-zero
            if stochasticity != 0 and not is_last_step:
                remask_threshold = relative_dt * stochasticity
                will_remask = (
                    torch.rand(
                        x.shape[:2],
                        device=unmask_threshold.device,
                        dtype=unmask_threshold.dtype,
                    )
                    < remask_threshold
                )
                # Only remask the tokens that were unmasked
                will_remask &= ~was_masked

                # Only remask tokens that aren't constant
                if should_noise is not None:
                    will_remask &= should_noise

                x[will_remask] = self.mask_token_id

            # B L
            x[will_unmask] = samples[will_unmask]


        return x


    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        target_ids = input_ids
        # Mask all tokens except the first and the last one
        # Create a mask: True for tokens to keep (first and last), False for others
        mask = torch.zeros_like(input_ids, dtype=torch.bool, device=input_ids.device)
        mask[:, 0] = True
        mask[:, -1] = True
        # Set all tokens except first and last to mask_token_id
        input_ids = input_ids.clone()
        input_ids[~mask] = self.mask_token_id
        sampled_input_ids = self.sample(num_sampling_steps=self.num_timesteps, x=input_ids, stochasticity=0, temperature=1.0)
        #self.model.eval()
        # Use all-but-last mode for validation
        # Compute per-sequence accuracy, exact match accuracy, and loss, ignoring first and last token

        # Ignore first and last token for both predictions and targets
        pred = sampled_input_ids[:, 1:-1]
        target = target_ids[:, 1:-1]

        # Per-token accuracy (mean over all tokens, per sequence)
        correct = (pred == target)
        per_seq_acc = correct.float().mean(dim=1)  # shape: (batch,)

        # Exact match accuracy (all tokens correct in a sequence)
        exact_match = correct.all(dim=1).float()  # shape: (batch,)

        # Compute loss (cross-entropy), flatten batch and sequence
        # Reshape to (batch * seq,)

        # Log metrics
        self.log('val_per_seq_acc', per_seq_acc.mean(), on_epoch=True, prog_bar=True)
        self.log('val_exact_match', exact_match.mean(), on_epoch=True, prog_bar=True)


        
        return per_seq_acc.mean()
    
    def configure_optimizers(self):
        if self.optimizer_name.lower() == "muon":
            if not MUON_AVAILABLE:
                print("Warning: Muon optimizer not available. Falling back to AdamW.")
                optimizer = AdamW(
                    self.parameters(),
                    lr=self.learning_rate,
                    weight_decay=self.weight_decay
                )
            else:
                try:
                    # Initialize distributed if not already done (required for Muon)
                    import torch.distributed as dist
                    if not dist.is_initialized():
                        import os
                        os.environ['MASTER_ADDR'] = 'localhost'
                        os.environ['MASTER_PORT'] = '12355'
                        os.environ['RANK'] = '0'
                        os.environ['WORLD_SIZE'] = '1'
                        dist.init_process_group(backend='nccl' if torch.cuda.is_available() else 'gloo', 
                                              rank=0, world_size=1)
                    
                    hidden_weights = [p for p in self.model.transformer.parameters() if p.ndim >= 2]
                    hidden_gains_biases = [p for p in self.model.transformer.parameters() if p.ndim < 2]
                    nonhidden_params = [*self.model.output_projection.parameters(), *self.model.embedding.parameters(), *self.model.pos_encoding.parameters()]
                    param_groups = [
                        dict(params=hidden_weights, use_muon=True,
                            lr=0.02, weight_decay=0.01),
                        dict(params=hidden_gains_biases+nonhidden_params, use_muon=False,
                            lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01),
                    ]
                    optimizer = MuonWithAuxAdam(param_groups)
                except Exception as e:
                    print(f"Warning: Failed to initialize Muon optimizer ({e}). Falling back to AdamW.")
                    optimizer = AdamW(
                        self.parameters(),
                        lr=self.learning_rate,
                        weight_decay=self.weight_decay
                    )
        else:
            # Default to AdamW
            optimizer = AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        
        # Simple warmup + cosine annealing scheduler - good default for language modeling
        def lr_lambda(step):
            if step < self.warmup_steps:
                # Linear warmup
                return step / self.warmup_steps
            else:
                # Cosine annealing after warmup with minimum lr of 0.1
                progress = (step - self.warmup_steps) / max(1, self.trainer.estimated_stepping_batches - self.warmup_steps)
                cosine_factor = 0.5 * (1 + torch.cos(torch.tensor(torch.pi * min(progress, 1.0))))
                return 0.1 + 0.9 * cosine_factor
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
    


def get_timestep_step_sizes(timesteps: torch.Tensor) -> torch.Tensor:
    return -torch.diff(
        timesteps,
        append=torch.zeros([1], device=timesteps.device, dtype=timesteps.dtype),
    )
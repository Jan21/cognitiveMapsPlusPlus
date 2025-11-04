import torch
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Dict, Any, Optional
import pytorch_lightning as pl
from hydra.utils import instantiate
from omegaconf import DictConfig
from model.architecture.transformer import TransformerModel
from model.architecture.diffusion_upsample import Diffusion_ResidualUpsample
from utils.metrics import NonGenerativeMetrics, GenerativeMetrics

def create_model(model_config: DictConfig, vocab_size: int):
    """Factory function to create models based on configuration."""
    
    # Set vocab_size if not already set
    if model_config.vocab_size is None:
        model_config.vocab_size = vocab_size
    
    
    # For other models, use hydra's instantiate
    return instantiate(model_config)



class PathPredictionModule(pl.LightningModule):
    """Lightning module for path prediction models"""

    def __init__(
        self,
        model_config: Dict[str, Any],
        vocab_size: int,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000,
        optimizer: str = "adamw",
        graph_type: str = "sphere",
        graph = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Model configuration
        self.model = create_model(model_config, vocab_size)
        #self.model = torch.compile(self.model, mode='reduce-overhead')
        self.vocab_size = vocab_size

        # Optimizer configuration
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.optimizer_name = optimizer

        self.graph = graph

        # Initialize metrics
        self.path_metrics = NonGenerativeMetrics(self.graph, vocab_size, model_config.enabled_metrics)
        self.generative_metrics = None
        if hasattr(model_config, 'enabled_gen_metrics') and model_config.enabled_gen_metrics:
            self.generative_metrics = GenerativeMetrics(self.graph, vocab_size, model_config.enabled_gen_metrics)

    def forward(self, input_ids, target_ids):
        output = self.model(input_ids, target_ids)
        return output["logits"]

    def compute_metrics(self, logits, batch):
        path_metrics = self.path_metrics.compute_metrics(logits, batch)
        if self.generative_metrics:
            gen_metrics = self.generative_metrics.evaluate_generative(
                self.model, batch['input_ids'], num_samples=1, max_length=64, temperature=1.0
            )
            path_metrics.update(gen_metrics)
        return path_metrics

    def training_step(self, batch, batch_idx):


        output = self.model(batch)
        output = self.model.get_loss(output, batch)
        loss = output["loss"]

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self._log_learning_rate()

        return loss

    def validation_step(self, batch, batch_idx):

        output = self.model(batch)
        output = self.model.get_loss(output, batch)
        logits = output["logits"]
        loss = output["loss"]

        metrics = self.compute_metrics(logits, batch)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        if "mse_sum_norm_diffs" in output:
            self.log('val_mse_sum_norm_diffs', output["mse_sum_norm_diffs"], on_epoch=True, prog_bar=True)

        for metric_name, (metric_value, batch_size) in metrics.items():
            self.log(f'val_{metric_name}', metric_value, on_epoch=True, prog_bar=True, batch_size=batch_size)
        if len(metrics) == 0:
            self.log('val_accuracy', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler"""

        # Create parameter groups with different learning rates
        param_groups = self.model.get_param_groups()

        # Default to AdamW with parameter groups
        optimizer = AdamW(param_groups)

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


 


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
from .metrics import NonGenerativeMetrics, GenerativeMetrics

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
        loss = output["loss"]

        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self._log_learning_rate()

        return loss

    def validation_step(self, batch, batch_idx):

        output = self.model(batch)
        logits = output["logits"]
        loss = output["loss"]

        metrics = self.compute_metrics(logits, batch)

        self.log('val_loss', loss, on_epoch=True, prog_bar=True)

        for metric_name, (metric_value, batch_size) in metrics.items():
            self.log(f'val_{metric_name}', metric_value, on_epoch=True, prog_bar=True, batch_size=batch_size)

        return loss

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




    # def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
    #     """
    #     Validation step for middle-node prediction.

    #     Args:
    #         batch: Dictionary with 'start_id', 'end_id', 'middle_id'
    #     """


    #     # Forward pass
    #     output = self.model(batch)
    #     logits = output["logits"]
    #     loss = output["loss"]

    #     # Get predictions (argmax)
    #     predictions = logits.argmax(dim=-2)
    #     labels = batch[4]
    #     # Compute accuracy
    #     correct = (predictions == labels).sum().item()
    #     # INSERT_YOUR_CODE
    #     total = predictions.numel()
    #     accuracy = correct / total
    #     predictions[:,0] = batch['input'][:,0]
    #     predictions[:,-1] = batch['input'][:,-1]
    #     # INSERT_YOUR_CODE
    #     # Compute exact match accuracy: how many sequences match all labels exactly
    #     # predictions: [batch_size, seq_len], labels: [batch_size, seq_len]
    #     exact_matches = (predictions[:, 1:-1] == labels[:, 1:-1]).all(dim=-1).float()
    #     exact_match_accuracy = exact_matches.mean().item()



    #     # Compute distance to actual middle point
    #     batch_size = predictions.size(0)
    #     paths,validity = self._construct_and_validate_paths(predictions)
    #     lengths_gt = batch['len']
    #     total_diff = 0
    #     valid_count = 0.00000001
    #     for i in range(batch_size):
    #         if validity[i]:
    #             valid_count += 1
    #             path_length = len(paths[i])
    #             diff = abs(path_length - lengths_gt[i])
    #             total_diff += diff
    #     avg_diff = total_diff / valid_count
    #     validity_accuracy = sum(validity) / batch_size
    #     self.log('val_avg_diff', avg_diff, on_epoch=True, prog_bar=True, batch_size=valid_count)
    #     # Log metrics
    #     self.log('val_loss', loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
    #     self.log('val_accuracy', accuracy, on_epoch=True, prog_bar=True, batch_size=batch_size)
    #     self.log('val_exact_match', exact_match_accuracy, on_epoch=True, prog_bar=True, batch_size=batch_size)
    #     self.log('val_validity', validity_accuracy, on_epoch=True, prog_bar=True, batch_size=batch_size)

    #     return accuracy


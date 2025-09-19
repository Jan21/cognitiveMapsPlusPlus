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

from .model import TransformerModel
from .metrics import NonGenerativeMetrics, GenerativeMetrics


class PathPredictionModule(pl.LightningModule):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 128,
        dropout: float = 0.1,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        warmup_steps: int = 1000,
        optimizer: str = "adamw",  # "adamw" or "muon"
        graph_type: str = "sphere",
        graph_path: str = "temp/sphere_graph.pkl",
        loss: str = "cross_entropy"

    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = TransformerModel(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            max_seq_length=max_seq_length,
            dropout=dropout
        )
        
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.optimizer_name = optimizer
        self.loss_name = loss
        
        graph_path = graph_path.replace('.pkl', f'_{graph_type}.pkl')
        # Load the graph for path validation
        with open(graph_path, 'rb') as f:
            self.graph = pickle.load(f)
        
        # Initialize metrics
        self.path_metrics = NonGenerativeMetrics(self.graph, vocab_size)
        self.generative_metrics = GenerativeMetrics(self.graph, vocab_size)
    
    def forward(self, input_ids):
        return self.model(input_ids)
    
    def compute_loss(self, logits, targets, use_prob_weighting=False):
        # logits: (batch_size, seq_len, vocab_size)
        # targets: (batch_size, seq_len) with -100 for padding tokens
        
        # Reshape for loss computation
        shift_logits = logits.view(-1, logits.size(-1))
        shift_labels = targets.view(-1)
        shift_labels_copy = shift_labels.clone()
        shift_labels_copy[shift_labels_copy == -100] = 0
        
        if self.loss_name == "prob_weighting":
            # Compute standard cross entropy loss
            loss = F.cross_entropy(
                shift_logits,
                shift_labels,
                ignore_index=-100,
                reduction='none'
            )
            
            # Apply probability coefficient weighting
            probs = torch.softmax(shift_logits, dim=-1)
            prob_coefficients = probs.gather(1, shift_labels_copy.unsqueeze(-1)).squeeze(-1)
            loss = loss * prob_coefficients.detach()
            
            # Only average over non-ignored tokens
            mask = (shift_labels != -100).float()
            loss = (loss * mask).sum() / mask.sum()
        elif self.loss_name == "cross_entropy":
            # Standard cross entropy loss
            loss = F.cross_entropy(
                shift_logits,
                shift_labels,
                ignore_index=-100
            )
        else:
            raise ValueError(f"Invalid loss function: {self.loss_name}")
        
        return loss
    
    def compute_metrics(self, logits, targets):
        return self.path_metrics.compute_metrics(logits, targets)
    
    
    def training_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        target_ids = batch['target_ids']
        
        logits = self(input_ids)
        loss = self.compute_loss(logits, target_ids)
        
        # Log learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('learning_rate', current_lr, on_step=True, on_epoch=True, prog_bar=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch['input_ids']
        target_ids = batch['target_ids']
        
        logits = self(input_ids)
        loss = self.compute_loss(logits, target_ids)
        
        metrics = self.compute_metrics(logits, target_ids)
        
        # Run generative evaluation only every 10 epochs and only after epoch 20
        current_epoch = self.current_epoch
        if current_epoch >= 20 and current_epoch % 10 == 0 and batch_idx % 10 == 0:
            gen_metrics = self.generative_metrics.evaluate_generative(self, batch, num_samples=1, max_length=64, temperature=1.0)
            self.log('val_gen_path_validity', gen_metrics['gen_path_validity'], on_epoch=True, prog_bar=False)
            self.log('val_gen_goal_accuracy', gen_metrics['gen_goal_accuracy'], on_epoch=True, prog_bar=False)
            self.log('val_gen_avg_path_length_diff', gen_metrics['gen_avg_path_length_diff'], on_epoch=True, prog_bar=False)
        
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        self.log('val_accuracy', metrics['accuracy'], on_epoch=True, prog_bar=True)
        self.log('val_path_validity', metrics['path_validity'], on_epoch=True, prog_bar=True)
        self.log('val_edge_accuracy', metrics['edge_accuracy'], on_epoch=True, prog_bar=True)
        self.log('val_exact_match_accuracy', metrics['exact_match_accuracy'], on_epoch=True, prog_bar=True)
        
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
import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
import os
from datetime import datetime
from data.datamodule import PathDataModule
from data.curriculum import curriculum_training
from model.lightningmodule import PathPredictionModule, DiffusionPathPredictionModule
from hydra.core.hydra_config import HydraConfig
import wandb

# Conditionally import GNN module
try:
    from model.gnn_lightningmodule import GNNPathPredictionModule
    GNN_AVAILABLE = True
except ImportError:
    GNN_AVAILABLE = False
    GNNPathPredictionModule = None

# Import RNN module
try:
    from model.rnn_lightningmodule import RNNMiddleNodeModule
    RNN_AVAILABLE = True
except ImportError:
    RNN_AVAILABLE = False
    RNNMiddleNodeModule = None


class SimplePruningCallback(Callback):
    def __init__(self, patience=3, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.wait = 0
        self.best_loss = float('inf')
        
    def on_validation_epoch_end(self, trainer, pl_module):
        current_loss = trainer.logged_metrics.get('val_loss', float('inf'))
        
        # Only start pruning after a few epochs
        if trainer.current_epoch < 2:
            return
            
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.wait = 0
        else:
            self.wait += 1
            
        # If loss hasn't improved for patience epochs and it's very high, stop
        if self.wait >= self.patience and current_loss > 2.0:  # Adjust threshold as needed
            trainer.should_stop = True



@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Compute vocab_size dynamically based on graph type
    if cfg.model.vocab_size is None:
        if cfg.graph_generation.type == "sphere":
            cfg.model.vocab_size = cfg.graph_generation.sphere_mesh.num_horizontal * cfg.graph_generation.sphere_mesh.num_vertical + 2
        elif cfg.graph_generation.type == "grid":
            cfg.model.vocab_size = cfg.graph_generation.grid_2d.width * cfg.graph_generation.grid_2d.height + 2
        else:
            raise ValueError(f"Unknown graph type: {cfg.graph_generation.type}. Expected 'sphere' or 'grid'.")
    
    if cfg.curriculum_learning.enabled:
        return curriculum_training(cfg)
    else:
        return standard_training(cfg)


def standard_training(cfg: DictConfig) -> float:
    # Check which model type is requested
    use_gnn = cfg.model.get('use_gnn', False)
    use_rnn = cfg.model.get('use_rnn', False)

    # Set up data module
    datamodule = PathDataModule(
        train_file=cfg.data.train_file,
        test_file=cfg.data.test_file,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        max_path_length=cfg.data.max_path_length,
        vocab_size=cfg.model.vocab_size,
        data_dir=cfg.paths.data_dir,
        graph_type=cfg.graph_generation.type,
        use_gnn=use_gnn,
        use_rnn=use_rnn,
    )

    # Set up model based on architecture type
    if use_rnn:
        if not RNN_AVAILABLE:
            raise ImportError("RNN model requested but module is not available.")
        model = RNNMiddleNodeModule(
            model_config=cfg.model,
            vocab_size=cfg.model.vocab_size,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            warmup_steps=cfg.training.warmup_steps,
            optimizer=cfg.training.optimizer,
            graph_type=cfg.graph_generation.type,
            graph_path=cfg.graph_generation.output.file_path,
        )
    elif use_gnn:
        if not GNN_AVAILABLE:
            raise ImportError(
                "GNN model requested but PyTorch Geometric is not available. "
                "Install it with: pip install torch-geometric"
            )
        model = GNNPathPredictionModule(
            model_config=cfg.model,
            vocab_size=cfg.model.vocab_size,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            warmup_steps=cfg.training.warmup_steps,
            optimizer=cfg.training.optimizer,
            graph_type=cfg.graph_generation.type,
            graph_path=cfg.graph_generation.output.file_path,
        )
    else:
        model = DiffusionPathPredictionModule(
            model_config=cfg.model,
            vocab_size=cfg.model.vocab_size,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            warmup_steps=cfg.training.warmup_steps,
            optimizer=cfg.training.optimizer,
            graph_type=cfg.graph_generation.type,
            graph_path=cfg.graph_generation.output.file_path,
            loss=cfg.training.loss
        )
    
    # Set up logger
    hydra_cfg = HydraConfig.get()
    job_id = hydra_cfg.job.get('num', 0) if hydra_cfg.job.get('num') is not None else 0
    

    hyperparams = f"dlr{cfg.training.learning_rate:.1e}"
    experiment_name = f"{cfg.logging.experiment_name}_{hyperparams}_trial{job_id}"

    
    logger = WandbLogger(
        project=cfg.logging.project_name,
        name=experiment_name,
        config=dict(cfg),
        log_model=False,
        group=None
    )
    
    # Set up callbacks
    callbacks = []
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.paths.checkpoint_dir,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_exact_match',
        mode='max'
    )
    callbacks.append(checkpoint_callback)
    
    early_stopping = EarlyStopping(
        monitor='val_exact_match',
        patience=500,
        mode='max'
    )
    callbacks.append(early_stopping)
    
    # Add simple pruning callback for multirun
    if hydra_cfg.mode == hydra_cfg.mode.MULTIRUN:
        pruning_callback = SimplePruningCallback(patience=7)
        callbacks.append(pruning_callback)
    
    # Set up trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        logger=logger,
        callbacks=callbacks,
        gradient_clip_val=cfg.training.gradient_clip_val,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        enable_checkpointing=True,
        enable_progress_bar=True
    )
    
    # Train the model
    trainer.fit(model, datamodule)
    wandb.finish()
    # Return the validation loss for Optuna optimization
    return trainer.callback_metrics.get("val_loss", float("inf"))




if __name__ == "__main__":
    main()
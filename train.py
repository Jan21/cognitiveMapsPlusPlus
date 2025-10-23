import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, Callback
import os
from datetime import datetime
from data.datamodule import PathDataModule
from model import lightning_module_class
from hydra.core.hydra_config import HydraConfig
import wandb
import pickle



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

    # Construct the graph file path based on graph type
    graph_type = cfg.graph_generation.type
    graph_path = cfg.data_generation.output_dir + f"/graph_{graph_type}.pkl"
    # Load the graph and count vertices
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)

    num_vertices = graph.number_of_nodes()
    cfg.model.vocab_size = num_vertices + 10 # 10 special tokens
    
    return standard_training(cfg, graph)


def standard_training(cfg: DictConfig, graph) -> float:

    # Set up data module
    datamodule = PathDataModule(
        train_file=cfg.data.train_file,
        test_file=cfg.data.test_file,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        max_path_length=cfg.data.max_path_length,
        vocab_size=cfg.model.vocab_size,
        graph_type=cfg.graph_generation.type,
        data_dir=cfg.paths.data_dir,
        dataset_type=cfg.model.dataset_type,
    )

    # Select Lightning module based on configuration
    lightning_module_type = cfg.model.get('lightning_module', 'path')
    lightning_module = lightning_module_class[lightning_module_type]

    model = lightning_module(
            model_config=cfg.model,
            vocab_size=cfg.model.vocab_size,
            learning_rate=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
            warmup_steps=cfg.training.warmup_steps,
            optimizer=cfg.training.optimizer,
            graph_type=cfg.graph_generation.type,
            graph=graph
        )
    
    # Set up logger
    hydra_cfg = HydraConfig.get()
    job_id = hydra_cfg.job.get('num', 0) if hydra_cfg.job.get('num') is not None else 0
    

    hyperparams = f"graph_{cfg.graph_generation.type}"
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
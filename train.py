import warnings
warnings.filterwarnings('ignore', message="'has_cuda' is deprecated")
warnings.filterwarnings('ignore', message="'has_cudnn' is deprecated")
warnings.filterwarnings('ignore', message="'has_mps' is deprecated")
warnings.filterwarnings('ignore', message="'has_mkldnn' is deprecated")


import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import os
from datetime import datetime
from data.datamodule import PathDataModule
from model import lightning_module_class
from hydra.core.hydra_config import HydraConfig
from utils.callbacks import setup_callbacks
import wandb
import pickle


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    # Construct the graph file path based on graph type
    graph_type = cfg.graph_generation.type
    graph_path = cfg.data_generation.output_dir + f"/graph_{graph_type}.pkl"
    # Load the graph and count vertices
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)

    num_vertices = graph.number_of_nodes()
    cfg.training.max_steps = int((num_vertices ** 2)*0.8 // cfg.training.batch_size * 20)
    cfg.model.vocab_size = num_vertices + 1 # 10 special tokens
    
    # Set up data module
    datamodule = PathDataModule(
        train_file=cfg.data.train_file,
        test_file=cfg.data.test_file,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        max_path_length_train=cfg.data.max_path_length_train,
        max_path_length_val=cfg.data.max_path_length_val,
        vocab_size=cfg.model.vocab_size,
        graph_type=cfg.graph_generation.type,
        data_dir=cfg.paths.data_dir,
        dataset_type=cfg.model.dataset_type,
        percentage_of_train_samples=cfg.data.get('percentage_of_train_samples', 1.0),
        num_val_samples=cfg.data.get('num_val_samples', None),
        graph=graph,
        use_array=cfg.model.get('use_array', False),
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
            graph=graph,
            save_embeddings=cfg.training.get('save_embeddings', True),
            embedding_save_interval=cfg.training.get('embedding_save_interval', 10),
            embedding_save_dir=cfg.training.get('embedding_save_dir', 'temp/embs'),
            visualize_embeddings=cfg.training.get('visualize_embeddings', True),
        )
    
    # Set up logger
    hydra_cfg = HydraConfig.get()
    job_id = hydra_cfg.job.get('num', 0) if hydra_cfg.job.get('num') is not None else 0
    

    hyperparams = f"graph_{cfg.graph_generation.type}_model_{cfg.model._target_}"
    experiment_name = f"{cfg.logging.experiment_name}_{hyperparams}_trial{job_id}"

    
    logger = WandbLogger(
        project=cfg.logging.project_name,
        name=experiment_name,
        config=dict(cfg),
        log_model=False,
        group=None
    )
    
    # Set up callbacks
    callbacks = setup_callbacks(cfg, experiment_name, hydra_cfg)

    # Set up trainer
    # Use max_steps if specified, otherwise use max_epochs
    trainer_kwargs = {
        'logger': logger,
        'callbacks': callbacks,
        'gradient_clip_val': cfg.training.gradient_clip_val,
        'log_every_n_steps': cfg.logging.log_every_n_steps,
        'enable_checkpointing': True,
        'enable_progress_bar': True,
        'precision': '16-mixed',
    }

    # Add max_steps or max_epochs based on config
    if cfg.training.get('max_epochs', -1) > 0:
        trainer_kwargs['max_epochs'] = cfg.training.max_epochs
    else:
        print(f"Using max_steps: {cfg.training.max_steps}")
        trainer_kwargs['max_steps'] = cfg.training.max_steps

    trainer = pl.Trainer(**trainer_kwargs)
    
    # Train the model
    trainer.fit(model, datamodule)
    wandb.finish()
    # Return the validation loss for Optuna optimization
    return trainer.callback_metrics.get("val_loss", float("inf"))

if __name__ == "__main__":
    main()
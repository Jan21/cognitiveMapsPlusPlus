import torch
from typing import List
import os
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from model.lightningmodule import PathPredictionModule, DiffusionPathPredictionModule
from data.datamodule import PathDataModule


class CurriculumAccuracyCallback(Callback):
    def __init__(self, target_accuracy: float, patience: int = 10):
        self.target_accuracy = target_accuracy
        self.patience = patience
        self.wait = 0
        self.best_accuracy = 0.0
        
    def on_validation_epoch_end(self, trainer, pl_module):
        current_accuracy = trainer.logged_metrics.get('val_per_seq_acc', 0.0)
        
        if current_accuracy > self.best_accuracy:
            self.best_accuracy = current_accuracy
            self.wait = 0
        else:
            self.wait += 1
            
        # Stop if target accuracy reached or patience exceeded
        if current_accuracy >= self.target_accuracy:
            trainer.should_stop = True
        elif self.wait >= self.patience:
            trainer.should_stop = True



def get_available_path_lengths(datamodule) -> List[int]:
    datamodule.setup()
    if hasattr(datamodule.train_dataset, 'paths'):
        lengths = [len(path) for path in datamodule.train_dataset.paths]
        return sorted(list(set(lengths)))
    return []


def interpolate_accuracy_threshold(length: int, min_length: int, max_length: int, 
                                 min_threshold: float, max_threshold: float) -> float:
    if min_length == max_length:
        return min_threshold
    ratio = (length - min_length) / (max_length - min_length)
    return min_threshold + ratio * (max_threshold - min_threshold)


def curriculum_training(cfg: DictConfig) -> float:
    from hydra.core.hydra_config import HydraConfig
    import wandb
    # Get available path lengths from the dataset
    temp_datamodule = PathDataModule(
        train_file=cfg.data.train_file,
        test_file=cfg.data.test_file,
        batch_size=cfg.training.batch_size,
        num_workers=cfg.data.num_workers,
        max_path_length=cfg.data.max_path_length,
        vocab_size=cfg.model.vocab_size,
        data_dir=cfg.paths.data_dir,
        graph_type=cfg.graph_generation.type
    )
    
    available_lengths = get_available_path_lengths(temp_datamodule)
    if not available_lengths:
        raise ValueError("No training data found")
    
    min_length = min(available_lengths)
    max_length = max(available_lengths)
    print(f"Available path lengths: {available_lengths}")
    
    # Set up model (shared across all curriculum stages)
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
    
    # Set up logger (persistent across curriculum stages)
    hydra_cfg = HydraConfig.get()
    job_id = hydra_cfg.job.get('num', 0) if hydra_cfg.job.get('num') is not None else 0
    
    hyperparams = f"d{cfg.model.d_model}__l{cfg.model.num_layers}_lr{cfg.training.learning_rate:.1e}"
    experiment_name = f"{cfg.logging.experiment_name}_curriculum_{hyperparams}_trial{job_id}"
    
    logger = WandbLogger(
        project=cfg.logging.project_name,
        name=experiment_name,
        config=dict(cfg),
        log_model=False,
        group=None
    )
    
    final_val_loss = float("inf")
    
    # Train on each path length sequentially
    for i,length in enumerate(available_lengths):
        model.model.num_timesteps = min(i + 1,20)
        print(f"Training on path length: {length}")
        
        # Calculate target accuracy for this length
        target_accuracy = interpolate_accuracy_threshold(
            length, min_length, max_length,
            cfg.curriculum_learning.min_accuracy_threshold,
            cfg.curriculum_learning.max_accuracy_threshold
        )
        
        # Create data module for this specific length
        datamodule = PathDataModule(
            train_file=cfg.data.train_file,
            test_file=cfg.data.test_file,
            batch_size=cfg.training.batch_size,
            num_workers=cfg.data.num_workers,
            max_path_length=cfg.data.max_path_length,
            vocab_size=cfg.model.vocab_size,
            data_dir=cfg.paths.data_dir,
            graph_type=cfg.graph_generation.type,
            curriculum_length=length
        )
        
        # Set up callbacks for this curriculum stage
        callbacks = []
        
        checkpoint_callback = ModelCheckpoint(
            dirpath=cfg.paths.checkpoint_dir,
            filename=f'length_{length}_epoch_{{epoch}}_val_loss_{{val_loss:.2f}}',
            save_top_k=1,
            monitor='val_per_seq_acc',
            mode='max'
        )
        callbacks.append(checkpoint_callback)
        
        # Curriculum-specific accuracy callback
        curriculum_callback = CurriculumAccuracyCallback(
            target_accuracy=target_accuracy,
            patience=cfg.curriculum_learning.patience
        )
        callbacks.append(curriculum_callback)
        
        # Set up trainer for this curriculum stage
        trainer = pl.Trainer(
            max_epochs=cfg.curriculum_learning.max_epochs_per_length,
            logger=logger,
            callbacks=callbacks,
            gradient_clip_val=cfg.training.gradient_clip_val,
            log_every_n_steps=cfg.logging.log_every_n_steps,
            enable_checkpointing=True,
            enable_progress_bar=True
        )
        
        # Log curriculum stage info
        logger.log_metrics({
            f"curriculum/length": length,
            f"curriculum/target_accuracy": target_accuracy,
        })
        
        # Train on this length
        trainer.fit(model, datamodule)
        
        # Update final validation loss
        current_val_loss = trainer.callback_metrics.get("val_loss", float("inf"))
        if current_val_loss < final_val_loss:
            final_val_loss = current_val_loss
    
    wandb.finish()
    return final_val_loss
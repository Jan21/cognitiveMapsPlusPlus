"""
Training callbacks for PyTorch Lightning.

This module contains all callback classes used during training.
"""

import os
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
import csv
from pathlib import Path
from datetime import datetime
import shutil

class ResultsLogger(Callback):
    """
    Logs final training results to a CSV file.
    Useful for tracking results across Hydra multirun sweeps.
    """

    def __init__(self, csv_path: str = "sweep_results.csv", config: dict = None):
        """
        Args:
            csv_path: Path to CSV file where results will be logged
            config: Hydra config dict to log hyperparameters
        """
        super().__init__()
        self.csv_path = csv_path
        self.config = config or {}

        # Create directory if it doesn't exist
        Path(csv_path).parent.mkdir(parents=True, exist_ok=True)

        # Check if file exists to determine if we need to write header
        self.file_exists = os.path.exists(csv_path)

    def on_train_end(self, trainer, pl_module):
        """Called when training ends."""

        # Collect metrics
        metrics = trainer.callback_metrics

        # Prepare row data
        row_data = {
            'timestamp': datetime.now().isoformat(),
            'num_train_samples': self.config.get('data', {}).get('num_train_samples', 'N/A'),
            'num_val_samples': self.config.get('data', {}).get('num_val_samples', 'N/A'),
            'learning_rate': self.config.get('training', {}).get('learning_rate', 'N/A'),
            'batch_size': self.config.get('training', {}).get('batch_size', 'N/A'),
            'max_epochs': self.config.get('training', {}).get('max_epochs', 'N/A'),
            'actual_epochs': trainer.current_epoch + 1,
            'val_loss': metrics.get('val_loss', float('inf')).item() if 'val_loss' in metrics else 'N/A',
            'val_accuracy': metrics.get('val_accuracy', 'N/A'),
            'val_exact_match': metrics.get('val_exact_match', 'N/A'),
            'train_loss': metrics.get('train_loss_epoch', 'N/A'),
            'graph_type': self.config.get('graph_generation', {}).get('type', 'N/A'),
            'model_type': self.config.get('model', {}).get('_target_', 'N/A'),
        }

        # Convert tensor values to float
        for key, value in row_data.items():
            if hasattr(value, 'item'):
                row_data[key] = value.item()

        # Write to CSV
        file_mode = 'a' if self.file_exists else 'w'
        with open(self.csv_path, file_mode, newline='') as f:
            writer = csv.DictWriter(f, fieldnames=row_data.keys())

            # Write header if file is new
            if not self.file_exists:
                writer.writeheader()
                self.file_exists = True

            writer.writerow(row_data)

        print(f"\n✓ Results logged to: {self.csv_path}")
        print(f"  - num_train_samples: {row_data['num_train_samples']}")
        print(f"  - val_loss: {row_data['val_loss']}")
        print(f"  - val_exact_match: {row_data['val_exact_match']}")

class SimplePruningCallback(Callback):
    """
    Early stopping callback for pruning unpromising trials in hyperparameter optimization.

    Stops training if validation loss hasn't improved for a certain number of epochs
    and the loss is above a threshold.

    Args:
        patience: Number of epochs to wait for improvement before stopping
        min_delta: Minimum change in loss to be considered an improvement
    """
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


def setup_callbacks(cfg, experiment_name, hydra_cfg):
    """
    Set up all training callbacks based on configuration.

    Args:
        cfg: OmegaConf configuration object
        experiment_name: Name of the current experiment/run
        hydra_cfg: Hydra configuration object

    Returns:
        List of callback instances
    """
    callbacks = []

    # Main checkpoint callback - saves top K models based on validation metric
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.paths.checkpoint_dir,
        filename='{epoch}-{val_loss:.2f}',
        save_top_k=3,
        monitor='val_accuracy',
        mode='max'
    )
    callbacks.append(checkpoint_callback)

    # Per-epoch checkpoint callback - saves all epochs if enabled
    if cfg.training.get('save_every_epoch', False):
        # Create run-specific temp directory
        run_checkpoint_dir = os.path.join('temp', 'checkpoints', experiment_name)
        if os.path.exists(run_checkpoint_dir):
            shutil.rmtree(run_checkpoint_dir)
        os.makedirs(run_checkpoint_dir, exist_ok=True)

        epoch_checkpoint_callback = ModelCheckpoint(
            dirpath=run_checkpoint_dir,
            filename='epoch_{epoch:03d}',
            save_top_k=-1,  # Save all checkpoints
            every_n_epochs=1,  # Save after every epoch
            save_last=True  # Also save the last checkpoint
        )
        callbacks.append(epoch_checkpoint_callback)

    # Early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_exact_match',
        patience=500,
        mode='max'
    )
    callbacks.append(early_stopping)

    # Add simple pruning callback for multirun (hyperparameter optimization)
    if hydra_cfg.mode == hydra_cfg.mode.MULTIRUN:
        pruning_callback = SimplePruningCallback(patience=7)
        callbacks.append(pruning_callback)

        # Import and add results logger to save results to CSV
        from model.results_logger import ResultsLogger
        results_logger = ResultsLogger(
            csv_path=f"temp/{cfg.logging.experiment_name}_{cfg.graph_generation.type}.csv",
            config=dict(cfg)
        )
        callbacks.append(results_logger)

    return callbacks

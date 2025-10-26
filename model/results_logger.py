"""
Callback to log training results to CSV file for multirun sweeps.
"""

import csv
import os
from pathlib import Path
from datetime import datetime
from pytorch_lightning.callbacks import Callback


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

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional
import os

from .dataset import create_dataset, collate_fn


class PathDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_file: str,
        test_file: str,
        batch_size: int = 32,
        num_workers: int = 4,
        max_path_length: int = 64,
        vocab_size: int = 10000,
        data_dir: str = "./data",
        graph_type: str = "sphere"
    ):
        super().__init__()
        self.train_file = os.path.join(data_dir, f"train_{graph_type}.json")
        self.test_file = os.path.join(data_dir, f"test_{graph_type}.json")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_path_length = max_path_length
        self.vocab_size = vocab_size
        
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
    
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.train_dataset = create_dataset(
                self.train_file, 
                self.max_path_length, 
                self.vocab_size
            )
            
            # Use test file as validation for simplicity
            self.val_dataset = create_dataset(
                self.test_file, 
                self.max_path_length, 
                self.vocab_size
            )
        
        if stage == "test" or stage is None:
            self.test_dataset = create_dataset(
                self.test_file, 
                self.max_path_length, 
                self.vocab_size
            )
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional
import os

from .dataset import PathDataset, collate_fn, SpreadPathDataset


dataset_type_to_dataset = {
    "path": PathDataset,
    "spread": SpreadPathDataset
}

dataset_type_to_collate_fn = {
    "path": lambda pad_token: lambda batch: collate_fn(pad_token, batch), 
    "spread": lambda pad_token: None,
}

class PathDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_file: str,
        test_file: str,
        batch_size: int = 32,
        num_workers: int = 4,
        max_path_length: int = 33,
        vocab_size: int = 10000,
        dataset_type: str = "spread",
        graph_type: str = "sphere",
        data_dir: str = "temp",
    ):
        super().__init__()
        self.train_file = os.path.join(data_dir, f"train_{graph_type}.json")
        self.test_file = os.path.join(data_dir, f"test_{graph_type}.json")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_path_length = max_path_length
        self.vocab_size = vocab_size
        self.pad_token = vocab_size - 2
        self.tensor_length = max_path_length - 1
        self.dataset_type = dataset_type  # "path" or "spread"


        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.dataset_class = dataset_type_to_dataset[self.dataset_type]
        self.collate_fn = dataset_type_to_collate_fn[self.dataset_type](self.pad_token)


    def setup(self, stage: Optional[str] = None):
        self.train_dataset = self.dataset_class(
            json_file=self.train_file,
            tensor_length=self.tensor_length,
            max_path_length=self.max_path_length,
            vocab_size=self.vocab_size
        )
        self.val_dataset = self.dataset_class(
            json_file=self.test_file,
            tensor_length=self.tensor_length,
            max_path_length=self.max_path_length,
            vocab_size=self.vocab_size
        )
        self.test_dataset = self.dataset_class(
            json_file=self.test_file,
            tensor_length=self.tensor_length,
            max_path_length=self.max_path_length,
            vocab_size=self.vocab_size
        )
       

    def train_dataloader(self):
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self.collate_fn
            )

    def val_dataloader(self):
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self.collate_fn
            )

    def test_dataloader(self):
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self.collate_fn
            )
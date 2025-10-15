import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import Optional
import os

from .dataset import create_dataset, create_curriculum_dataset, collate_fn, SpreadPathDataset

# Conditionally import PyTorch Geometric components
try:
    from torch_geometric.loader import DataLoader as GeometricDataLoader
    from .gnn_dataset import BipartitePathDataset
    PYTORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    PYTORCH_GEOMETRIC_AVAILABLE = False
    GeometricDataLoader = None
    BipartitePathDataset = None

# Import RNN dataset
from .rnn_dataset import RNNMiddleNodeDataset


class PathDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_file: str,
        test_file: str,
        batch_size: int = 32,
        num_workers: int = 4,
        max_path_length: int = 33,
        vocab_size: int = 10000,
        data_dir: str = "./data",
        graph_type: str = "sphere",
        curriculum_length: Optional[int] = None,
        use_gnn: bool = False,
        use_rnn: bool = False,
        graph_file: Optional[str] = None
    ):
        super().__init__()
        self.train_file = os.path.join(data_dir, f"train_{graph_type}.json")
        self.test_file = os.path.join(data_dir, f"test_{graph_type}.json")
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_path_length = max_path_length
        self.vocab_size = vocab_size
        self.pad_token = vocab_size - 2
        self.curriculum_length = curriculum_length
        self.use_gnn = use_gnn
        self.use_rnn = use_rnn
        self.tensor_length = max_path_length - 1

        # Set graph file path for GNN/RNN models
        if graph_file is None:
            self.graph_file = os.path.join(data_dir, f"graph_{graph_type}.pkl")
        else:
            self.graph_file = graph_file

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # Validate GNN usage
        if self.use_gnn and not PYTORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "PyTorch Geometric is required for GNN models. "
                "Install it with: pip install torch-geometric"
            )

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            if self.use_rnn:
                # Use RNN dataset for middle-node prediction
                self.train_dataset = SpreadPathDataset(
                    json_file=self.train_file,
                    tensor_length=self.tensor_length,
                    max_path_length=self.max_path_length,
                    vocab_size=self.vocab_size
                )
                self.val_dataset = SpreadPathDataset(
                    json_file=self.test_file,
                    tensor_length=self.tensor_length,
                    max_path_length=self.max_path_length,
                    vocab_size=self.vocab_size
                )
            elif self.use_gnn:
                # Use GNN dataset (no curriculum support yet for GNN)
                self.train_dataset = BipartitePathDataset(
                    json_file=self.train_file,
                    graph_file=self.graph_file,
                    max_path_length=self.max_path_length,
                    vocab_size=self.vocab_size
                )
                self.val_dataset = BipartitePathDataset(
                    json_file=self.test_file,
                    graph_file=self.graph_file,
                    max_path_length=self.max_path_length,
                    vocab_size=self.vocab_size
                )
            else:
                # Use standard dataset
                if self.curriculum_length is not None:
                    self.train_dataset = create_curriculum_dataset(
                        self.train_file,
                        self.curriculum_length,
                        self.max_path_length,
                        self.vocab_size
                    )
                    self.val_dataset = create_curriculum_dataset(
                        self.test_file,
                        self.curriculum_length,
                        self.max_path_length,
                        self.vocab_size
                    )
                else:
                    self.train_dataset = create_dataset(
                        self.train_file,
                        self.max_path_length,
                        self.vocab_size
                    )
                    self.val_dataset = create_dataset(
                        self.test_file,
                        self.max_path_length,
                        self.vocab_size
                    )

        if stage == "test" or stage is None:
            if self.use_rnn:
                self.test_dataset = SpreadPathDataset(
                    json_file=self.test_file,
                    tensor_length=self.tensor_length,
                    max_path_length=self.max_path_length,
                    vocab_size=self.vocab_size
                )
            elif self.use_gnn:
                self.test_dataset = BipartitePathDataset(
                    json_file=self.test_file,
                    graph_file=self.graph_file,
                    max_path_length=self.max_path_length,
                    vocab_size=self.vocab_size
                )
            else:
                self.test_dataset = create_dataset(
                    self.test_file,
                    self.max_path_length,
                    self.vocab_size
                )
    def collate_fn_with_pad_token(self, batch):
            return collate_fn(self.pad_token, batch)

    def train_dataloader(self):
        if self.use_gnn:
            return GeometricDataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                follow_batch=['x_v', 'x_e']
            )
        elif self.use_rnn:
            # RNN dataset returns simple dict, no special collation needed
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True
            )
        else:
            return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self.collate_fn_with_pad_token
            )

    def val_dataloader(self):
        if self.use_gnn:
            return GeometricDataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                follow_batch=['x_v', 'x_e']
            )
        elif self.use_rnn:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
        else:
            return DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self.collate_fn_with_pad_token
            )

    def test_dataloader(self):
        if self.use_gnn:
            return GeometricDataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                follow_batch=['x_v', 'x_e']
            )
        elif self.use_rnn:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True
            )
        else:
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
                collate_fn=self.collate_fn_with_pad_token
            )
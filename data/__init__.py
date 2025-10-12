"""Data module for path prediction tasks."""

from .dataset import (
    PathDataset,
    DiffusionPathDataset,
    CurriculumPathDataset,
    collate_fn,
    create_dataset,
    create_curriculum_dataset
)
from .datamodule import PathDataModule

__all__ = [
    'PathDataset',
    'DiffusionPathDataset',
    'CurriculumPathDataset',
    'collate_fn',
    'create_dataset',
    'create_curriculum_dataset',
    'PathDataModule',
]

# Conditionally export GNN components if available
try:
    from .gnn_dataset import BipartitePathDataset
    __all__.extend(['BipartitePathDataset'])
except ImportError:
    pass

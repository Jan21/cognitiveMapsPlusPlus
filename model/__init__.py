"""Model module for path prediction architectures."""

from .base_lightningmodule import BasePathPredictionModule
from .lightningmodule import PathPredictionModule, DiffusionPathPredictionModule
from .model_factory import create_model

__all__ = [
    'BasePathPredictionModule',
    'PathPredictionModule',
    'DiffusionPathPredictionModule',
    'create_model',
]

# Conditionally export GNN Lightning module if PyTorch Geometric is available
try:
    from .gnn_lightningmodule import GNNPathPredictionModule
    __all__.append('GNNPathPredictionModule')
except ImportError:
    pass

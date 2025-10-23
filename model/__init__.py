"""Model module for path prediction architectures."""

from .lightningmodule import PathPredictionModule

lightning_module_class = {
    'path': PathPredictionModule,
}

__all__ = [
    'PathPredictionModule',
    'lightning_module_class',
]

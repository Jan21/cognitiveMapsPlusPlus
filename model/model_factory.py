from hydra.utils import instantiate
from omegaconf import DictConfig
from model.architecture.model import TransformerModel
from model.architecture.diffusion_model import DiffusionModel
from model.architecture.past import PAST, PASTConfig

def create_model(model_config: DictConfig, vocab_size: int):
    """Factory function to create models based on configuration."""
    
    # Set vocab_size if not already set
    if model_config.vocab_size is None:
        model_config.vocab_size = vocab_size
    
    # Handle PAST model specially as it needs PASTConfig
    if model_config._target_ == "model.architecture.past.PAST":
        past_config = PASTConfig(
            vocab_size=model_config.vocab_size,
            n_layer=model_config.num_layers,
            n_head=model_config.num_heads,
            n_embd=model_config.d_model,
            dropout=model_config.dropout,
            train_mode=model_config.get("train_mode", "absorbing")
        )
        return PAST(past_config)
    
    # For other models, use hydra's instantiate
    return instantiate(model_config)
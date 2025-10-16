"""
Factory for creating different architectural variants.
"""

from .rnn_middle_predictor import RNNMiddleNodePredictor
from .rnn_middle_predictor_variants import (
    VariantV1_ResidualUpsample,
    VariantV2_TransformerBased,
    VariantV3_UNetStyle,
    VariantV4_SimpleProgressive,
    VariantV5_GRUBased,
)


VARIANT_REGISTRY = {
    "original": RNNMiddleNodePredictor,
    "v1_residual": VariantV1_ResidualUpsample,
    "v2_transformer": VariantV2_TransformerBased,
    "v3_unet": VariantV3_UNetStyle,
    "v4_simple": VariantV4_SimpleProgressive,
    "v5_gru": VariantV5_GRUBased,
}


def create_model(variant: str, **kwargs):
    """
    Create a model based on the variant name.

    Args:
        variant: Name of the variant (see VARIANT_REGISTRY keys)
        **kwargs: Arguments to pass to the model constructor

    Returns:
        Instantiated model

    Example:
        model = create_model("v1_residual", vocab_size=1000, d_model=128)
    """
    if variant not in VARIANT_REGISTRY:
        available = ", ".join(VARIANT_REGISTRY.keys())
        raise ValueError(f"Unknown variant '{variant}'. Available variants: {available}")

    model_class = VARIANT_REGISTRY[variant]
    return model_class(**kwargs)


def list_variants():
    """List all available model variants"""
    return list(VARIANT_REGISTRY.keys())

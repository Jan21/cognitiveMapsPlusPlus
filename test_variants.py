#!/usr/bin/env python3
"""
Script to test all model variants and ensure they work correctly.
Tests forward pass, backward pass, and loss computation.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F

# Direct imports to avoid dependency issues
from model.architecture.rnn_middle_predictor_variants import (
    VariantV1_ResidualUpsample,
    VariantV2_TransformerBased,
    VariantV3_UNetStyle,
    VariantV4_SimpleProgressive,
    VariantV5_GRUBased,
)
from model.architecture.rnn_middle_predictor import RNNMiddleNodePredictor

# Local variant registry
VARIANT_REGISTRY = {
    "original": RNNMiddleNodePredictor,
    "v1_residual": VariantV1_ResidualUpsample,
    "v2_transformer": VariantV2_TransformerBased,
    "v3_unet": VariantV3_UNetStyle,
    "v4_simple": VariantV4_SimpleProgressive,
    "v5_gru": VariantV5_GRUBased,
}

def create_model(variant: str, **kwargs):
    """Create a model based on variant name"""
    if variant not in VARIANT_REGISTRY:
        raise ValueError(f"Unknown variant '{variant}'")
    return VARIANT_REGISTRY[variant](**kwargs)

def list_variants():
    """List all available variants"""
    return list(VARIANT_REGISTRY.keys())


def create_dummy_batch(batch_size=4, vocab_size=100, upscale_depth=5):
    """Create a dummy batch matching SpreadPathDataset format"""
    batch = {
        'input': torch.randint(0, vocab_size - 1, (batch_size, 33))
    }

    # Add labels for each level
    for level in range(upscale_depth):
        seq_len = 2 ** (level + 1) + 1
        labels = torch.randint(0, vocab_size - 1, (batch_size, seq_len))
        # Mask first and last (start/end nodes)
        labels[:, 0] = -100
        labels[:, -1] = -100
        batch[level] = labels

    return batch


def test_variant(variant_name, vocab_size=100, d_model=64):
    """Test a single variant"""
    print(f"\n{'='*60}")
    print(f"Testing: {variant_name}")
    print(f"{'='*60}")

    try:
        # Create model
        model = create_model(
            variant=variant_name,
            vocab_size=vocab_size,
            d_model=d_model,
            num_iterations=2,
            upscale_depth=3,  # Smaller for faster testing
            dropout=0.1,
            nhead=4
        )

        # Count parameters
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"✓ Model created successfully")
        print(f"  Parameters: {n_params:,}")

        # Test forward pass
        batch = create_dummy_batch(batch_size=4, vocab_size=vocab_size, upscale_depth=3)
        output = model(batch)

        print(f"✓ Forward pass successful")
        print(f"  Logits shape: {output['logits'].shape}")

        # Check loss exists
        assert 'loss' in output, "Loss not in output"
        loss = output['loss']
        print(f"✓ Loss computed: {loss.item():.4f}")

        # Test backward pass
        loss.backward()
        print(f"✓ Backward pass successful")

        # Check gradients
        has_grad = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for _ in model.parameters())
        print(f"✓ Gradients computed: {has_grad}/{total_params} parameters")

        # Check for NaN/Inf
        has_nan = any(torch.isnan(p.grad).any() for p in model.parameters() if p.grad is not None)
        has_inf = any(torch.isinf(p.grad).any() for p in model.parameters() if p.grad is not None)

        if has_nan:
            print(f"⚠ WARNING: NaN in gradients!")
            return False
        if has_inf:
            print(f"⚠ WARNING: Inf in gradients!")
            return False

        print(f"✓ No NaN/Inf in gradients")

        # Test prediction mode
        model.eval()
        with torch.no_grad():
            output_eval = model(batch)
        print(f"✓ Eval mode works")

        # Test gradient clipping (common in training)
        model.zero_grad()
        loss = model(batch)['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        print(f"✓ Gradient clipping works")

        print(f"\n✅ {variant_name}: ALL TESTS PASSED")
        return True

    except Exception as e:
        print(f"\n❌ {variant_name}: FAILED")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_variants():
    """Test all available variants"""
    variants = list_variants()
    print(f"Testing {len(variants)} variants: {variants}")

    results = {}
    for variant in variants:
        results[variant] = test_variant(variant)

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")

    passed = sum(results.values())
    total = len(results)

    for variant, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}: {variant}")

    print(f"\nTotal: {passed}/{total} passed")

    return all(results.values())


def compare_model_sizes():
    """Compare model sizes across variants"""
    print(f"\n{'='*60}")
    print("MODEL SIZE COMPARISON")
    print(f"{'='*60}")

    vocab_size = 1000
    d_model = 128

    print(f"Configuration: vocab_size={vocab_size}, d_model={d_model}")
    print(f"\n{'Variant':<20} {'Parameters':>15} {'Ratio':>10}")
    print("-" * 50)

    sizes = {}
    for variant in list_variants():
        try:
            model = create_model(
                variant=variant,
                vocab_size=vocab_size,
                d_model=d_model,
                num_iterations=3,
                upscale_depth=5,
                dropout=0.1,
                nhead=4
            )
            n_params = sum(p.numel() for p in model.parameters())
            sizes[variant] = n_params
        except Exception as e:
            sizes[variant] = None
            print(f"{'ERROR':<20} {str(e):<15}")

    # Find baseline for ratio
    baseline = sizes.get('original', max(s for s in sizes.values() if s))

    for variant, n_params in sorted(sizes.items(), key=lambda x: x[1] or 0, reverse=True):
        if n_params:
            ratio = n_params / baseline
            print(f"{variant:<20} {n_params:>15,} {ratio:>9.2f}x")


def test_training_step():
    """Simulate a few training steps"""
    print(f"\n{'='*60}")
    print("SIMULATED TRAINING TEST")
    print(f"{'='*60}")

    variant = "v4_simple"  # Most reliable
    print(f"Using variant: {variant}")

    model = create_model(
        variant=variant,
        vocab_size=100,
        d_model=128,
        num_iterations=2,
        upscale_depth=3,
        dropout=0.1
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    print("\nTraining for 5 steps...")
    losses = []

    model.train()
    for step in range(5):
        batch = create_dummy_batch(batch_size=8, vocab_size=100, upscale_depth=3)

        optimizer.zero_grad()
        output = model(batch)
        loss = output['loss']
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f"Step {step+1}: loss = {loss.item():.4f}")

    # Check if loss is changing (not stuck)
    if len(set(losses)) == 1:
        print("⚠ WARNING: Loss is not changing!")
    else:
        print("✓ Loss is changing (model is learning)")

    # Check loss is reasonable
    if losses[-1] > 100 or losses[-1] < 0:
        print(f"⚠ WARNING: Loss seems unreasonable: {losses[-1]}")
    else:
        print("✓ Loss is in reasonable range")

    print("\n✅ Training simulation completed")


if __name__ == "__main__":
    print("Model Variants Test Suite")
    print("=" * 60)

    # Test all variants work
    all_passed = test_all_variants()

    # Compare sizes
    compare_model_sizes()

    # Test training
    test_training_step()

    print("\n" + "=" * 60)
    if all_passed:
        print("✅ ALL TESTS PASSED - Variants are ready to use!")
    else:
        print("❌ SOME TESTS FAILED - Check errors above")
    print("=" * 60)

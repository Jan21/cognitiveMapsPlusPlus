# Model Variants for RNN Middle Predictor

## Quick Start

Your original model wasn't learning. I've created **5 improved variants** that address common training issues.

### 🚀 Recommended: Start with Variant 4

```yaml
# Add to your config
model:
  variant: v4_simple
  d_model: 256
  num_iterations: 2
  upscale_depth: 5
  dropout: 0.15
```

**Why V4?** Most reliable, parameter-efficient, and easiest to train.

---

## Available Variants

| Variant | Best For | Key Feature |
|---------|----------|-------------|
| `v1_residual` | Stable training | Proper residual connections + LayerNorm |
| `v2_transformer` | Long-range dependencies | Self-attention mechanism |
| `v3_unet` | Fine details | Skip connections across levels |
| `v4_simple` | ⭐ **Start here** | Shared parameters, reliable |
| `v5_gru` | Sequential structure | Bidirectional GRU + attention |

---

## Test the Variants

```bash
python test_variants.py
```

This will:
- ✓ Test all variants work
- ✓ Compare model sizes
- ✓ Simulate training steps
- ✓ Check for NaN/Inf issues

---

## What Was Fixed

The original model had these issues:
1. ❌ Improper residual connections (dimension mismatch)
2. ❌ No normalization layers (training instability)
3. ❌ `num_iterations` hardcoded to 1
4. ❌ Poor gradient flow through deep network
5. ❌ Suboptimal initialization

All variants fix these problems. ✅

---

## Usage Example

### In Your Config File

```yaml
# config/training/default.yaml
model:
  variant: v4_simple  # Change this to try different variants
  d_model: 256
  num_iterations: 2
  upscale_depth: 5
  dropout: 0.15

learning_rate: 3e-3
batch_size: 64
max_epochs: 500
```

### In Code (if needed)

```python
from model.architecture.variant_factory import create_model

model = create_model(
    variant="v4_simple",
    vocab_size=1000,
    d_model=256,
    num_iterations=2,
    upscale_depth=5,
    dropout=0.15
)
```

---

## Troubleshooting

### Model still not learning?

1. **Check data**: Print a batch and verify it looks correct
2. **Try smaller model**: `d_model: 64`
3. **Lower learning rate**: `learning_rate: 1e-3`
4. **Overfit one batch**: Make sure loss can decrease on a single batch

### Model overfitting?

1. **Increase dropout**: `dropout: 0.2`
2. **Add weight decay**: `weight_decay: 1e-4`
3. **Use v4_simple**: Parameter sharing helps regularization

---

## Detailed Documentation

See [docs/model_variants_analysis.md](docs/model_variants_analysis.md) for:
- Detailed explanations of each variant
- Training tips and recommendations
- Debugging checklist
- Expected performance characteristics

---

## File Structure

```
model/architecture/
├── rnn_middle_predictor.py          # Original model
├── rnn_middle_predictor_variants.py # 5 new variants
└── variant_factory.py               # Factory to create models

config/
└── model_variants.yaml              # Example configs for each variant

docs/
└── model_variants_analysis.md       # Detailed analysis

test_variants.py                      # Test script
```

---

## Next Steps

1. **Test variants**: `python test_variants.py`
2. **Try v4_simple**: Update your config with the settings above
3. **Train**: Run your training script
4. **Monitor**: Watch `val_loss`, `val_accuracy`, `val_validity`
5. **Iterate**: If v4 doesn't work, try v1, v2, v3, or v5

Good luck! 🎯

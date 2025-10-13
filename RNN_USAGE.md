# RNN Middle-Node Predictor - Usage Guide

## Overview

The RNN middle-node predictor is a simple recurrent model that predicts an intermediate waypoint on a path given the start and end nodes.

## Default Configuration

The RNN model is now set as the **default** in `config/config.yaml`. To train with default settings:

```bash
python train.py
```

## Architecture

**Input:** Start node ID + End node ID

**Process:**
1. Embed start and end nodes separately
2. Concatenate embeddings: `[start_emb, end_emb]` (size: 2 × d_model)
3. Initialize LSTM hidden state (h, c) to zeros
4. Run N recurrent iterations with LSTMCell:
   - Input: concatenated embeddings (constant)
   - Hidden state: evolves with each iteration
5. Project final hidden state to vocabulary size
6. Predict middle node via argmax or sampling

**Output:** Predicted middle node ID

## Configuration

### Model Config (`config/model/rnn_middle_predictor.yaml`)

```yaml
d_model: 128              # Embedding dimension
num_iterations: 6         # Number of recurrent updates
dropout: 0.1             # Dropout probability
vocab_size: null         # Computed automatically from graph
use_rnn: true            # Enable RNN mode
```

### Switching Between Models

**Use RNN (default):**
```bash
python train.py model=rnn_middle_predictor
```

**Use GNN:**
```bash
python train.py model=bipartite_gnn
```

**Use Diffusion Transformer:**
```bash
python train.py model=diffusion
```

## Hyperparameter Tuning

### Embedding Dimension
```bash
python train.py model.d_model=256
```

### Number of Recurrent Iterations
```bash
python train.py model.num_iterations=10
```

### Learning Rate
```bash
python train.py training.learning_rate=1e-3
```

### Batch Size
```bash
python train.py training.batch_size=64
```

## Dataset

The RNN model uses `RNNMiddleNodeDataset` which:
- Requires paths with ≥3 nodes (start, middle, end)
- Randomly selects one intermediate node as the "middle" for each path
- Returns batches with: `start_id`, `end_id`, `middle_id`

## Metrics

During training/validation, the following metrics are logged:
- `train_loss`: Cross-entropy loss on training set
- `val_loss`: Cross-entropy loss on validation set
- `val_accuracy`: Accuracy of middle-node predictions
- `val_exact_match`: Same as accuracy (for compatibility)

## Example Training Command

```bash
# Train RNN with custom hyperparameters
python train.py \
  model=rnn_middle_predictor \
  model.d_model=256 \
  model.num_iterations=10 \
  training.learning_rate=1e-3 \
  training.batch_size=128
```

## Model Comparison

| Model | Task | Input | Architecture |
|-------|------|-------|--------------|
| **RNN** | Predict 1 middle node | Start + End | Recurrent (LSTMCell) |
| **GNN** | Predict 1 middle node | Start + Middle (masked) + End | Bipartite message passing |
| **Diffusion** | Predict full path | Start + End | Transformer with denoising |

## Files

- Dataset: `data/rnn_dataset.py`
- Architecture: `model/architecture/rnn_middle_predictor.py`
- Lightning Module: `model/rnn_lightningmodule.py`
- Config: `config/model/rnn_middle_predictor.yaml`

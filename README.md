# CognitiveMaps++

A comprehensive framework for experimenting with transformer-based models on graph navigation tasks to analyze latent geometries of world models.

## Overview

This repository implements a configurable framework for training and evaluating neural network models on graph navigation problems. The project explores how neural networks learn spatial representations by training models to predict optimal paths through different graph topologies.

## Directory Structure

```
cognitiveMapsPlusPlus/
├── config/               # Hydra configuration files
│   └── config.yaml      # Main configuration file
├── data/                # Data handling modules
│   ├── datamodule.py    # PyTorch Lightning data module
│   └── dataset.py       # Custom dataset classes and collation
├── generate/            # Graph and data generation scripts
│   ├── generate_data.py # Training/test data generation
│   └── generate_graph.py # Graph structure generation
├── model/               # Neural network components
│   ├── lightningmodule.py # PyTorch Lightning training module
│   ├── metrics.py       # Evaluation metrics
│   └── model.py         # Transformer architecture
├── train.py            # Main training script
└── requirements.txt    # Python dependencies
```

## Quick Start

### 1. Installation
```bash
# Clone the repository
git clone <repository-url>
cd cognitiveMapsPlusPlus

# Install dependencies
pip install -r requirements.txt
```

### 2. Generate a Graph
```bash
# Generate a sphere graph (default configuration)
python generate/generate_graph.py

# Generate a grid graph
python generate/generate_graph.py graph_generation.type=grid
```

### 3. Generate Training Data
```bash
# Generate training and test datasets
python generate/generate_data.py
```

### 4. Train a Model
```bash
# Single training run
python train.py

# Hyperparameter sweep
python train.py --multirun training.learning_rate=1e-3,1e-4,1e-5
```

## Configuration

The framework uses Hydra for configuration management. Key configuration sections:

### Graph Generation
```yaml
graph_generation:
  type: "sphere"  # "sphere" or "grid"
  sphere_mesh:
    num_horizontal: 20  # Latitude circles
    num_vertical: 20    # Longitude circles
  grid_2d:
    width: 20
    height: 20
```

### Model Architecture
```yaml
model:
  d_model: 64        # Hidden dimension
  num_heads: 4       # Attention heads
  num_layers: 2      # Transformer layers
  d_ff: 64          # Feed-forward dimension
  max_seq_length: 128
  dropout: 0.1
```

### Training Parameters
```yaml
training:
  learning_rate: 1e-4
  batch_size: 125
  max_epochs: 200
  optimizer: "adamw"  # "adamw" or "muon"
  loss: "cross_entropy"
```

### Data Generation
```yaml
data_generation:
  train:
    num_paths: 300000
    use_perturbed: true    # Use perturbed shortest paths
    perturbation_max: 7    # Maximum perturbation steps
  test:
    num_paths: 2000
```


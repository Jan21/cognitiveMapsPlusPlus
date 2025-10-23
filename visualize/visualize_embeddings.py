"""
Visualize node embeddings from trained models using UMAP or PCA.

This script loads a trained model checkpoint and extracts node embeddings,
then visualizes them in 2D or 3D using either UMAP or PCA dimensionality reduction.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from sklearn.decomposition import PCA
from model.rnn_lightningmodule import RNNMiddleNodeModule
from model.gnn_lightningmodule import GNNPathPredictionModule


def load_model_from_checkpoint(checkpoint_path: str, model_type: str = "rnn"):
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to the .ckpt file
        model_type: Type of model ("rnn" or "gnn")

    Returns:
        Loaded PyTorch Lightning module
    """
    print(f"Loading {model_type.upper()} model from {checkpoint_path}")

    if model_type == "rnn":
        model = RNNMiddleNodeModule.load_from_checkpoint(checkpoint_path)
    elif model_type == "gnn":
        model = GNNPathPredictionModule.load_from_checkpoint(checkpoint_path)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Expected 'rnn' or 'gnn'")

    model.eval()
    return model


def extract_node_embeddings(model, model_type: str = "rnn"):
    """
    Extract node embeddings from the model.

    Args:
        model: Trained PyTorch Lightning module
        model_type: Type of model ("rnn" or "gnn")

    Returns:
        numpy array of embeddings [vocab_size, d_model]
    """
    print(f"Extracting embeddings from {model_type.upper()} model")

    with torch.no_grad():
        if model_type == "rnn":
            # RNN model: extract from node_embedding
            embeddings = model.model.node_embedding.weight.cpu().numpy()
        elif model_type == "gnn":
            # GNN model: extract from vertex_embedding
            embeddings = model.model.vertex_embedding.weight.cpu().numpy()
            # Remove padding embedding (last one)
            embeddings = embeddings[:-1]
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    print(f"Extracted embeddings shape: {embeddings.shape}")
    return embeddings


def load_graph_metadata(graph_path: str, graph_type: str = "sphere"):
    """
    Load graph metadata to get node positions for visualization.

    Args:
        graph_path: Path to the graph pickle file
        graph_type: Type of graph ("sphere" or "grid")

    Returns:
        Dictionary with node metadata (positions, etc.)
    """
    # Adjust path based on graph type
    if not graph_path.endswith(f'_{graph_type}.pkl'):
        graph_path = graph_path.replace('.pkl', f'_{graph_type}.pkl')

    print(f"Loading graph from {graph_path}")
    with open(graph_path, 'rb') as f:
        graph = pickle.load(f)

    # Extract node positions if available
    node_positions = {}
    for node in graph.nodes():
        if 'pos' in graph.nodes[node]:
            node_positions[node] = graph.nodes[node]['pos']

    return {
        'graph': graph,
        'positions': node_positions,
        'num_nodes': graph.number_of_nodes()
    }


def reduce_dimensions(embeddings, method='umap', n_components=2):
    """
    Reduce embeddings to lower dimensions using specified method.

    Args:
        embeddings: numpy array of embeddings [num_nodes, d_model]
        method: 'umap' or 'pca'
        n_components: number of dimensions (2 or 3)

    Returns:
        numpy array of reduced embeddings [num_nodes, n_components]
    """
    if method == 'umap':
        if not UMAP_AVAILABLE:
            raise ImportError("UMAP not installed. Install with: pip install umap-learn")
        print("Running UMAP dimensionality reduction...")
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            n_components=n_components,
            metric='cosine',
            random_state=42
        )
        embedding_reduced = reducer.fit_transform(embeddings)
    elif method == 'pca':
        print("Running PCA dimensionality reduction...")
        reducer = PCA(n_components=n_components, random_state=42)
        embedding_reduced = reducer.fit_transform(embeddings)

        # Print explained variance for PCA
        explained_var = reducer.explained_variance_ratio_
        print(f"Explained variance: {explained_var}")
        print(f"Total explained variance: {sum(explained_var):.4f}")
    else:
        raise ValueError(f"Unknown method: {method}. Expected 'umap' or 'pca'")

    return embedding_reduced


def visualize_embeddings(embeddings, graph_metadata=None, save_path=None, method='umap', n_dims=2):
    """
    Visualize embeddings using dimensionality reduction.

    Args:
        embeddings: numpy array of embeddings [num_nodes, d_model]
        graph_metadata: Optional dictionary with graph information
        save_path: Optional path to save the figure
        method: Dimensionality reduction method ('umap' or 'pca')
        n_dims: Number of dimensions for visualization (2 or 3)
    """
    # Reduce dimensions
    embedding_reduced = reduce_dimensions(embeddings, method=method, n_components=n_dims)

    # Create visualization based on dimensions
    if n_dims == 2:
        visualize_2d(embedding_reduced, graph_metadata, save_path, method)
    elif n_dims == 3:
        visualize_3d(embedding_reduced, graph_metadata, save_path, method)
    else:
        raise ValueError(f"n_dims must be 2 or 3, got {n_dims}")


def visualize_2d(embedding_2d, graph_metadata=None, save_path=None, method='umap'):
    """Create 2D visualization of embeddings."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # Plot 1: Reduced embedding
    ax1 = axes[0]
    scatter1 = ax1.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=np.arange(len(embedding_2d)),
        cmap='viridis',
        alpha=0.6,
        s=30
    )
    method_name = method.upper()
    ax1.set_title(f'{method_name} Embedding of Node Representations (2D)', fontsize=14, fontweight='bold')
    ax1.set_xlabel(f'{method_name} Dimension 1')
    ax1.set_ylabel(f'{method_name} Dimension 2')
    plt.colorbar(scatter1, ax=ax1, label='Node ID')

    # Plot 2: If we have graph positions, show actual graph structure
    if graph_metadata and graph_metadata['positions']:
        ax2 = axes[1]
        positions = graph_metadata['positions']

        # Extract positions in order
        node_ids = sorted(positions.keys())
        pos_array = np.array([positions[nid] for nid in node_ids])

        # Handle 3D positions (sphere) vs 2D positions (grid)
        if pos_array.shape[1] == 3:
            # For sphere, use first two dimensions or project to 2D
            scatter2 = ax2.scatter(
                pos_array[:, 0],
                pos_array[:, 1],
                c=node_ids,
                cmap='viridis',
                alpha=0.6,
                s=30
            )
            ax2.set_title('Original Graph Structure (X-Y projection)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('X Position')
            ax2.set_ylabel('Y Position')
        else:
            # For 2D grid
            scatter2 = ax2.scatter(
                pos_array[:, 0],
                pos_array[:, 1],
                c=node_ids,
                cmap='viridis',
                alpha=0.6,
                s=30
            )
            ax2.set_title('Original Graph Structure', fontsize=14, fontweight='bold')
            ax2.set_xlabel('X Position')
            ax2.set_ylabel('Y Position')

        plt.colorbar(scatter2, ax=ax2, label='Node ID')
        ax2.set_aspect('equal')
    else:
        # If no positions available, create a simple plot
        ax2 = axes[1]
        ax2.text(0.5, 0.5, 'No graph position data available',
                ha='center', va='center', fontsize=12)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()


def visualize_3d(embedding_3d, graph_metadata=None, save_path=None, method='umap'):
    """Create 3D visualization of embeddings."""
    fig = plt.figure(figsize=(16, 7))

    # Plot 1: Reduced embedding in 3D
    ax1 = fig.add_subplot(121, projection='3d')
    colors = np.arange(len(embedding_3d))
    scatter1 = ax1.scatter(
        embedding_3d[:, 0],
        embedding_3d[:, 1],
        embedding_3d[:, 2],
        c=colors,
        cmap='viridis',
        alpha=0.6,
        s=30
    )
    method_name = method.upper()
    ax1.set_title(f'{method_name} Embedding of Node Representations (3D)', fontsize=14, fontweight='bold')
    ax1.set_xlabel(f'{method_name} Dimension 1')
    ax1.set_ylabel(f'{method_name} Dimension 2')
    ax1.set_zlabel(f'{method_name} Dimension 3')
    plt.colorbar(scatter1, ax=ax1, label='Node ID', shrink=0.5)

    # Plot 2: If we have graph positions, show actual graph structure
    if graph_metadata and graph_metadata['positions']:
        positions = graph_metadata['positions']
        node_ids = sorted(positions.keys())
        pos_array = np.array([positions[nid] for nid in node_ids])

        if pos_array.shape[1] == 3:
            # 3D graph structure (e.g., sphere)
            ax2 = fig.add_subplot(122, projection='3d')
            scatter2 = ax2.scatter(
                pos_array[:, 0],
                pos_array[:, 1],
                pos_array[:, 2],
                c=node_ids,
                cmap='viridis',
                alpha=0.6,
                s=30
            )
            ax2.set_title('Original Graph Structure (3D)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('X Position')
            ax2.set_ylabel('Y Position')
            ax2.set_zlabel('Z Position')
            plt.colorbar(scatter2, ax=ax2, label='Node ID', shrink=0.5)
        else:
            # 2D graph structure (e.g., grid)
            ax2 = fig.add_subplot(122)
            scatter2 = ax2.scatter(
                pos_array[:, 0],
                pos_array[:, 1],
                c=node_ids,
                cmap='viridis',
                alpha=0.6,
                s=30
            )
            ax2.set_title('Original Graph Structure (2D)', fontsize=14, fontweight='bold')
            ax2.set_xlabel('X Position')
            ax2.set_ylabel('Y Position')
            plt.colorbar(scatter2, ax=ax2, label='Node ID')
            ax2.set_aspect('equal')
    else:
        # If no positions available, create a simple plot
        ax2 = fig.add_subplot(122)
        ax2.text(0.5, 0.5, 'No graph position data available',
                ha='center', va='center', fontsize=12)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize node embeddings from trained models using UMAP or PCA"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/epoch=19-val_loss=0.47.ckpt",
        help="Path to model checkpoint (.ckpt file)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["rnn", "gnn"],
        default="rnn",
        help="Type of model (rnn or gnn)"
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        default="temp/graph_sphere.pkl",
        help="Path to graph pickle file"
    )
    parser.add_argument(
        "--graph-type",
        type=str,
        choices=["sphere", "grid"],
        default="sphere",
        help="Type of graph (sphere or grid)"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["umap", "pca"],
        default="pca",
        help="Dimensionality reduction method (umap or pca)"
    )
    parser.add_argument(
        "--dims",
        type=int,
        choices=[2, 3],
        default=3,
        help="Number of dimensions for visualization (2 or 3)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save the visualization (if not specified, only displays)"
    )

    args = parser.parse_args()

    # Check if UMAP is available when requested
    if args.method == "umap" and not UMAP_AVAILABLE:
        print("Error: UMAP is not installed. Please install with: pip install umap-learn")
        print("Falling back to PCA...")
        args.method = "pca"

    # Load model
    model = load_model_from_checkpoint(args.checkpoint, args.model_type)

    # Extract embeddings
    embeddings = extract_node_embeddings(model, args.model_type)[:381,:]

    # Load graph metadata
    try:
        graph_metadata = load_graph_metadata(args.graph_path, args.graph_type)
    except Exception as e:
        print(f"Warning: Could not load graph metadata: {e}")
        graph_metadata = None

    # Visualize
    output_path = args.output
    #if output_path is None:
    #    # Auto-generate output path
    #    checkpoint_name = Path(args.checkpoint).stem
    #    output_path = f"visualize/embeddings_{args.method}_{args.dims}d_{args.model_type}_{checkpoint_name}.png"

    visualize_embeddings(embeddings, graph_metadata, output_path, method=args.method, n_dims=args.dims)

    print("\nDone!")


if __name__ == "__main__":
    main()

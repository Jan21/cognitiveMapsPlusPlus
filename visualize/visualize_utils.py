"""
Utility functions for visualizing embeddings during training.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
import umap


def generate_node_colors(graph, num_vertices):
    """
    Generate colors for nodes based on their properties in the graph.

    Args:
        graph: NetworkX graph object
        num_vertices: Number of vertices to color

    Returns:
        Array of colors for each node
    """
    # Simple coloring based on node index
    # You can customize this based on graph properties
    colors = np.zeros(num_vertices)
    colors[num_vertices // 2:] = 1  # First half 0, second half 1
    return colors


def visualize_embeddings_3d(embeddings, graph, epoch, num_vertices, save_dir):
    """
    Visualize embeddings in 3D using PCA and save to file.

    Args:
        embeddings: Tensor or numpy array of embeddings [vocab_size, d_model]
        graph: NetworkX graph object
        epoch: Current epoch number
        num_vertices: Number of vertices in the graph
        save_dir: Directory to save the visualization
    """
    # Convert to numpy if needed
    if torch.is_tensor(embeddings):
        embeddings = embeddings.detach().cpu().numpy()

    # Remove padding token if present (last embedding)
    if embeddings.shape[0] > num_vertices:
        embeddings = embeddings[:-1, :]

    # Apply PCA to reduce to 3D
    #reducer = PCA(n_components=3, random_state=42)
    reducer = umap.UMAP(n_components=3, random_state=42)
    embeddings_3d = reducer.fit_transform(embeddings)

    # Create figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    num_embs = num_vertices
    assert embeddings_3d.shape[0] == num_embs, f"Expected {num_embs} embeddings but got {embeddings_3d.shape[0]}."

    # Generate node colors once using the helper function
    colors = generate_node_colors(graph, num_embs)

    # Plot the embeddings
    ax.scatter(
        embeddings_3d[:, 0],
        embeddings_3d[:, 1],
        embeddings_3d[:, 2],
        s=50,
        c=colors
    )

    # # Annotate each point with its node index (i) instead of its node id (name)
    # for i in range(num_embs):
    #     ax.text(
    #         embeddings_3d[i, 0],
    #         embeddings_3d[i, 1],
    #         embeddings_3d[i, 2],
    #         str(i),
    #         fontsize=3,
    #         color='black'
    #     )

    ax.set_title(f"Node Embeddings at Epoch {epoch}")
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")

    # Set equal aspect ratio (no stretching!)
    max_range = (embeddings_3d.max(axis=0) - embeddings_3d.min(axis=0)).max() / 2.0
    mid = embeddings_3d.mean(axis=0)
    ax.set_xlim(mid[0] - max_range, mid[0] + max_range)
    ax.set_ylim(mid[1] - max_range, mid[1] + max_range)
    ax.set_zlim(mid[2] - max_range, mid[2] + max_range)

    plt.tight_layout()

    # Create directory if it doesn't exist
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save with higher resolution (dpi=300)
    output_file = save_path / f"embedding_epoch_{epoch}.png"
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

    return output_file

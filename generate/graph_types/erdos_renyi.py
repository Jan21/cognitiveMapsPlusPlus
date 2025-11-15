import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate(n_nodes=100, edge_prob=0.1, seed=42):
    """
    Generate an Erdős-Rényi random graph.

    In this model, each possible edge exists independently with probability p.

    Args:
        n_nodes: Number of nodes in the graph
        edge_prob: Probability of edge creation between any two nodes
        seed: Random seed for reproducibility

    Returns:
        NetworkX graph
    """
    G = nx.erdos_renyi_graph(n_nodes, edge_prob, seed=seed)

    # Add positions using spring layout for visualization
    pos = nx.spring_layout(G, seed=seed, dim=3)
    for node, position in pos.items():
        G.nodes[node]['pos'] = tuple(position)

    return G


def visualize(G, title="Erdős-Rényi Random Graph"):
    """Visualize a random graph in 3D using spring layout"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Get node positions (should already be set by generation function)
    pos = nx.get_node_attributes(G, 'pos')

    if not pos:
        # Fallback: create positions if not available
        pos = nx.spring_layout(G, dim=3, seed=42)
        for node, position in pos.items():
            G.nodes[node]['pos'] = tuple(position)

    # Draw edges
    for edge in G.edges():
        x_coords = [pos[edge[0]][0], pos[edge[1]][0]]
        y_coords = [pos[edge[0]][1], pos[edge[1]][1]]
        z_coords = [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(x_coords, y_coords, z_coords, 'b-', alpha=0.2, linewidth=0.5)

    # Draw nodes
    xs = [pos[node][0] for node in G.nodes()]
    ys = [pos[node][1] for node in G.nodes()]
    zs = [pos[node][2] for node in G.nodes()]

    # Color by degree
    degrees = [G.degree(node) for node in G.nodes()]
    scatter = ax.scatter(xs, ys, zs, c=degrees, s=50, alpha=0.8, cmap='viridis')

    ax.set_title(title)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Add colorbar
    plt.colorbar(scatter, ax=ax, label='Node Degree')

    plt.tight_layout()
    plt.show()

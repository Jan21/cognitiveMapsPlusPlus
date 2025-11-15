import networkx as nx
import matplotlib.pyplot as plt


def generate(width=20, height=20):
    """
    Generate a 2D grid graph.

    Args:
        width: Number of nodes in the x direction
        height: Number of nodes in the y direction

    Returns:
        NetworkX graph representing the 2D grid
    """
    G = nx.Graph()

    # Generate nodes with 2D positions
    node_id = 0
    for y in range(height):
        for x in range(width):
            G.add_node(node_id, pos=(x, y))
            node_id += 1

    # Add horizontal edges
    for y in range(height):
        for x in range(width - 1):
            current = y * width + x
            next_node = y * width + (x + 1)
            G.add_edge(current, next_node)

    # Add vertical edges
    for y in range(height - 1):
        for x in range(width):
            current = y * width + x
            next_node = (y + 1) * width + x
            G.add_edge(current, next_node)

    return G


def visualize(G):
    """Visualize the 2D grid graph"""
    fig, ax = plt.subplots(figsize=(10, 8))

    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    # Draw the graph
    nx.draw(G, pos, ax=ax, node_color='red', node_size=20,
            edge_color='blue', width=0.5, alpha=0.6)

    ax.set_title('2D Grid Graph')
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()

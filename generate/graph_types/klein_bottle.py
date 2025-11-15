import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate(u_points=30, v_points=30):
    """
    Generate a Klein bottle mesh graph using standard 3D immersion.

    The Klein bottle is a non-orientable surface that cannot be embedded in 3D
    without self-intersection, but can be immersed (visualized with self-intersection).

    Args:
        u_points: Number of discretization points in u direction [0, 2π]
        v_points: Number of discretization points in v direction [0, 2π]

    Returns:
        NetworkX graph representing the Klein bottle mesh
    """
    G = nx.Graph()

    # Klein bottle 3D immersion parameterization
    def klein_bottle_coords(u, v):
        """Klein bottle parameterization (figure-8 immersion)"""
        # Standard Klein bottle parameterization
        # This creates the characteristic self-intersecting shape
        a = 2.0  # Scale parameter

        # Figure-8 immersion of Klein bottle
        if 0 <= u < np.pi:
            # First half: outer loop
            x = (a + np.cos(u/2) * np.sin(v) - np.sin(u/2) * np.sin(2*v)) * np.cos(u)
            y = (a + np.cos(u/2) * np.sin(v) - np.sin(u/2) * np.sin(2*v)) * np.sin(u)
            z = np.sin(u/2) * np.sin(v) + np.cos(u/2) * np.sin(2*v)
        else:
            # Second half: inner loop that passes through the first half
            x = (a + np.cos(u/2) * np.sin(v) - np.sin(u/2) * np.sin(2*v)) * np.cos(u)
            y = (a + np.cos(u/2) * np.sin(v) - np.sin(u/2) * np.sin(2*v)) * np.sin(u)
            z = np.sin(u/2) * np.sin(v) + np.cos(u/2) * np.sin(2*v)

        return (x, y, z)

    # Create nodes
    node_id = 0
    coord_to_node = {}

    for i in range(u_points):
        for j in range(v_points):
            u = 2 * np.pi * i / u_points
            v = 2 * np.pi * j / v_points

            pos = klein_bottle_coords(u, v)
            G.add_node(node_id, pos=pos, u=u, v=v)
            coord_to_node[(i, j)] = node_id
            node_id += 1

    # Add edges creating mesh topology with Klein bottle wraparound
    for i in range(u_points):
        for j in range(v_points):
            current = coord_to_node[(i, j)]

            # Connect to next in v direction (with wraparound)
            next_v = coord_to_node[(i, (j + 1) % v_points)]
            G.add_edge(current, next_v)

            # Connect to next in u direction
            # Klein bottle topology: when wrapping around u, v direction reverses
            next_i = (i + 1) % u_points
            if i == u_points - 1:  # Wrapping around u
                # Reversed v direction for Klein bottle topology
                next_u = coord_to_node[(next_i, (v_points - 1 - j) % v_points)]
            else:
                next_u = coord_to_node[(next_i, j)]
            G.add_edge(current, next_u)

    return G


def visualize(G):
    """Visualize the Klein bottle graph in 3D"""
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    # Draw edges
    for edge in G.edges():
        x_coords = [pos[edge[0]][0], pos[edge[1]][0]]
        y_coords = [pos[edge[0]][1], pos[edge[1]][1]]
        z_coords = [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(x_coords, y_coords, z_coords, 'b-', alpha=0.3, linewidth=0.5)

    # Draw nodes
    xs = [pos[node][0] for node in G.nodes()]
    ys = [pos[node][1] for node in G.nodes()]
    zs = [pos[node][2] for node in G.nodes()]
    ax.scatter(xs, ys, zs, c='red', s=10, alpha=0.6)

    ax.set_title('Klein Bottle Graph (3D Immersion with Twist)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Set equal aspect ratio
    max_range = 5
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])

    plt.tight_layout()
    plt.show()

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import itertools


def generate(dimensions):
    """
    Generate an n-dimensional torus graph (grid with wraparound edges).

    Args:
        dimensions: List of integers specifying grid size in each dimension
                   e.g., [10, 10] for 2D torus, [8, 8, 8] for 3D torus

    Returns:
        NetworkX graph representing the n-dimensional torus
    """
    G = nx.Graph()
    n_dims = len(dimensions)

    # Generate all node indices as tuples
    node_indices = list(itertools.product(*[range(d) for d in dimensions]))

    # Create nodes with n-dimensional positions
    for idx, coords in enumerate(node_indices):
        # Normalize coordinates to [0, 1] range for each dimension
        normalized_pos = tuple(c / (d - 1) if d > 1 else 0 for c, d in zip(coords, dimensions))
        node_array = np.zeros(dimensions, dtype=np.float32)
        node_array[coords] = 1.0
        G.add_node(idx, pos=normalized_pos, coords=coords, array=node_array)

    # Create a mapping from coordinates to node index
    coord_to_idx = {coords: idx for idx, coords in enumerate(node_indices)}

    # Add edges with wraparound in each dimension
    for idx, coords in enumerate(node_indices):
        for dim in range(n_dims):
            # Create neighbor coordinates by incrementing in this dimension (with wraparound)
            neighbor_coords = list(coords)
            neighbor_coords[dim] = (coords[dim] + 1) % dimensions[dim]
            neighbor_coords = tuple(neighbor_coords)

            neighbor_idx = coord_to_idx[neighbor_coords]
            G.add_edge(idx, neighbor_idx)

    return G


def visualize(G):
    """Visualize the n-dimensional torus graph"""
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')

    if not pos:
        print("No position data available for visualization")
        return

    # Determine dimensionality
    first_pos = next(iter(pos.values()))
    n_dims = len(first_pos)

    if n_dims == 2:
        # 2D torus - visualize as 3D donut (torus embedding)
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Map 2D grid coordinates to 3D torus surface
        # Torus parameters
        R = 3  # Major radius (distance from center of tube to center of torus)
        r = 1  # Minor radius (radius of the tube)

        # Create 3D positions by mapping normalized coords to torus surface
        pos_3d = {}
        for node, p in pos.items():
            # p[0] and p[1] are normalized coordinates in [0, 1]
            u = 2 * np.pi * p[0]  # Angle around the major circle
            v = 2 * np.pi * p[1]  # Angle around the minor circle

            # Torus parametric equations
            x = (R + r * np.cos(v)) * np.cos(u)
            y = (R + r * np.cos(v)) * np.sin(u)
            z = r * np.sin(v)

            pos_3d[node] = (x, y, z)

        # Draw edges
        for edge in G.edges():
            x_coords = [pos_3d[edge[0]][0], pos_3d[edge[1]][0]]
            y_coords = [pos_3d[edge[0]][1], pos_3d[edge[1]][1]]
            z_coords = [pos_3d[edge[0]][2], pos_3d[edge[1]][2]]
            ax.plot(x_coords, y_coords, z_coords, 'b-', alpha=0.5, linewidth=0.5)

        # Draw nodes
        xs = [p[0] for p in pos_3d.values()]
        ys = [p[1] for p in pos_3d.values()]
        zs = [p[2] for p in pos_3d.values()]
        ax.scatter(xs, ys, zs, c='red', s=20, alpha=0.8)

        ax.set_title(f'{n_dims}D Torus Graph (3D Donut Embedding)')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        # Set equal aspect ratio
        max_range = R + r
        ax.set_xlim([-max_range, max_range])
        ax.set_ylim([-max_range, max_range])
        ax.set_zlim([-max_range, max_range])

        plt.tight_layout()
        plt.show()

    elif n_dims == 3:
        # 3D torus - visualize in 3D
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Draw edges
        for edge in G.edges():
            x_coords = [pos[edge[0]][0], pos[edge[1]][0]]
            y_coords = [pos[edge[0]][1], pos[edge[1]][1]]
            z_coords = [pos[edge[0]][2], pos[edge[1]][2]]
            ax.plot(x_coords, y_coords, z_coords, 'b-', alpha=0.4, linewidth=0.5)

        # Draw nodes
        xs = [p[0] for p in pos.values()]
        ys = [p[1] for p in pos.values()]
        zs = [p[2] for p in pos.values()]
        ax.scatter(xs, ys, zs, c='red', s=20, alpha=0.8)

        ax.set_title(f'{n_dims}D Torus Graph')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        plt.tight_layout()
        plt.show()

    else:
        # Higher dimensional - project to first 3 dimensions for visualization
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        # Project to 3D (use first 3 dimensions)
        pos_3d = {node: (p[0], p[1], p[2] if len(p) > 2 else 0) for node, p in pos.items()}

        # Draw edges
        for edge in G.edges():
            x_coords = [pos_3d[edge[0]][0], pos_3d[edge[1]][0]]
            y_coords = [pos_3d[edge[0]][1], pos_3d[edge[1]][1]]
            z_coords = [pos_3d[edge[0]][2], pos_3d[edge[1]][2]]
            ax.plot(x_coords, y_coords, z_coords, 'b-', alpha=0.3, linewidth=0.5)

        # Draw nodes
        xs = [p[0] for p in pos_3d.values()]
        ys = [p[1] for p in pos_3d.values()]
        zs = [p[2] for p in pos_3d.values()]
        ax.scatter(xs, ys, zs, c='red', s=20, alpha=0.8)

        ax.set_title(f'{n_dims}D Torus Graph (projected to 3D)')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_zlabel('Dimension 3')
        plt.tight_layout()
        plt.show()

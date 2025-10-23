import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import hydra
from omegaconf import DictConfig
import os

def generate_sphere_wire_mesh(num_horizontal=20, num_vertical=20):
    """
    Generate a wire mesh sphere graph with horizontal and vertical circles.
    
    Args:
        num_horizontal: Number of horizontal circles (latitude lines)
        num_vertical: Number of vertical circles (longitude lines)
    
    Returns:
        NetworkX graph representing the sphere wire mesh
    """
    G = nx.Graph()
    
    # Generate nodes
    nodes = []
    node_id = 0
    
    # Add north pole
    north_pole = node_id
    nodes.append((0, 0, 1))  # (x, y, z)
    G.add_node(node_id, pos=(0, 0, 1))
    node_id += 1
    
    # Add nodes for horizontal circles (excluding poles)
    for i in range(1, num_horizontal):
        # Latitude angle from north pole
        theta = np.pi * i / num_horizontal
        z = np.cos(theta)
        r = np.sin(theta)  # radius at this height
        
        # Add nodes around this circle
        circle_nodes = []
        for j in range(num_vertical):
            phi = 2 * np.pi * j / num_vertical
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            
            nodes.append((x, y, z))
            G.add_node(node_id, pos=(x, y, z))
            circle_nodes.append(node_id)
            node_id += 1
    
    # Add south pole
    south_pole = node_id
    nodes.append((0, 0, -1))
    G.add_node(node_id, pos=(0, 0, -1))
    node_id += 1
    
    # Add horizontal edges (latitude circles)
    current_node = 1  # Start after north pole
    
    for i in range(1, num_horizontal):
        # Connect nodes in this horizontal circle
        for j in range(num_vertical):
            current = current_node + j
            next_node = current_node + (j + 1) % num_vertical
            G.add_edge(current, next_node)
        current_node += num_vertical
    
    # Add vertical edges (longitude circles)
    for j in range(num_vertical):
        # Connect north pole to first ring
        first_ring_node = 1 + j
        G.add_edge(north_pole, first_ring_node)
        
        # Connect between horizontal rings
        for i in range(1, num_horizontal - 1):
            current = 1 + (i - 1) * num_vertical + j
            next_ring = 1 + i * num_vertical + j
            G.add_edge(current, next_ring)
        
        # Connect last ring to south pole
        last_ring_node = 1 + (num_horizontal - 2) * num_vertical + j
        G.add_edge(last_ring_node, south_pole)
    
    return G

def visualize_sphere_mesh(G):
    """Visualize the sphere wire mesh in 3D"""
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Get node positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Extract coordinates
    xs = [pos[node][0] for node in G.nodes()]
    ys = [pos[node][1] for node in G.nodes()]
    zs = [pos[node][2] for node in G.nodes()]
    
    # Plot edges
    for edge in G.edges():
        x_coords = [pos[edge[0]][0], pos[edge[1]][0]]
        y_coords = [pos[edge[0]][1], pos[edge[1]][1]]
        z_coords = [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(x_coords, y_coords, z_coords, 'b-', alpha=0.6, linewidth=0.5)
    
    # Plot nodes
    ax.scatter(xs, ys, zs, c='red', s=20, alpha=0.8)
    
    # Set equal aspect ratio
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Sphere Wire Mesh Graph')
    
    # Make the plot look more spherical
    max_range = 1.1
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_zlim([-max_range, max_range])
    
    plt.tight_layout()
    plt.show()

def generate_2d_grid(width=20, height=20):
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

def visualize_2d_grid(G):
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

def visualize_nd_torus(G):
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
        # 2D torus - visualize as flat grid
        fig, ax = plt.subplots(figsize=(10, 8))
        pos_2d = {node: (p[0], p[1]) for node, p in pos.items()}

        # Draw edges
        for edge in G.edges():
            x_coords = [pos_2d[edge[0]][0], pos_2d[edge[1]][0]]
            y_coords = [pos_2d[edge[0]][1], pos_2d[edge[1]][1]]
            ax.plot(x_coords, y_coords, 'b-', alpha=0.5, linewidth=0.5)

        # Draw nodes
        xs = [p[0] for p in pos_2d.values()]
        ys = [p[1] for p in pos_2d.values()]
        ax.scatter(xs, ys, c='red', s=20, alpha=0.8)

        ax.set_title(f'{n_dims}D Torus Graph')
        ax.set_xlabel('Dimension 1')
        ax.set_ylabel('Dimension 2')
        ax.set_aspect('equal')
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

def visualize_klein_bottle(G):
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

    ax.set_title('Klein Bottle Graph (Figure-8 Immersion)')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    plt.tight_layout()
    plt.show()

def visualize_random_graph(G, title="Random Graph"):
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

def generate_nd_torus(dimensions):
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
    import itertools
    node_indices = list(itertools.product(*[range(d) for d in dimensions]))

    # Create nodes with n-dimensional positions
    for idx, coords in enumerate(node_indices):
        # Normalize coordinates to [0, 1] range for each dimension
        normalized_pos = tuple(c / (d - 1) if d > 1 else 0 for c, d in zip(coords, dimensions))
        G.add_node(idx, pos=normalized_pos, coords=coords)

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

def generate_klein_bottle(u_points=30, v_points=30):
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

    # Figure-8 immersion parameterization
    def klein_bottle_coords(u, v):
        """Klein bottle parameterization (figure-8 immersion)"""
        r = 4
        if u < np.pi:
            x = 6 * np.cos(u) * (1 + np.sin(u)) + r * np.cos(u) * np.cos(v)
            y = 16 * np.sin(u) + r * np.sin(u) * np.cos(v)
        else:
            x = 6 * np.cos(u) * (1 + np.sin(u)) + r * np.cos(v + np.pi)
            y = 16 * np.sin(u)
        z = r * np.sin(v)
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

def generate_erdos_renyi(n_nodes=100, edge_prob=0.1, seed=42):
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

def generate_barabasi_albert(n_nodes=100, m_edges=3, seed=42):
    """
    Generate a Barabási-Albert scale-free network using preferential attachment.

    New nodes attach to existing nodes with probability proportional to their degree.
    This creates a power-law degree distribution.

    Args:
        n_nodes: Number of nodes in the graph
        m_edges: Number of edges to attach from a new node to existing nodes
        seed: Random seed for reproducibility

    Returns:
        NetworkX graph
    """
    G = nx.barabasi_albert_graph(n_nodes, m_edges, seed=seed)

    # Add positions using spring layout for visualization
    pos = nx.spring_layout(G, seed=seed, dim=3)
    for node, position in pos.items():
        G.nodes[node]['pos'] = tuple(position)

    return G

def generate_watts_strogatz(n_nodes=100, k_neighbors=6, rewire_prob=0.3, seed=42):
    """
    Generate a Watts-Strogatz small-world network.

    Starts with a ring lattice where each node is connected to k nearest neighbors,
    then randomly rewires edges with probability p, creating small-world properties.

    Args:
        n_nodes: Number of nodes in the graph
        k_neighbors: Each node is connected to k nearest neighbors in ring topology
        rewire_prob: Probability of rewiring each edge
        seed: Random seed for reproducibility

    Returns:
        NetworkX graph
    """
    G = nx.watts_strogatz_graph(n_nodes, k_neighbors, rewire_prob, seed=seed)

    # Add positions using circular layout initially, then spring layout
    pos = nx.spring_layout(G, seed=seed, dim=3)
    for node, position in pos.items():
        G.nodes[node]['pos'] = tuple(position)

    return G

def print_graph_stats(G):
    """Print statistics about the generated graph"""
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Average degree: {2 * G.number_of_edges() / G.number_of_nodes():.2f}")
    print(f"Is connected: {nx.is_connected(G)}")

@hydra.main(version_base=None, config_path="../config", config_name="config")
def generate_graph(cfg: DictConfig) -> None:
    """Main function to generate and save graph"""
    
    graph_type = cfg.graph_generation.type
    
    # Generate graph based on type
    if graph_type == "sphere":
        print(f"Generating sphere mesh with {cfg.graph_generation.sphere_mesh.num_horizontal} horizontal and {cfg.graph_generation.sphere_mesh.num_vertical} vertical circles")
        graph = generate_sphere_wire_mesh(
            num_horizontal=cfg.graph_generation.sphere_mesh.num_horizontal,
            num_vertical=cfg.graph_generation.sphere_mesh.num_vertical
        )
        visualize_func = visualize_sphere_mesh
    elif graph_type == "grid":
        print(f"Generating 2D grid with {cfg.graph_generation.grid_2d.width}x{cfg.graph_generation.grid_2d.height} nodes")
        graph = generate_2d_grid(
            width=cfg.graph_generation.grid_2d.width,
            height=cfg.graph_generation.grid_2d.height
        )
        visualize_func = visualize_2d_grid
    elif graph_type == "torus":
        dimensions = cfg.graph_generation.nd_torus.dimensions
        print(f"Generating {len(dimensions)}D torus with dimensions {dimensions}")
        graph = generate_nd_torus(dimensions=dimensions)
        visualize_func = visualize_nd_torus
    elif graph_type == "klein_bottle":
        print(f"Generating Klein bottle with {cfg.graph_generation.klein_bottle.u_points}x{cfg.graph_generation.klein_bottle.v_points} mesh")
        graph = generate_klein_bottle(
            u_points=cfg.graph_generation.klein_bottle.u_points,
            v_points=cfg.graph_generation.klein_bottle.v_points
        )
        visualize_func = visualize_klein_bottle
    elif graph_type == "erdos_renyi":
        print(f"Generating Erdős-Rényi random graph with {cfg.graph_generation.erdos_renyi.n_nodes} nodes and edge probability {cfg.graph_generation.erdos_renyi.edge_prob}")
        graph = generate_erdos_renyi(
            n_nodes=cfg.graph_generation.erdos_renyi.n_nodes,
            edge_prob=cfg.graph_generation.erdos_renyi.edge_prob,
            seed=cfg.graph_generation.erdos_renyi.seed
        )
        visualize_func = lambda g: visualize_random_graph(g, "Erdős-Rényi Random Graph")
    elif graph_type == "barabasi_albert":
        print(f"Generating Barabási-Albert scale-free graph with {cfg.graph_generation.barabasi_albert.n_nodes} nodes and m={cfg.graph_generation.barabasi_albert.m_edges}")
        graph = generate_barabasi_albert(
            n_nodes=cfg.graph_generation.barabasi_albert.n_nodes,
            m_edges=cfg.graph_generation.barabasi_albert.m_edges,
            seed=cfg.graph_generation.barabasi_albert.seed
        )
        visualize_func = lambda g: visualize_random_graph(g, "Barabási-Albert Scale-Free Graph")
    elif graph_type == "watts_strogatz":
        print(f"Generating Watts-Strogatz small-world graph with {cfg.graph_generation.watts_strogatz.n_nodes} nodes, k={cfg.graph_generation.watts_strogatz.k_neighbors}, p={cfg.graph_generation.watts_strogatz.rewire_prob}")
        graph = generate_watts_strogatz(
            n_nodes=cfg.graph_generation.watts_strogatz.n_nodes,
            k_neighbors=cfg.graph_generation.watts_strogatz.k_neighbors,
            rewire_prob=cfg.graph_generation.watts_strogatz.rewire_prob,
            seed=cfg.graph_generation.watts_strogatz.seed
        )
        visualize_func = lambda g: visualize_random_graph(g, "Watts-Strogatz Small-World Graph")
    else:
        raise ValueError(f"Unknown graph type: {graph_type}. Supported types: 'sphere', 'grid', 'torus', 'klein_bottle', 'erdos_renyi', 'barabasi_albert', 'watts_strogatz'")
    
    # Print statistics if requested
    if cfg.graph_generation.output.print_stats:
        print_graph_stats(graph)
    
    # Create output path with suffix
    base_path = cfg.graph_generation.output.file_path
    if base_path.endswith('.pkl'):
        output_path = base_path.replace('.pkl', f'_{graph_type}.pkl')
    else:
        output_path = f"{base_path}_{graph_type}.pkl"
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save graph to pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(graph, f)
    print(f"Graph saved to {output_path}")
    
    # Visualize the graph if requested
    if cfg.graph_generation.output.visualize:
        visualize_func(graph)


if __name__ == "__main__":
    generate_graph()
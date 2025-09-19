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
    else:
        raise ValueError(f"Unknown graph type: {graph_type}. Supported types: 'sphere', 'grid'")
    
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
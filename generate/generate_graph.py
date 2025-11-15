import pickle
import hydra
from omegaconf import DictConfig
import os
import networkx as nx
from graph_types.loader import generate_graph as gen_graph, visualize_graph


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

    # Prepare generation parameters based on graph type
    gen_params = {}

    if graph_type == "sphere":
        print(f"Generating sphere mesh with {cfg.graph_generation.sphere_mesh.num_horizontal} horizontal and {cfg.graph_generation.sphere_mesh.num_vertical} vertical circles")
        gen_params = {
            "num_horizontal": cfg.graph_generation.sphere_mesh.num_horizontal,
            "num_vertical": cfg.graph_generation.sphere_mesh.num_vertical
        }
    elif graph_type == "grid":
        print(f"Generating 2D grid with {cfg.graph_generation.grid_2d.width}x{cfg.graph_generation.grid_2d.height} nodes")
        gen_params = {
            "width": cfg.graph_generation.grid_2d.width,
            "height": cfg.graph_generation.grid_2d.height
        }
    elif graph_type == "torus":
        dimensions = cfg.graph_generation.nd_torus.dimensions
        print(f"Generating {len(dimensions)}D torus with dimensions {dimensions}")
        gen_params = {"dimensions": dimensions}
    elif graph_type == "klein_bottle":
        print(f"Generating Klein bottle with {cfg.graph_generation.klein_bottle.u_points}x{cfg.graph_generation.klein_bottle.v_points} mesh")
        gen_params = {
            "u_points": cfg.graph_generation.klein_bottle.u_points,
            "v_points": cfg.graph_generation.klein_bottle.v_points
        }
    elif graph_type == "erdos_renyi":
        print(f"Generating Erdős-Rényi random graph with {cfg.graph_generation.erdos_renyi.n_nodes} nodes and edge probability {cfg.graph_generation.erdos_renyi.edge_prob}")
        gen_params = {
            "n_nodes": cfg.graph_generation.erdos_renyi.n_nodes,
            "edge_prob": cfg.graph_generation.erdos_renyi.edge_prob,
            "seed": cfg.graph_generation.erdos_renyi.seed
        }
    elif graph_type == "barabasi_albert":
        print(f"Generating Barabási-Albert scale-free graph with {cfg.graph_generation.barabasi_albert.n_nodes} nodes and m={cfg.graph_generation.barabasi_albert.m_edges}")
        gen_params = {
            "n_nodes": cfg.graph_generation.barabasi_albert.n_nodes,
            "m_edges": cfg.graph_generation.barabasi_albert.m_edges,
            "seed": cfg.graph_generation.barabasi_albert.seed
        }
    elif graph_type == "watts_strogatz":
        print(f"Generating Watts-Strogatz small-world graph with {cfg.graph_generation.watts_strogatz.n_nodes} nodes, k={cfg.graph_generation.watts_strogatz.k_neighbors}, p={cfg.graph_generation.watts_strogatz.rewire_prob}")
        gen_params = {
            "n_nodes": cfg.graph_generation.watts_strogatz.n_nodes,
            "k_neighbors": cfg.graph_generation.watts_strogatz.k_neighbors,
            "rewire_prob": cfg.graph_generation.watts_strogatz.rewire_prob,
            "seed": cfg.graph_generation.watts_strogatz.seed
        }
    elif graph_type == "maze":
        print(f"Generating {cfg.graph_generation.maze.width}x{cfg.graph_generation.maze.height} maze using {cfg.graph_generation.maze.algorithm} algorithm")
        gen_params = {
            "width": cfg.graph_generation.maze.width,
            "height": cfg.graph_generation.maze.height,
            "algorithm": cfg.graph_generation.maze.algorithm,
            "seed": cfg.graph_generation.maze.seed
        }
    else:
        # This should be caught by the loader, but kept here for clarity
        raise ValueError(f"Unknown graph type: {graph_type}")

    # Generate the graph using the modular loader
    graph = gen_graph(graph_type, **gen_params)

    # Print statistics
    print_graph_stats(graph)

    # Create output path with suffix
    output_path = cfg.data_generation.output_dir + f"/graph_{graph_type}.pkl"
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Save graph to pickle file
    with open(output_path, 'wb') as f:
        pickle.dump(graph, f)
    print(f"Graph saved to {output_path}")

    # Visualize the graph using the modular loader
    visualize_graph(graph_type, graph)


if __name__ == "__main__":
    generate_graph()

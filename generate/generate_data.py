import pickle
import json
import random
import networkx as nx
import hydra
from omegaconf import DictConfig
import os
from typing import List, Dict, Tuple


def load_graph(graph_path: str) -> nx.Graph:
    """Load graph from pickle file"""
    with open(graph_path, 'rb') as f:
        return pickle.load(f)


def compute_all_pairs_shortest_paths(graph: nx.Graph) -> Dict[Tuple[int, int], List[int]]:
    """
    Compute all pairs shortest paths using NetworkX.
    Returns a dictionary mapping (source, target) -> path
    """
    all_paths = {}
    nodes = list(graph.nodes())

    print(f"Computing shortest paths for {len(nodes)} nodes...")

    # Compute all pairs shortest paths
    for i, source in enumerate(nodes):
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(nodes)} nodes processed")

        # Get shortest paths from source to all other nodes
        paths = nx.single_source_shortest_path(graph, source)

        # Add all paths (except self-loops)
        for target, path in paths.items():
            if source != target:
                all_paths[(source, target)] = path

    return all_paths


def split_paths_train_test(all_paths: Dict[Tuple[int, int], List[int]],
                           train_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
    """
    Split paths into train (80%) and test (20%) sets.
    Returns data in the format: {"input": [start, end], "output": path}
    """
    # Convert to list of (pair, path) tuples
    path_items = list(all_paths.items())

    # Shuffle for random split
    random.shuffle(path_items)

    # Calculate split point
    split_idx = int(len(path_items) * train_ratio)

    # Split into train and test
    train_items = path_items[:split_idx]
    test_items = path_items[split_idx:]

    # Format as required
    train_data = [
        {
            "input": [pair[0], pair[1]],
            "output": path
        }
        for pair, path in train_items
    ]

    test_data = [
        {
            "input": [pair[0], pair[1]],
            "output": path
        }
        for pair, path in test_items
    ]

    return train_data, test_data


@hydra.main(version_base=None, config_path="../config", config_name="config")
def generate_dataset(cfg: DictConfig) -> None:
    """Main function to generate training and test datasets using NetworkX"""

    # Construct the correct graph path with suffix based on graph type
    graph_type = cfg.graph_generation.type
    graph_path = cfg.data_generation.output_dir + f"/graph_{graph_type}.pkl"

    # Load NetworkX graph
    print(f"Loading graph from {graph_path}")
    graph = load_graph(graph_path)
    print(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")

    # Compute all pairs shortest paths
    print("Computing all pairs shortest paths...")
    all_paths = compute_all_pairs_shortest_paths(graph)
    print(f"Computed {len(all_paths)} paths")

    # Split into train (80%) and test (20%)
    print("Splitting into train and test sets...")
    train_ratio = 0.8
    train_data, test_data = split_paths_train_test(all_paths, train_ratio)

    # Save datasets
    train_file = os.path.join(cfg.data_generation.output_dir, f"train_{graph_type}.json")
    test_file = os.path.join(cfg.data_generation.output_dir, f"test_{graph_type}.json")

    print(f"Saving training data to {train_file}")
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"Training data saved: {len(train_data)} examples")

    print(f"Saving test data to {test_file}")
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"Test data saved: {len(test_data)} examples")

    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"Total paths computed: {len(all_paths)}")
    print(f"Training examples: {len(train_data)} ({train_ratio*100}%)")
    print(f"Test examples: {len(test_data)} ({(1-train_ratio)*100}%)")

    if train_data:
        train_lengths = [len(example["output"]) for example in train_data]
        print(f"Training path lengths: min={min(train_lengths)}, max={max(train_lengths)}, avg={sum(train_lengths)/len(train_lengths):.2f}")

    if test_data:
        test_lengths = [len(example["output"]) for example in test_data]
        print(f"Test path lengths: min={min(test_lengths)}, max={max(test_lengths)}, avg={sum(test_lengths)/len(test_lengths):.2f}")


if __name__ == "__main__":
    generate_dataset()

import pickle
import json
import random
import networkx as nx
from typing import List, Dict, Tuple
import hydra
from omegaconf import DictConfig
import os


def load_graph(graph_path: str) -> nx.Graph:
    """Load graph from pickle file"""
    with open(graph_path, 'rb') as f:
        return pickle.load(f)


def random_walk(graph: nx.Graph, start_node: int, length: int) -> List[int]:
    """Perform random walk from start_node for given length"""
    path = [start_node]
    current = start_node
    
    for _ in range(length - 1):
        neighbors = list(graph.neighbors(current))
        if not neighbors:
            break
        
        # Remove previous node from neighbors to avoid going back
        if len(path) >= 2:
            previous = path[-2]
            if previous in neighbors:
                neighbors.remove(previous)
        
        # If no valid neighbors remain, break
        if not neighbors:
            break
            
        current = random.choice(neighbors)
        path.append(current)
    
    return path


def perturb_point(graph: nx.Graph, node: int, perturbation_steps: int) -> int:
    """Perturb a node by moving it random steps away"""
    current = node
    for _ in range(perturbation_steps):
        neighbors = list(graph.neighbors(current))
        if neighbors:
            current = random.choice(neighbors)
        else:
            break
    return current


def generate_perturbed_path(graph: nx.Graph, start_node: int, end_node: int, 
                          n_segments_range: Tuple[int, int], 
                          perturbation_range: Tuple[int, int]) -> List[int]:
    """Generate a perturbed path between start and end nodes"""
    try:
        # Get shortest path
        shortest_path = nx.shortest_path(graph, start_node, end_node)
        if len(shortest_path) < 3:
            return shortest_path
        
        # Sample number of segments
        n_segments = random.randint(*n_segments_range)
        if n_segments >= len(shortest_path):
            n_segments = len(shortest_path) - 1
        
        # Split path into segments
        segment_length = len(shortest_path) // n_segments
        segments = []
        
        for i in range(n_segments):
            start_idx = i * segment_length
            if i == n_segments - 1:
                end_idx = len(shortest_path)
            else:
                end_idx = (i + 1) * segment_length
            segments.append(shortest_path[start_idx:end_idx])
        
        # Perturb segment endpoints (except the last one)
        perturbed_path = [start_node]
        
        for i, segment in enumerate(segments):
            if i < len(segments) - 1:  # Not the last segment
                # Perturb the endpoint
                original_endpoint = segment[-1]
                perturbation_steps = random.randint(*perturbation_range)
                perturbed_endpoint = perturb_point(graph, original_endpoint, perturbation_steps)
                
                # Connect current position to perturbed endpoint
                try:
                    if perturbed_path[-1] != perturbed_endpoint:
                        connection = nx.shortest_path(graph, perturbed_path[-1], perturbed_endpoint)
                        perturbed_path.extend(connection[1:])
                except nx.NetworkXNoPath:
                    # If no path exists, use original segment
                    perturbed_path.extend(segment[1:])
            else:
                # Last segment: connect to end_node
                try:
                    if perturbed_path[-1] != end_node:
                        connection = nx.shortest_path(graph, perturbed_path[-1], end_node)
                        perturbed_path.extend(connection[1:])
                except nx.NetworkXNoPath:
                    # If no path exists, use original segment
                    perturbed_path.extend(segment[1:])
        
        return perturbed_path
        
    except nx.NetworkXNoPath:
        return []


def generate_train_data(graph: nx.Graph, num_paths: int, min_length: int, max_length: int) -> List[Dict]:
    """Generate training data with random walks"""
    train_data = []
    nodes = list(graph.nodes())
    
    for _ in range(num_paths):
        # Random path length
        path_length = random.randint(min_length, max_length)
        
        # Random starting node
        start_node = random.choice(nodes)
        
        # Perform random walk
        path = random_walk(graph, start_node, path_length)
        
        # Create training example
        if len(path) >= 2:
            train_example = {
                "input": [path[0], path[-1]],
                "output": path
            }
            train_data.append(train_example)
    
    return train_data


def sample_train_test_pairs(graph: nx.Graph, num_train: int, num_test: int) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Sample non-overlapping train and test node pairs"""
    nodes = list(graph.nodes())
    all_pairs = []
    
    # Generate all possible pairs with valid paths
    for i, start in enumerate(nodes):
        for j, end in enumerate(nodes):
            if i != j:  # Skip same node pairs
                try:
                    # Check if path exists
                    nx.shortest_path(graph, start, end)
                    all_pairs.append((start, end))
                except nx.NetworkXNoPath:
                    continue
    
    # Shuffle and split
    random.shuffle(all_pairs)
    total_needed = num_train + num_test
    
    if len(all_pairs) < total_needed:
        print(f"Warning: Only {len(all_pairs)} valid pairs available, needed {total_needed}")
        # Adjust numbers proportionally
        ratio = len(all_pairs) / total_needed
        num_train = int(num_train * ratio)
        num_test = len(all_pairs) - num_train
    
    train_pairs = all_pairs[:num_train]
    test_pairs = all_pairs[num_train:num_train + num_test]
    
    return train_pairs, test_pairs


def generate_perturbed_train_data(graph: nx.Graph, train_pairs: List[Tuple[int, int]], 
                                num_paths: int, n_segments_range: Tuple[int, int],
                                perturbation_range: Tuple[int, int]) -> List[Dict]:
    """Generate training data with perturbed shortest paths for pre-sampled pairs"""
    train_data = []
    
    # Calculate how many paths per pair we need
    paths_per_pair = max(1, num_paths // len(train_pairs))
    remaining_paths = num_paths % len(train_pairs)
    
    for i, (start_node, end_node) in enumerate(train_pairs):
        # Some pairs get one extra path to reach exact num_paths
        current_paths = paths_per_pair + (1 if i < remaining_paths else 0)
        
        for _ in range(current_paths):
            # Generate perturbed path for this pair
            perturbed_path = generate_perturbed_path(
                graph, start_node, end_node, n_segments_range, perturbation_range
            )
            
            if len(perturbed_path) >= 2:
                train_example = {
                    "input": [start_node, end_node],
                    "output": perturbed_path
                }
                train_data.append(train_example)
                
                # Stop if we've reached the target number
                if len(train_data) >= num_paths:
                    break
        
        if len(train_data) >= num_paths:
            break
    
    return train_data


def generate_test_data(graph: nx.Graph, test_pairs: List[Tuple[int, int]], num_paths: int) -> List[Dict]:
    """Generate test data with shortest paths for pre-sampled pairs"""
    test_data = []
    
    for start_node, end_node in test_pairs:
        # Find shortest path
        try:
            shortest_path = nx.shortest_path(graph, start_node, end_node)
            
            test_example = {
                "input": [start_node, end_node],
                "output": shortest_path
            }
            test_data.append(test_example)
            
        except nx.NetworkXNoPath:
            # No path exists between these nodes (shouldn't happen with pre-sampled pairs)
            continue
    # Subsample test data if we have more than needed
    if len(test_data) > num_paths:
        test_data = random.sample(test_data, num_paths)
    return test_data


def generate_test_data_legacy(graph: nx.Graph, num_paths: int, train_pairs: set) -> List[Dict]:
    """Generate test data with shortest paths, avoiding train pairs (legacy version)"""
    test_data = []
    nodes = list(graph.nodes())
    attempts = 0
    max_attempts = num_paths * 10  # Prevent infinite loop
    
    while len(test_data) < num_paths and attempts < max_attempts:
        attempts += 1
        
        # Random pair of nodes
        start_node = random.choice(nodes)
        end_node = random.choice(nodes)
        
        # Skip if same node or pair already in training
        if start_node == end_node or (start_node, end_node) in train_pairs or (end_node, start_node) in train_pairs:
            continue
        
        # Find shortest path
        try:
            shortest_path = nx.shortest_path(graph, start_node, end_node)
            
            test_example = {
                "input": [start_node, end_node],
                "output": shortest_path
            }
            test_data.append(test_example)
            
        except nx.NetworkXNoPath:
            # No path exists between these nodes
            continue
    
    return test_data


@hydra.main(version_base=None, config_path="../config", config_name="config")
def generate_dataset(cfg: DictConfig) -> None:
    """Main function to generate training and test datasets"""
    
    # Construct the correct graph path with suffix based on graph type
    base_path = cfg.graph_generation.output.file_path
    graph_type = cfg.graph_generation.type
    
    if base_path.endswith('.pkl'):
        graph_path = base_path.replace('.pkl', f'_{graph_type}.pkl')
    else:
        graph_path = f"{base_path}_{graph_type}.pkl"
    
    # Load graph
    print(f"Loading graph from {graph_path}")
    graph = load_graph(graph_path)
    print(f"Graph loaded: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges")
    
    # Check if perturbed training is enabled
    if hasattr(cfg.data_generation.train, 'use_perturbed') and cfg.data_generation.train.use_perturbed:
        # Sample non-overlapping train/test pairs first
        print("Sampling non-overlapping train/test pairs...")
        # Get all possible pairs and split 80/20
        all_nodes = list(graph.nodes())
        all_pairs = []
        for i, start in enumerate(all_nodes):
            for end in all_nodes[i+1:]:
                # Check if path exists between nodes
                try:
                    nx.shortest_path(graph, start, end)
                    all_pairs.append((start, end))
                except nx.NetworkXNoPath:
                    continue
        
        # Shuffle and split 80/20
        random.shuffle(all_pairs)
        split_idx = int(0.2 * len(all_pairs))
        train_pairs_list = all_pairs[:split_idx]
        test_pairs_list = all_pairs[split_idx:]
        
        print(f"Train pairs: {len(train_pairs_list)}")
        print(f"Test pairs: {len(test_pairs_list)}")
        
        # Generate training data with perturbed paths
        print(f"Generating {cfg.data_generation.train.num_paths} training examples with perturbed paths...")
        n_segments_range = (cfg.data_generation.train.n_segments_min, cfg.data_generation.train.n_segments_max)
        perturbation_range = (cfg.data_generation.train.perturbation_min, cfg.data_generation.train.perturbation_max)
        
        train_data = generate_perturbed_train_data(
            graph,
            train_pairs_list,
            cfg.data_generation.train.num_paths,
            n_segments_range,
            perturbation_range
        )
        
        # Generate test data with shortest paths
        print(f"Generating {len(test_pairs_list)} test examples...")
        test_data = generate_test_data(graph, test_pairs_list, cfg.data_generation.test.num_paths)
        
    else:
        # Original random walk approach
        print(f"Generating {cfg.data_generation.train.num_paths} training examples...")
        train_data = generate_train_data(
            graph,
            cfg.data_generation.train.num_paths,
            cfg.data_generation.train.min_length,
            cfg.data_generation.train.max_length
        )
        
        # Extract training pairs for test set filtering
        train_pairs = set()
        for example in train_data:
            start, end = example["input"]
            train_pairs.add((start, end))
        
        # Generate test data
        print(f"Generating {len(test_pairs_list)} test examples...")
        test_data = generate_test_data_legacy(
            graph,
            cfg.data_generation.test.num_paths,
            train_pairs
        )
    
    # Save datasets
    train_file = os.path.join(cfg.data_generation.output_dir, f"train_{graph_type}.json")
    test_file = os.path.join(cfg.data_generation.output_dir, f"test_{graph_type}.json")
    
    with open(train_file, 'w') as f:
        json.dump(train_data, f, indent=2)
    print(f"Training data saved to {train_file}")
    
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    print(f"Test data saved to {test_file}")
    
    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"Training examples: {len(train_data)}")
    print(f"Test examples: {len(test_data)}")
    
    if train_data:
        train_lengths = [len(example["output"]) for example in train_data]
        print(f"Training path lengths: min={min(train_lengths)}, max={max(train_lengths)}, avg={sum(train_lengths)/len(train_lengths):.2f}")
    
    if test_data:
        test_lengths = [len(example["output"]) for example in test_data]
        print(f"Test path lengths: min={min(test_lengths)}, max={max(test_lengths)}, avg={sum(test_lengths)/len(test_lengths):.2f}")


if __name__ == "__main__":
    generate_dataset()
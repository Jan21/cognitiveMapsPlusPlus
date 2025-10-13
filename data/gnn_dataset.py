import json
import torch
import pickle
from torch_geometric.data import Data, Dataset
from torch_sparse import SparseTensor
from typing import List


class PathInstance(Data):
    """
    Data object for a single path instance in the bipartite graph.

    New structure for middle-node prediction:
    - 3 vertex nodes: start, middle (to predict), end
    - 1 constraint node: connects all 3 vertices
    """

    def __init__(
        self,
        x_v=None,
        x_e=None,
        targets=None,
        target_mask=None,
        num_vertices=None,
        num_edges=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.x_v = x_v
        self.x_e = x_e
        self.targets = targets
        self.target_mask = target_mask
        self.num_vertices = num_vertices
        self.num_edges = num_edges

        # Create adjacency matrix if we have the necessary info
        if num_vertices is not None and num_edges is not None:
            self._create_adjacency()

    def _create_adjacency(self):
        """
        Create sparse adjacency matrix [num_edges, num_vertices].

        For the new bipartite structure:
        - num_vertices = 3 (start, middle, end)
        - num_edges = 1 (constraint node)
        - The constraint node connects to all 3 vertices
        """
        row_indices = []
        col_indices = []
        values = []

        # Single constraint edge connects to all 3 vertices
        for vertex_idx in range(self.num_vertices):
            row_indices.append(0)  # Only one edge (constraint node)
            col_indices.append(vertex_idx)
            values.append(1.0)

        self.adj_t = SparseTensor(
            row=torch.tensor(row_indices, dtype=torch.long),
            col=torch.tensor(col_indices, dtype=torch.long),
            value=torch.tensor(values, dtype=torch.float),
            sparse_sizes=(self.num_edges, self.num_vertices)
        )

    def __inc__(self, key, value, *args, **kwargs):
        """
        Define how to increment indices when batching.
        Similar to SATInstance.__inc__.
        """
        if key == 'adj_t':
            return torch.tensor([
                [self.num_edges],
                [self.num_vertices]
            ])
        else:
            return super().__inc__(key, value, *args, **kwargs)


class BipartitePathDataset(Dataset):
    """
    Dataset for GNN-based middle-node prediction.

    Creates bipartite graphs where:
    - Vertex nodes: [start, middle, end] (3 nodes)
    - Constraint node: connects all 3 vertices (1 node)
    - Task: Predict the middle node given start and end

    Follows the pattern from cnf_data.py for proper PyTorch Geometric integration.
    """

    def __init__(
        self,
        json_file: str,
        graph_file: str,
        max_path_length: int = 64,
        vocab_size: int = 10000
    ):
        super().__init__()

        self.max_path_length = max_path_length
        self.vocab_size = vocab_size

        # Load the graph structure
        with open(graph_file, 'rb') as f:
            self.graph = pickle.load(f)

        # Load path data
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.paths = self._extract_paths()

    def _extract_paths(self) -> List[List[int]]:
        """Extract valid paths from the data. Need at least 3 nodes for middle-node prediction."""
        paths = []
        for item in self.data:
            if 'output' in item and isinstance(item['output'], list):
                path = item['output']
                # Need at least 3 nodes: start, middle, end
                if 3 <= len(path) <= self.max_path_length:
                    paths.append(path)
        return paths

    def len(self) -> int:
        return len(self.paths)

    def get(self, idx: int) -> PathInstance:
        """
        Get a single path instance as a bipartite graph.

        New format: Predict middle node given start and end nodes.
        - Vertices: [start, middle, end] (3 nodes)
        - Edges: [constraint] (1 node connecting all 3)

        Returns:
            PathInstance with:
                - x_v: vertex features [3, 1] - IDs of start, middle, end
                - x_e: edge features [1, 1] - single constraint node
                - adj_t: sparse adjacency matrix [1, 3] - constraint connects to all vertices
                - targets: ground truth vertex IDs [3]
                - target_mask: which vertices to predict [3] - only middle is True
        """
        path = self.paths[idx]

        # We need at least 3 nodes (start, middle, end)
        if len(path) < 3:
            # For paths with only 2 nodes, we can't predict a middle
            # Skip these or handle differently
            raise ValueError(f"Path {idx} has only {len(path)} nodes, need at least 3")

        # Extract start, middle, and end nodes
        # For paths longer than 3, we pick a random middle node
        start_node = path[0]
        end_node = path[-1]

        if len(path) == 3:
            middle_node = path[1]
        else:
            # Pick the (rounded down) middle intermediate node as the middle
            middle_idx = (len(path) - 1) // 2
            if middle_idx == 0 or middle_idx == len(path) - 1:
                raise ValueError(f"Cannot pick a true middle node for path of length {len(path)}")
            middle_node = path[middle_idx]

        # Create vertex features: 3 vertices [start, middle, end]
        # During training, middle will be masked
        num_vertices = 3
        x_v = torch.tensor([start_node, middle_node, end_node], dtype=torch.long).unsqueeze(1)

        # Create edge features: 1 constraint node (can be a dummy feature)
        num_edges = 1
        x_e = torch.zeros((num_edges, 1), dtype=torch.long)

        # Target mask: only predict the middle vertex (index 1)
        target_mask = torch.zeros(num_vertices, dtype=torch.bool)
        target_mask[1] = True  # Only middle node

        # Ground truth targets
        targets = torch.tensor([start_node, middle_node, end_node], dtype=torch.long)

        return PathInstance(
            x_v=x_v,
            x_e=x_e,
            targets=targets,
            target_mask=target_mask,
            num_vertices=num_vertices,
            num_edges=num_edges
        )

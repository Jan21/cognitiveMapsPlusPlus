import json
import torch
import pickle
from torch_geometric.data import Data, Dataset
from torch_sparse import SparseTensor
from typing import List


class PathInstance(Data):
    """
    Data object for a single path instance in the bipartite graph.

    Similar to SATInstance - handles adjacency matrix creation in __init__.
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

        Each edge connects to exactly 2 consecutive vertices.
        Edge i connects to vertices i and i+1.
        """
        row_indices = []
        col_indices = []
        values = []

        for edge_idx in range(self.num_edges):
            # Edge connects to two consecutive vertices
            row_indices.extend([edge_idx, edge_idx])
            col_indices.extend([edge_idx, edge_idx + 1])
            values.extend([1.0, 1.0])

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
    Dataset for GNN-based shortest path prediction.

    Creates bipartite graphs where:
    - Vertex nodes represent waypoints in the path
    - Edge nodes represent connections between consecutive vertices
    - First and last vertex values are given (boundary conditions)
    - Intermediate vertex values need to be predicted

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
        """Extract valid paths from the data."""
        paths = []
        for item in self.data:
            if 'output' in item and isinstance(item['output'], list):
                path = item['output']
                if 2 <= len(path) <= self.max_path_length:
                    paths.append(path)
        return paths

    def len(self) -> int:
        return len(self.paths)

    def get(self, idx: int) -> PathInstance:
        """
        Get a single path instance as a bipartite graph.

        Returns:
            PathInstance with:
                - x_v: vertex features [num_vertices, 1]
                - x_e: edge features [num_edges, 2]
                - adj_t: sparse adjacency matrix [num_edges, num_vertices]
                - targets: ground truth vertex IDs [num_vertices]
                - target_mask: which vertices to predict [num_vertices]
        """
        path = self.paths[idx]
        num_vertices = len(path)
        num_edges = num_vertices - 1

        # Vertex features: all set to actual node IDs
        # (masking happens in the Lightning module during training)
        x_v = torch.tensor(path, dtype=torch.long).unsqueeze(1)

        # Edge features: store the two vertices each edge connects
        x_e = torch.zeros((num_edges, 2), dtype=torch.long)
        # for i in range(num_edges):
        #     x_e[i, 0] = path[i]
        #     x_e[i, 1] = path[i + 1]

        # Target mask: predict all intermediate vertices (not first/last)
        target_mask = torch.zeros(num_vertices, dtype=torch.bool)
        if num_vertices > 2:
            target_mask[1:-1] = True

        # Ground truth targets
        targets = torch.tensor(path, dtype=torch.long)

        return PathInstance(
            x_v=x_v,
            x_e=x_e,
            targets=targets,
            target_mask=target_mask,
            num_vertices=num_vertices,
            num_edges=num_edges
        )

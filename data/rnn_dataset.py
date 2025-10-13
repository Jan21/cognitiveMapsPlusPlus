import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any
import pickle


class RNNMiddleNodeDataset(Dataset):
    """
    Dataset for RNN-based middle-node prediction.

    Returns:
        - start_id: Start node ID
        - end_id: End node ID
        - middle_id: Middle node ID (target)

    The RNN will concatenate embeddings of start and end,
    then run recurrent updates to predict the middle node.
    """

    def __init__(
        self,
        json_file: str,
        graph_file: str,
        max_path_length: int = 64,
        vocab_size: int = 10000
    ):
        self.max_path_length = max_path_length
        self.vocab_size = vocab_size

        # Load the graph structure (optional, for validation)
        with open(graph_file, 'rb') as f:
            self.graph = pickle.load(f)

        # Load path data
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.paths = self._extract_paths()

    def _extract_paths(self) -> List[List[int]]:
        """Extract valid paths from the data. Need at least 3 nodes."""
        paths = []
        for item in self.data:
            if 'output' in item and isinstance(item['output'], list):
                path = item['output']
                # Need at least 3 nodes: start, middle, end
                if 3 <= len(path) <= self.max_path_length:
                    paths.append(path)
        return paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single training example.

        Returns:
            Dictionary with:
                - start_id: Start node ID [1]
                - end_id: End node ID [1]
                - middle_id: Middle node ID (target) [1]
        """
        path = self.paths[idx]

        # Extract start, middle, and end nodes
        start_node = path[0]
        end_node = path[-1]

        # For paths longer than 3, pick a random intermediate node as middle
        if len(path) == 3:
            middle_node = path[1]
        else:
            # Pick the middle intermediate node as the middle
            middle_idx = (len(path) - 1) // 2
            middle_node = path[middle_idx]

        return {
            'start_id': torch.tensor(start_node, dtype=torch.long),
            'end_id': torch.tensor(end_node, dtype=torch.long),
            'middle_id': torch.tensor(middle_node, dtype=torch.long),
        }

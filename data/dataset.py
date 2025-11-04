import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any, Optional
import os
import numpy as np
import random
import gc
import networkx as nx


class PathDataset(Dataset):
    def __init__(self, json_file: str, max_path_length: int = 64, vocab_size: int = 10000, percentage_of_samples: float = 1.0, num_samples: int = None, **kwargs):
        self.max_path_length = max_path_length
        self.vocab_size = vocab_size
        self.eos_token = vocab_size - 1


        with open(json_file, 'r') as f:
            self.data = json.load(f)

        if num_samples is None:
            self.num_samples = int(percentage_of_samples * len(self.data))
        else:
            self.num_samples = num_samples
        self.paths = self._extract_paths()
        del self.data
        gc.collect()

    def _extract_paths(self) -> List[List[int]]:
        paths = []
        for item in self.data:
            if 'output' in item and isinstance(item['output'], list):
                path = item['output']
                if len(path) <= self.max_path_length:
                    paths.append(path)

        # Limit number of samples if specified
        random.shuffle(paths)
        if self.num_samples is not None and self.num_samples > 0:
            paths = paths[:self.num_samples]

        return paths


    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.paths[idx]
    
        path = [path[-1]] + path + [self.eos_token] # add eos token to the end
        # Create input (all tokens except last) and target (all tokens except first)
        input_ids = torch.tensor(path, dtype=torch.long)
        target_ids = torch.tensor(path[1:]+[-100], dtype=torch.long)
        target_ids[0] = -100 # mask the first two tokens which are the pair of points
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
        }


class SpreadPathDataset(Dataset):
    """
    Dataset that spreads paths evenly into a fixed-length tensor.

    The path elements are distributed evenly across the tensor:
    - First element at position 0
    - Last element at position -1
    - Middle elements spread evenly in between
    - Empty positions filled with vocab_size - 1
    """


    def __init__(self, json_file: str, max_path_length: int = 33, vocab_size: int = 10000, percentage_of_samples: float = 1.0, num_samples: int = None, **kwargs):
        self.tensor_length = kwargs.get('tensor_length', max_path_length - 1)
        self.max_path_length = max_path_length
        self.vocab_size = vocab_size
        self.pad_token = vocab_size - 1
        


        with open(json_file, 'r') as f:
            self.data = json.load(f)
        if num_samples is None:
            self.num_samples = int(percentage_of_samples * len(self.data))
        else:
            self.num_samples = num_samples
        self.paths = self._extract_paths()
        del self.data
        gc.collect()
        self.max_tensor_length = kwargs.get('tensor_length', max_path_length - 1)
        self.position_dicts = {}
        self.indices_per_level = {}
        self.create_position_dicts()

    def create_position_dicts(self):

        def generate_sequence(min_val, max_val):
            """Generate sequence of form 2^n + 1 between min_val and max_val"""
            sequence = []
            n = 1
            while True:
                term = 2**n + 1
                if term > max_val:
                    break
                if term >= min_val:
                    sequence.append(term)
                n += 1
            return sequence

        # Usage:
        num_els_per_level = generate_sequence(3, 33)
        for level, num_els in enumerate(num_els_per_level):
            indices = torch.linspace(0, self.max_path_length-1, steps=num_els).round().long()
            self.indices_per_level[level] = indices

        # recursively divides the tensor and chooses the middle point of each segment as a position for a given token
        def get_midpoints(path, tensor, increment=0):
            if len(path) == 0:
                return []
            mid_tensor = (len(tensor)) // 2
            mid_path = (len(path)) // 2
            left_tensor = tensor[:mid_tensor]
            right_tensor = tensor[mid_tensor+1:]
            left_path = path[:mid_path]
            right_path = path[mid_path+1:]
            left_midpoints = []
            if len(left_path) > 0:
                left_midpoints = get_midpoints(left_path, left_tensor)
            else:
                left_midpoints = []
            right_midpoints = []
            if len(right_path) > 0:
                right_midpoints = get_midpoints(right_path, right_tensor)
            else:
                right_midpoints = []
            return left_midpoints + [int(tensor[mid_tensor])] + right_midpoints

        for path_len in range(2,34):   
            tensor = torch.tensor(list(range(1,self.max_tensor_length)))
            midpoints = [0] + get_midpoints(list(range(path_len-2)), tensor, 0) + [self.max_tensor_length]
            self.position_dicts[path_len] = midpoints



    def _extract_paths(self) -> List[List[int]]:
        paths = []
        for item in self.data:
            if 'output' in item and isinstance(item['output'], list):
                path = item['output']
                if len(path) <= self.max_path_length:
                    paths.append(path)

        # Limit number of samples if specified
        random.shuffle(paths)
        if self.num_samples is not None and self.num_samples > 0:
            paths = paths[:self.num_samples]

        return paths

    def _spread_path_evenly(self, path: List[int]) -> torch.Tensor:
        """
        Spread path elements evenly into a tensor of length self.tensor_length.

        Args:
            path: List of node IDs in the path

        Returns:
            Tensor of shape (self.tensor_length,) with path spread evenly,
            remaining positions filled with self.pad_token
        """
        # Initialize tensor with pad tokens
        tensor = torch.full((self.max_tensor_length+1,), self.pad_token, dtype=torch.long)

        path_len = len(path)

        if path_len == 0:
            return tensor

        if path_len == 1:
            # Single element goes to first position
            tensor[0] = path[0]
            return tensor

        # Calculate evenly spaced indices in the tensor for path elements
        # We want to place path elements at indices that are evenly distributed
        
        indices = self.position_dicts[path_len]
        tensor[indices] = torch.tensor(path, dtype=torch.long)
        next_non_pad_token_idx = 1
        # Place path elements at calculated indices
        n = tensor.shape[0]
        for i in range(n):
            if tensor[i] == path[next_non_pad_token_idx]:
                next_non_pad_token_idx += 1
                continue
            if tensor[i] == self.pad_token:
                tensor[i] = path[next_non_pad_token_idx]
        return tensor

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.paths[idx]

        # Spread the path evenly into the tensor
        ret_dic = {}
        spread_tensor = self._spread_path_evenly(path)
        for level, indices in self.indices_per_level.items():
            labels = spread_tensor[indices].clone()
            labels[0] = -100
            labels[-1] = -100
            ret_dic[level] = labels
        ret_dic['input_ids'] = spread_tensor
        ret_dic['target_ids'] = labels
        ret_dic['len'] = len(path)
        return ret_dic


def collate_fn(pad_token: int, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function that pads each sequence with its own last token.
    Each sequence in the batch is padded with its own last token rather than a fixed pad token.
    """
    # Find the maximum sequence length in this batch
    max_len = max(len(item['input_ids']) for item in batch)

    # Pad all sequences to max_len
    input_ids = []
    target_ids = []

    for item in batch:
        seq_len = len(item['input_ids'])
        pad_len = max_len - seq_len

        # Get the last token of this specific sequence
        last_token = item['input_ids'][-1]

        # Pad input_ids with the last token, target_ids with -100
        padded_input = torch.cat([item['input_ids'], torch.full((pad_len,), pad_token, dtype=torch.long)])
        padded_target = torch.cat([item['target_ids'], torch.full((pad_len,), -100, dtype=torch.long)])

        input_ids.append(padded_input)
        target_ids.append(padded_target)

    return {
        'input_ids': torch.stack(input_ids),
        'target_ids': torch.stack(target_ids),
    }


class QuasimetricEmbeddingsDataset(Dataset):
    """
    Dataset that contains all pairs of nodes in the graph which share an edge.
    Each example is a dict with keys: x, y, action.
    - x: vertex on one side of the edge
    - y: vertex on the other side of the edge
    - action: None (for all examples)
    """

    def __init__(self, graph: nx.Graph, **kwargs):
        self.graph = graph
        self.edge_pairs = self._extract_edge_pairs()

    def _extract_edge_pairs(self) -> List[tuple]:
        """Extract all edge pairs from the graph."""
        edge_pairs = []
        for u, v in self.graph.edges():
            edge_pairs.append((torch.tensor(u, dtype=torch.long), torch.tensor(v, dtype=torch.long)))
        random.shuffle(edge_pairs)
        return edge_pairs

    def __len__(self) -> int:
        return len(self.edge_pairs)

    def __getitem__(self, idx: int) -> Dict[str, Optional[Any]]:
        x, y = self.edge_pairs[idx]
        return {
            'x': x,
            'y': y,
            'action': torch.tensor(0, dtype=torch.long)
        }



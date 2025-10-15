import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any
import os
import numpy as np


class PathDataset(Dataset):
    def __init__(self, json_file: str, max_path_length: int = 64, vocab_size: int = 10000):
        self.max_path_length = max_path_length
        self.vocab_size = vocab_size
        self.eos_token = vocab_size - 1
        
        with open(json_file, 'r') as f:
            self.data = json.load(f)
        
        self.paths = self._extract_paths()
    
    def _extract_paths(self) -> List[List[int]]:
        paths = []
        for item in self.data:
            if 'output' in item and isinstance(item['output'], list):
                path = item['output']
                if len(path) <= self.max_path_length:
                    paths.append(path)
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

class DiffusionPathDataset(Dataset):
    def __init__(self, json_file: str, max_path_length: int = 64, vocab_size: int = 10000):
        self.max_path_length = max_path_length
        self.vocab_size = vocab_size

        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.paths = self._extract_paths()

    def _extract_paths(self) -> List[List[int]]:
        paths = []
        for item in self.data:
            if 'output' in item and isinstance(item['output'], list):
                path = item['output']
                if len(path) <= self.max_path_length:
                    paths.append(path)
        return paths


    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.paths[idx]

        # Create input (all tokens except last) and target (all tokens except first)
        input_ids = torch.tensor(path, dtype=torch.long)
        target_ids = input_ids


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

    
    def __init__(self, json_file: str, tensor_length: int = 32, max_path_length: int = 33, vocab_size: int = 10000):
        self.tensor_length = tensor_length
        self.max_path_length = max_path_length
        self.vocab_size = vocab_size
        self.pad_token = vocab_size - 1

        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.paths = self._extract_paths()
        self.max_tensor_length = tensor_length
        self.position_dicts = {}
        self.create_position_dicts()

    def create_position_dicts(self):

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

        # Place path elements at calculated indices
        for path_idx, tensor_idx in enumerate(indices):
            tensor[tensor_idx] = path[path_idx]

        return tensor

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.paths[idx]

        # Spread the path evenly into the tensor
        spread_tensor = self._spread_path_evenly(path)

        return spread_tensor



class CurriculumPathDataset(PathDataset):
    def __init__(self, json_file: str, target_length: int, max_path_length: int = 64, vocab_size: int = 10000):
        super().__init__(json_file, max_path_length, vocab_size)
        self.target_length = target_length
        self.filtered_paths = self._filter_paths_by_length()
    
    def _filter_paths_by_length(self) -> List[List[int]]:
        return [path for path in self.paths if len(path)  <= self.target_length]
    
    def __len__(self) -> int:
        return len(self.filtered_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.filtered_paths[idx]
        
        path = [path[-1]] + path + [self.eos_token]
        input_ids = torch.tensor(path[:-1], dtype=torch.long)
        target_ids = torch.tensor(path[1:], dtype=torch.long)
        target_ids[0] = -100
        
        return {
            'input_ids': input_ids,
            'target_ids': target_ids,
        }
    

def collate_fn_old(pad_token: int, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function that pads sequences to the longest sequence in the batch.
    """
    # Find the maximum sequence length in this batch
    max_len = max(len(item['input_ids']) for item in batch)
    
    # Pad all sequences to max_len
    input_ids = []
    target_ids = []
    
    for item in batch:
        seq_len = len(item['input_ids'])
        pad_len = max_len - seq_len
        
        # Pad input_ids and target_ids with zeros
        padded_input = torch.cat([item['input_ids'], torch.full((pad_len,), pad_token, dtype=torch.long)])
        padded_target = torch.cat([item['target_ids'], torch.full((pad_len,), -100, dtype=torch.long)])
        
        input_ids.append(padded_input)
        target_ids.append(padded_target)

    
    return {
        'input_ids': torch.stack(input_ids),
        'target_ids': torch.stack(target_ids),
    }


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
        padded_input = torch.cat([item['input_ids'], torch.full((pad_len,), last_token, dtype=torch.long)])
        padded_target = torch.cat([item['target_ids'], torch.full((pad_len,), -100, dtype=torch.long)])

        input_ids.append(padded_input)
        target_ids.append(padded_target)

    return {
        'input_ids': torch.stack(input_ids),
        'target_ids': torch.stack(target_ids),
    }


def create_dataset(file_path: str, max_path_length: int = 64, vocab_size: int = 10000) -> PathDataset:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    return DiffusionPathDataset(file_path, max_path_length, vocab_size)


def create_curriculum_dataset(file_path: str, target_length: int, max_path_length: int = 64, vocab_size: int = 10000) -> CurriculumPathDataset:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file not found: {file_path}")
    
    return CurriculumPathDataset(file_path, target_length, max_path_length, vocab_size)
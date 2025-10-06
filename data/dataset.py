import json
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Any
import os


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
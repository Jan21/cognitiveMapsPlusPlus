import torch
import torch.nn.functional as F
from typing import Dict, Any
import networkx as nx


class NonGenerativeMetrics:
    """Class to compute various metrics for path prediction tasks"""
    
    def __init__(self, graph: nx.Graph, vocab_size: int, enabled_metrics: list = None):
        self.graph = graph
        self.vocab_size = vocab_size
        self.eos_token = vocab_size - 1
        
        # Default to all metrics if none specified
        if enabled_metrics is None:
            enabled_metrics = ['accuracy']
        self.enabled_metrics = enabled_metrics
        
        # Map metric names to computation methods
        self.metric_methods = {
            'accuracy': self._compute_accuracy,
            'exact_match_accuracy': self._compute_exact_match_accuracy,
            'path_validity': self._compute_path_validity,
            'edge_accuracy': self._compute_edge_accuracy,
            'path_optimality': self._compute_path_optimality
        }
    
    def compute_metrics(self, logits: torch.Tensor, targets: torch.Tensor, input_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute enabled metrics for predictions vs targets"""
        predictions = torch.argmax(logits, dim=-1)
        mask = (targets != -100)
        
        results = {}
        for metric_name in self.enabled_metrics:
            if metric_name in self.metric_methods:
                if metric_name == 'accuracy':
                    results[metric_name] = self.metric_methods[metric_name](predictions, targets, mask)
                elif metric_name == 'exact_match_accuracy':
                    results[metric_name] = self.metric_methods[metric_name](predictions, targets, mask)
                elif metric_name in ['path_validity', 'path_optimality']:
                    results[metric_name] = self.metric_methods[metric_name](predictions, targets, input_ids)
                else:
                    results[metric_name] = self.metric_methods[metric_name](predictions, targets)
        
        return results
    
    def _compute_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute token-level accuracy"""
        correct = (predictions == targets) & mask
        total = mask.sum()
        return correct.sum().float() / total.float() if total > 0 else torch.tensor(0.0)
    
    def _compute_exact_match_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute exact match accuracy (whole sequence correct)"""
        batch_size = predictions.size(0)
        exact_matches = 0
        
        for i in range(batch_size):
            # Only consider non-padded tokens for this example
            example_mask = mask[i]
            if example_mask.sum() > 0:  # Only if there are non-padded tokens
                example_correct = (predictions[i] == targets[i]) & example_mask
                if example_correct.sum() == example_mask.sum():
                    exact_matches += 1
        
        return torch.tensor(exact_matches / batch_size, dtype=torch.float32, device=predictions.device)
    
    def _compute_path_validity(self, predictions: torch.Tensor, targets: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Check if predicted paths is a valid path from start to goal"""
        batch_size = predictions.size(0)
        valid_paths = 0
        
        for i in range(batch_size):
            # Only consider non-padded tokens (where targets != -100)
            pred_path = predictions[i][1:]
            start_node = input_ids[i][1].item()
            goal_node = input_ids[i][0].item()
            
            # Convert to list
            pred_list = [start_node] + pred_path.cpu().tolist()
            
            
            # Split predicted path at eos token
            if self.eos_token in pred_list:
                eos_idx = pred_list.index(self.eos_token)
                path_list = pred_list[:eos_idx]
            else:
                path_list = pred_list
            
            
            if len(path_list) <= 1:
                # Single node or empty path - valid if it matches goal
                if len(path_list) == 1 and goal_node is not None:
                    if path_list[0] == goal_node:
                        valid_paths += 1
                elif len(path_list) == 0:
                    valid_paths += 1  # Empty path is considered valid
                continue
            
            # Check if consecutive nodes are connected
            is_valid = True
            for j in range(len(path_list) - 1):
                node1, node2 = path_list[j], path_list[j + 1]
                if not self.graph.has_edge(node1, node2):
                    is_valid = False
                    break
            
            # Check if path ends at the goal node
            if is_valid and goal_node is not None:
                if path_list[-1] != goal_node:
                    is_valid = False
            
            if is_valid:
                valid_paths += 1
        
        return torch.tensor(valid_paths / batch_size, dtype=torch.float32, device=predictions.device)
    
    def _compute_edge_accuracy(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute the proportion of predicted edges that are valid (exist in the graph)"""
        batch_size = predictions.size(0)
        total_valid_edges = 0
        total_edges = 0
        
        for i in range(batch_size):
            # Only consider non-padded tokens (where targets != -100)
            mask = targets[i] != -100
            pred_path = predictions[i][mask]
            
            # Convert to list and remove padding tokens
            pred_list = [node for node in pred_path.cpu().tolist()][:-1]  # remove the eos token
            
            # Skip if path has less than 2 nodes (no edges)
            if len(pred_list) < 2:
                continue
            
            # Extract edges from predicted path
            pred_edges = [(pred_list[j], pred_list[j+1]) for j in range(len(pred_list)-1)]
            
            # Count valid edges (edges that exist in the graph)
            valid_edges = 0
            for edge in pred_edges:
                node1, node2 = edge
                if self.graph.has_edge(node1, node2):
                    valid_edges += 1
            
            total_valid_edges += valid_edges
            total_edges += len(pred_edges)
        
        if total_edges == 0:
            return torch.tensor(0.0, dtype=torch.float32, device=predictions.device)
        
        return torch.tensor(total_valid_edges / total_edges, dtype=torch.float32, device=predictions.device)
    
    def _compute_path_optimality(self, predictions: torch.Tensor, targets: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute path optimality as abs(optimal_length - predicted_length)"""
        batch_size = predictions.size(0)
        optimality_scores = []
        
        for i in range(batch_size):
            # Get mask from targets and compute optimal path length
            mask = targets[i] != -100
            optimal_length = mask.sum().item()
            
            # Find EOS token in predictions and compute predicted length
            pred_sequence = predictions[i].cpu().tolist()
            if self.eos_token in pred_sequence:
                eos_idx = pred_sequence.index(self.eos_token)
                predicted_length = eos_idx
            else:
                predicted_length = len(pred_sequence)
            
            # Compute abs(optimal - predicted)
            optimality_score = abs(optimal_length - predicted_length)
            optimality_scores.append(optimality_score)
        
        return torch.tensor(sum(optimality_scores) / len(optimality_scores), dtype=torch.float32, device=predictions.device)


class GenerativeMetrics:
    """Class to handle generative evaluation metrics"""
    
    def __init__(self, graph: nx.Graph, vocab_size: int, enabled_metrics: list = None):
        self.graph = graph
        self.vocab_size = vocab_size
        self.eos_token = vocab_size - 1
        

    
    def validate_generated_path(self, goal_state: int, start_state: int, generated_path: list) -> bool:
        """Validate if generated path is a valid path from start to goal"""
        # Full path is: start_state + generated_path
        full_path = [start_state] + generated_path
        
        # If no generated tokens, check if start equals goal
        if not generated_path:
            return start_state == goal_state
        
        # Check if path ends at goal
        if full_path[-1] != goal_state:
            return False
        
        # Check if all consecutive edges exist in graph
        for i in range(len(full_path) - 1):
            if not self.graph.has_edge(full_path[i], full_path[i + 1]):
                return False
        
        return True
    
    def evaluate_generative(self, model, input_ids: torch.Tensor, num_samples: int = 1, 
                          max_length: int = 64, temperature: float = 1.0) -> Dict[str, float]:
        """Evaluate model in generative mode on a batch"""
        batch_size = input_ids.size(0)
        
        total_valid_paths = 0
        total_correct_goals = 0
        total_samples = 0
        length_differences = []
        
        for i in range(batch_size):
            # Extract goal and start state from input
            goal_state = input_ids[i, 0].item()
            start_state = input_ids[i, 1].item()
            
            for _ in range(num_samples):
                # Generate path using model's generate_path method
                generated_path = model.generate_path(goal_state, start_state, max_length, temperature)
                
                # Validate path
                is_valid = self.validate_generated_path(goal_state, start_state, generated_path)
                
                if is_valid:
                    total_valid_paths += 1
                    # For valid paths, compute difference from optimal path
                    if generated_path:
                        full_path = [start_state] + generated_path
                        try:
                            optimal_path = nx.shortest_path(self.graph, start_state, goal_state)
                            optimal_length = len(optimal_path) - 1  # Number of edges
                            generated_length = len(generated_path)  # Number of generated tokens (edges)
                            
                            # Store length difference for aggregation
                            length_differences.append(abs(generated_length - optimal_length)/optimal_length)
                        except nx.NetworkXNoPath:
                            # No optimal path exists, skip this metric
                            pass
                
                # Check if path ends at correct goal
                if generated_path and generated_path[-1] == goal_state:
                    total_correct_goals += 1
                elif not generated_path and start_state == goal_state:
                    total_correct_goals += 1
                
                total_samples += 1
        
        # Only compute and return enabled metrics
        results = {}

        results['gen_path_validity'] = total_valid_paths / total_samples if total_samples > 0 else 0.0
        results['gen_goal_accuracy'] = total_correct_goals / total_samples if total_samples > 0 else 0.0
        results['gen_avg_path_length_diff'] = sum(length_differences) / len(length_differences) if length_differences else 0.0
        
        return results
import torch
import torch.nn.functional as F
from typing import Dict, Any
import networkx as nx


class NonGenerativeMetrics:
    """Class to compute various metrics for path prediction tasks"""
    
    def __init__(self, graph: nx.Graph, vocab_size: int):
        self.graph = graph
        self.vocab_size = vocab_size
        self.eos_token = vocab_size - 1
    
    def compute_metrics(self, logits: torch.Tensor, targets: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute all metrics for predictions vs targets"""
        predictions = torch.argmax(logits, dim=-1)
        
        # Only compute accuracy on non-padded tokens (targets != -100)
        mask = (targets != -100)
        correct = (predictions == targets) & mask
        total = mask.sum()
        accuracy = correct.sum().float() / total.float() if total > 0 else torch.tensor(0.0)
        
        # Compute exact match accuracy (whole sequence correct)
        exact_match_accuracy = self._compute_exact_match_accuracy(predictions, targets, mask)
        
        # Compute path validity and edge accuracy
        path_validity = self._compute_path_validity(predictions, targets)
        edge_accuracy = self._compute_edge_accuracy(predictions, targets)
        
        return {
            'accuracy': accuracy,
            'exact_match_accuracy': exact_match_accuracy,
            'path_validity': path_validity,
            'edge_accuracy': edge_accuracy
        }
    
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
    
    def _compute_path_validity(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Check if predicted paths contain valid edges in the graph"""
        batch_size = predictions.size(0)
        valid_paths = 0
        
        for i in range(batch_size):
            # Only consider non-padded tokens (where targets != -100)
            mask = targets[i] != -100
            path = predictions[i][mask]
            
            # Convert to list and remove padding tokens (assuming 0 is padding)
            path_list = path.cpu().tolist()[:-1]  # remove the eos token
            path_list = [node for node in path_list if node != 0]
            
            if len(path_list) <= 1:
                valid_paths += 1  # Single node or empty path is valid
                continue
            
            # Check if consecutive nodes are connected
            is_valid = True
            for j in range(len(path_list) - 1):
                node1, node2 = path_list[j], path_list[j + 1]
                if not self.graph.has_edge(node1, node2):
                    is_valid = False
                    break
            
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


class GenerativeMetrics:
    """Class to handle generative evaluation metrics"""
    
    def __init__(self, graph: nx.Graph, vocab_size: int):
        self.graph = graph
        self.vocab_size = vocab_size
        self.eos_token = vocab_size - 1
    
    def generate_path(self, model, goal_state: int, start_state: int, max_length: int = 64, 
                     temperature: float = 1.0, top_k: int = None) -> list:
        """Generate a path from start_state to goal_state using autoregressive sampling"""
        model.eval()
        device = next(model.parameters()).device
        
        # Initialize with goal and start state
        input_sequence = torch.tensor([goal_state, start_state], dtype=torch.long, device=device).unsqueeze(0)
        
        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                logits = model(input_sequence)
                
                # Get logits for the last position
                next_token_logits = logits[0, -1, :] / temperature
                
                # Apply top-k filtering if specified
                if top_k is not None:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits[top_k_indices] = top_k_logits
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)
                
                # Add to sequence
                input_sequence = torch.cat([input_sequence, next_token.unsqueeze(0)], dim=1)
                
                # Check if EOS token generated
                if next_token.item() == self.eos_token:
                    break
        
        # Extract generated path (excluding goal and start tokens, including any EOS)
        full_sequence = input_sequence[0].cpu().tolist()
        
        # Remove goal (first token) and start (second token) 
        # The remaining should be the continuation of the path
        generated_tokens = full_sequence[2:]
        
        # Remove EOS token if present
        if generated_tokens and generated_tokens[-1] == self.eos_token:
            generated_tokens = generated_tokens[:-1]
        
        return generated_tokens
    
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
    
    def evaluate_generative(self, model, batch: Dict[str, torch.Tensor], num_samples: int = 1, 
                          max_length: int = 64, temperature: float = 1.0) -> Dict[str, float]:
        """Evaluate model in generative mode on a batch"""
        input_ids = batch['input_ids']
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
                # Generate path
                generated_path = self.generate_path(model, goal_state, start_state, max_length, temperature)
                
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
        
        metrics = {
            'gen_path_validity': total_valid_paths / total_samples if total_samples > 0 else 0.0,
            'gen_goal_accuracy': total_correct_goals / total_samples if total_samples > 0 else 0.0
        }
        
        # Add average path length difference if we have data
        if length_differences:
            metrics['gen_avg_path_length_diff'] = sum(length_differences) / len(length_differences)
        else:
            metrics['gen_avg_path_length_diff'] = 0.0
        
        return metrics
import torch
import torch.nn as nn
import torch.nn.functional as F


class IterativeConvModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_iterations: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()

        self.d_model = d_model
        self.vocab_size = vocab_size
        self.num_iterations = num_iterations
        self.mask_token_id = vocab_size - 1
        # Token embedding
        self.embedding = nn.Embedding(vocab_size, d_model)

        # Convolutional block (will be applied iteratively)
        padding = (kernel_size - 1) // 2
        self.conv1 = nn.Conv1d(d_model, d_model, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(d_model, d_model, kernel_size, padding=padding)
        self.dropout = nn.Dropout(dropout)

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.Conv1d):
                torch.nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)

    def forward(self, input_ids, target_ids, attention_mask=None):
        # input_ids: (batch_size, seq_len)

        # Embed tokens
        x = self.embedding(input_ids)  # (batch_size, seq_len, d_model)

        # Transpose for conv1d: (batch_size, d_model, seq_len)
        x = x.transpose(1, 2)

        # Apply convolutional block iteratively
        for _ in range(self.num_iterations):
            residual = x

            # First conv + ReLU
            x = self.conv1(x)
            x = F.relu(x)
            x = self.dropout(x)

            # Second conv + ReLU
            x = self.conv2(x)
            x = F.relu(x)
            x = self.dropout(x)

            # Residual connection
            x = x + residual

        # Transpose back: (batch_size, seq_len, d_model)
        x = x.transpose(1, 2)

        # Project to vocabulary
        logits = self.output_projection(x)

        # Compute loss if target_ids is provided
        result = {"logits": logits}
        if target_ids is not None:
            loss = self.get_loss(logits, target_ids, input_ids)
            result["loss"] = loss

        return result

    def get_loss(self, logits, targets, masked_input_ids):
        # logits: (batch_size, seq_len, vocab_size)
        # targets: (batch_size, seq_len) with -100 for padding tokens

        mask_token_id = self.mask_token_id
        mask = (masked_input_ids == mask_token_id)
        # Set targets to -100 (ignore_index) where not masked
        targets = targets.clone()
        targets[~mask] = -100

        shift_logits = logits.view(-1, logits.size(-1))
        shift_labels = targets.view(-1)

        loss = F.cross_entropy(
            shift_logits,
            shift_labels,
            ignore_index=-100
        )
        return loss

    def generate_path(self, goal_state: int, start_state: int, max_length: int = 64,
                     temperature: float = 1.0, top_k: int = None) -> list:
        """Generate a path from start_state to goal_state using autoregressive sampling"""
        self.eval()
        device = next(self.parameters()).device
        eos_token = self.vocab_size - 1

        # Initialize with goal and start state
        input_sequence = torch.tensor([goal_state, start_state], dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            for _ in range(max_length):
                # Get model predictions
                output = self(input_sequence, None)
                logits = output["logits"]

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
                if next_token.item() == eos_token:
                    break

        # Extract generated path (excluding goal and start tokens, including any EOS)
        full_sequence = input_sequence[0].cpu().tolist()

        # Remove goal (first token) and start (second token)
        generated_tokens = full_sequence[2:]

        # Remove EOS token if present
        if generated_tokens and generated_tokens[-1] == eos_token:
            generated_tokens = generated_tokens[:-1]

        return generated_tokens

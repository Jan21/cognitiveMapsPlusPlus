import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_seq_length: int = 128):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:x.size(0), :]


class TransformerModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 128,
        dropout: float = 0.1, **kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_seq_length)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, batch, attention_mask=None):
        # input_ids: (batch_size, seq_len)
        # attention_mask: (batch_size, seq_len)
        input_ids = batch['input_ids']
        target_ids = batch['target_ids']
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = self.pos_encoding(x.transpose(0, 1)).transpose(0, 1)
        x = self.dropout(x)
        
        # Create causal mask for autoregressive generation
        seq_len = input_ids.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(input_ids.device)
        
        # Convert attention_mask to the format expected by transformer
        if attention_mask is not None:
            # attention_mask: 1 for real tokens, 0 for padding
            # transformer expects: 0 for real tokens, -inf for masked
            key_padding_mask = (attention_mask == 0)
        else:
            key_padding_mask = None
        
        x = self.transformer(
            x, 
            mask=causal_mask,
            src_key_padding_mask=key_padding_mask
        )
        
        logits = self.output_projection(x)
        
        # Only compute loss if target_ids is provided
        if target_ids is not None:
            loss = self.get_loss(logits, target_ids)
            return {"logits": logits, "loss": loss}
        else:
            return {"logits": logits}
    
    def get_loss(self, logits, targets):
        # logits: (batch_size, seq_len, vocab_size)
        # targets: (batch_size, seq_len) with -100 for padding tokens
        
        # Reshape for loss computation
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
                output = self({'input_ids': input_sequence, 'target_ids': None})
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
        # The remaining should be the continuation of the path
        generated_tokens = full_sequence[2:]
        
        # Remove EOS token if present
        if generated_tokens and generated_tokens[-1] == eos_token:
            generated_tokens = generated_tokens[:-1]
        
        return generated_tokens
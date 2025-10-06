import torch
import torch.nn as nn
import math
import torch.nn.functional as F
class DiffusionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, context, attention_mask=None):
        # Self-attention
        attn_out, _ = self.self_attention(x, x, x, key_padding_mask=attention_mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Cross-attention with context (goal + start tokens)
        attn_out, _ = self.cross_attention(x, context, context)
        x = self.norm2(x + self.dropout(attn_out))
        
        # Feed forward
        ff_out = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_out))
        
        return x


class ConcatenatedDiffusionBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        
        self.self_attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, path_tokens, context_tokens, attention_mask=None):
        """
        Args:
            path_tokens: (batch_size, path_len, d_model) - Path token embeddings
            context_tokens: (batch_size, context_len, d_model) - Context token embeddings
            attention_mask: Optional attention mask
        
        Returns:
            path_tokens: (batch_size, path_len, d_model) - Updated path tokens only
        """
        batch_size, path_len, d_model = path_tokens.shape
        context_len = context_tokens.size(1)
        
        # Concatenate context and path tokens
        concatenated = torch.cat([context_tokens, path_tokens], dim=1)  # (batch_size, context_len + path_len, d_model)
        
        # Self-attention across all tokens (context + path)
        attn_out, _ = self.self_attention(concatenated, concatenated, concatenated, key_padding_mask=attention_mask)
        concatenated = self.norm1(concatenated + self.dropout(attn_out))
        
        # Feed forward
        ff_out = self.feed_forward(concatenated)
        concatenated = self.norm2(concatenated + self.dropout(ff_out))
        
        # Return only the path tokens (excluding context tokens)
        updated_path_tokens = concatenated[:, context_len:, :]  # (batch_size, path_len, d_model)
        
        return updated_path_tokens


class DiffusionModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        num_heads: int = 8,
        num_layers: int = 6,
        d_ff: int = 2048,
        max_seq_length: int = 128,
        max_path_length: int = 126,  # max_seq_length - 2 (goal + start)
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_path_length = max_path_length
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_length, d_model)     
        
        # Context encoder for goal + start tokens
        self.context_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation='relu',
                batch_first=True
            ),
            num_layers=2
        )
        
        # Diffusion transformer blocks
        self.internal_blocks = nn.ModuleList([
            ConcatenatedDiffusionBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(2)
        ])

        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.num_layers = num_layers
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, input_ids, target_ids, attention_mask=None):
        """
        Args:
            input_ids: (batch_size, seq_len) - For training: full sequence (goal, start, path...)
                                               - For inference: only (goal, start) + noisy path tokens
            timesteps: (batch_size,) - Diffusion timesteps
            attention_mask: (batch_size, seq_len) - 1 for real tokens, 0 for padding
        
        Returns:
            logits: (batch_size, seq_len, vocab_size) - Same format as transformer
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Extract context (goal + start tokens - first 2 tokens)
        context_tokens = input_ids[:, :2]  # (batch_size, 2)
        
        # For the path tokens (everything after goal + start)
        if seq_len > 2:
            path_tokens = input_ids[:, 2:]  # (batch_size, path_len)
            path_len = path_tokens.size(1) + 1
        else:
            # During inference, we might only have goal + start
            path_len = self.max_path_length + 1
            path_tokens = torch.zeros(batch_size, path_len, dtype=torch.long, device=device)
        
        # Embed context tokens
        context_embeds = self.token_embedding(context_tokens) * math.sqrt(self.d_model)
        context_pos = torch.arange(2, device=device).unsqueeze(0).expand(batch_size, -1)
        context_embeds = context_embeds + self.position_embedding(context_pos)
        
        # Encode context
        context = context_embeds  # (batch_size, 2, d_model)
        
        # Initialize path embeddings as random vectors
        path_embeds = torch.randn(batch_size, path_len, self.d_model, device=device) * math.sqrt(self.d_model)
        path_pos = torch.arange(2, 2 + path_len, device=device).unsqueeze(0).expand(batch_size, -1)
        path_embeds = path_embeds + self.position_embedding(path_pos)
        
        
        x = self.dropout(path_embeds)
        
        # Create attention mask for path tokens only
        if attention_mask is not None and seq_len > 2:
            path_attention_mask = attention_mask[:, 2:]
            path_attention_mask = (path_attention_mask == 0)  # Convert to key_padding_mask format
        else:
            path_attention_mask = None
        
        # Pass through diffusion blocks
        for i in range(self.num_layers):
            for block in self.internal_blocks:
                x = block(x, context, path_attention_mask)
        
        # Project to vocabulary
        path_logits = self.output_projection(x)  # (batch_size, path_len, vocab_size)
        
        # Reconstruct full sequence logits (context + path)
        if seq_len > 2:
            # During training, we need to match the input sequence length
            # Create dummy logits for context tokens (they won't be used in loss)
            context_logits = torch.zeros(batch_size, 1, self.vocab_size, device=device)
            logits = torch.cat([context_logits, path_logits], dim=1)
            
            # Trim to match input sequence length
            logits = logits[:, :seq_len, :]
        else:
            # During inference with only goal + start, return all predictions
            context_logits = torch.zeros(batch_size, 1, self.vocab_size, device=device)
            logits = torch.cat([context_logits, path_logits], dim=1)
        
        loss = self.get_loss(logits, target_ids)
        return {"logits": logits, "loss": loss}
    
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
    
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional
import math


@dataclass
class PASTConfig:
    vocab_size: int = 256
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True
    train_mode: str = "absorbing"
    tie_lmhead: bool = True


class LayerNorm(nn.Module):
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class RotaryPositionalEmbeddings(nn.Module):
    def __init__(self, d: int, base: int = 10_000):
        super().__init__()
        self.base = base
        self.d = d
        self.cos_cached = None
        self.sin_cached = None

    def _build_cache(self, x: torch.Tensor):
        if self.cos_cached is not None and x.shape[-2] <= self.cos_cached.shape[0]:
            return

        seq_len = x.shape[-2]
        theta = 1.0 / (
            self.base ** (torch.arange(0, self.d, 2, device=x.device).float() / self.d)
        )
        seq_idx = torch.arange(seq_len, device=x.device).float()
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)
        
        self.cos_cached = idx_theta2.cos()
        self.sin_cached = idx_theta2.sin()

    def _neg_half(self, x: torch.Tensor):
        d_2 = self.d // 2
        return torch.cat([-x[..., d_2:], x[..., :d_2]], dim=-1)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x: torch.Tensor):
        self._build_cache(x)
        x_rope, x_pass = x[..., : self.d], x[..., self.d :]
        neg_half_x = self._neg_half(x_rope)
        x_rope = (x_rope * self.cos_cached[: x.shape[-2]]) + (
            neg_half_x * self.sin_cached[: x.shape[-2]]
        )
        return torch.cat((x_rope, x_pass), dim=-1)


class Attention(nn.Module):
    def __init__(self, config, rotary=None, self_attention=True):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.rotary = rotary
        self.self_attention = self_attention
        
        if self_attention:
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        else:
            self.c_attn = nn.Linear(config.n_embd, 1 * config.n_embd, bias=config.bias)
            self.c_attn_mem = nn.Linear(
                config.n_embd, 2 * config.n_embd, bias=config.bias
            )

        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

    def forward(self, x, attn_mask=None, mems=None, mems_mask=None):
        B, T, C = x.size()
        
        if self.self_attention:
            q, k, v = self.c_attn(x).split(C, dim=2)
        else:
            assert mems.size(0) == B
            mems_mask = mems_mask.expand(B, 1, T, -1)
            q = self.c_attn(x)
            k, v = self.c_attn_mem(mems).split(C, dim=2)
            attn_mask = mems_mask

        k = k.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, -1, self.n_head, C // self.n_head).transpose(1, 2)
        q, k = self.rotary(q), self.rotary(k)

        y = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.resid_dropout(self.c_proj(y))
        return y


class Block(nn.Module):
    def __init__(self, config, rotary, self_attention):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = Attention(config, rotary, self_attention)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x, attn_mask=None, mems=None, mems_mask=None):
        x = x + self.attn(
            self.ln_1(x), attn_mask=attn_mask, mems=mems, mems_mask=mems_mask
        )
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(torch.nn.Module):
    def __init__(self, config, rotary=None, self_attention=True):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [Block(config, rotary, self_attention) for _ in range(config.n_layer)]
        )
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)

    def forward(self, x, mask=None, mems_list=None, mems_mask=None):
        new_mems = []
        mems_list = mems_list or [None] * len(self.layers)
        for layer, mems in zip(self.layers, mems_list):
            x = layer(x, mask, mems, mems_mask)
            new_mems.append(x)
        x = self.ln_f(x)
        return x, new_mems


class PAST(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.rotary = RotaryPositionalEmbeddings(config.n_embd // config.n_head, 10_000)
        self.enc = Transformer(config, rotary=self.rotary, self_attention=True)
        self.dec = Transformer(config, rotary=self.rotary, self_attention=False)
        self.wte = torch.nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = torch.nn.Embedding(1, config.n_embd)
        self.lm_head = torch.nn.Linear(config.n_embd, config.vocab_size)
        if config.tie_lmhead:
            self.wte.weight = self.lm_head.weight

    def forward(
        self,
        input_ids,
        target_ids,
        attention_mask=None
    ):
        mode = "absorbing" #if self.training else "all_but_last"
        B, T, device = input_ids.size(0), input_ids.size(1), input_ids.device

        # if attention_mask is not None:
        #     attention_mask = attention_mask.float() > 0
        # else:
        #     attention_mask = torch.ones(B, 1, 1, T, dtype=torch.bool, device=device)
        #attention_mask = attention_mask.view(B, 1, 1, T) # attention mask is used to mask out padding tokens
        # Find the padding token (401) and create a mask for everything after it
        padding_token = 401
        
        # Find the position of the first padding token in each sequence
        padding_positions = (input_ids == padding_token).int()
        
        # Create a cumulative mask: once we hit a padding token, everything after should be masked
        # First, find the first occurrence of padding token in each sequence
        first_padding_pos = torch.argmax(padding_positions, dim=1)  # Position of first padding token
        
        # Handle sequences that don't have padding tokens
        has_padding = padding_positions.sum(dim=1) > 0
        first_padding_pos = torch.where(has_padding, first_padding_pos, T)
        
        # Create attention mask that masks everything from first padding token onwards
        positions = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        attention_mask = positions < first_padding_pos.unsqueeze(1)
        attention_mask = attention_mask.view(B, 1, 1, T)
        
        #prediction_mask = (prediction_mask.float() > 0).view(B, 1, 1, T) # prediction mask should put zeros on prefix tokens and ones on the rest
        prediction_mask = torch.ones(B, 1, 1, T, dtype=torch.bool, device=device)
        prediction_mask[:, :, :, :2] = False


        enc_mask, model_pred_mask = self.get_mask(
                prediction_mask, mode, device=device
            ) # model pred mask is negated enc mask

        enc_mask = enc_mask & attention_mask # which tokens are used for encoding, attention mask is used to mask out padding tokens
        prediction_mask = prediction_mask & model_pred_mask & attention_mask # which tokens are used for prediction
        
        pos_emb = self.wpe(torch.zeros(T, device=device, dtype=torch.long))
        pos_emb = pos_emb.view(1, T, -1).expand(B, T, -1)
        tok_emb = self.wte(input_ids)
        _, mems_list = self.enc(tok_emb, enc_mask)
        x, _ = self.dec(pos_emb, None, mems_list, enc_mask)
        logits = self.lm_head(x)
        loss = self.get_loss(logits, input_ids, prediction_mask)
        # Remove logits for the first token and add pad zero logits for one additional token
        logits = logits[:, 1:, :]  # Remove first token logits
        # Add zero logits for one additional token at the end
        B, T_new, vocab_size = logits.shape
        zero_logits = torch.zeros(B, 1, vocab_size, device=logits.device, dtype=logits.dtype)
        logits = torch.cat([logits, zero_logits], dim=1)
        return {"logits": logits, "loss": loss}

    def get_loss(
        self,
        logits: torch.FloatTensor,
        input_ids: torch.IntTensor,
        prediction_mask: torch.BoolTensor,
    ):
        prediction_mask = prediction_mask.view(input_ids.shape).float()
        logits = logits.view(-1, logits.size(-1))
        tgts = input_ids.view(-1)
        loss = torch.nn.functional.cross_entropy(logits, tgts, reduction="none")
        loss = loss.view(input_ids.shape) * prediction_mask
        return loss.sum() / prediction_mask.sum()

    def get_mask(self, need_to_pred, mode, device):
        need_to_pred = need_to_pred.bool()
        B, T = need_to_pred.size(0), need_to_pred.size(-1)
        
        if mode == "absorbing":
            enc_mask = self._get_diffusion_masks(B, T, device=device) | ~need_to_pred
            prediction_mask = ~enc_mask[:, :, :1, :]
        elif mode == "all_but_last":
            enc_mask = torch.ones(B, 1, T, T, dtype=torch.bool, device=device)
            enc_mask[:, :, :, -1] = False
            prediction_mask = ~enc_mask[:, :, :1, :]
        else:
            raise ValueError(f"Unsupported mode: {mode}")
            
        return enc_mask, prediction_mask

    def _get_diffusion_masks(self, bsz, seq, rate=None, device=None):
        uniform_tensor = torch.rand(bsz, seq, device=device)
        prob_tensor = torch.rand(bsz, 1, device=device) if rate is None else rate
        mask_array = prob_tensor > uniform_tensor
        mask_array[:, 0] = True
        return mask_array.unsqueeze(1).unsqueeze(1).expand(bsz, 1, seq, seq)
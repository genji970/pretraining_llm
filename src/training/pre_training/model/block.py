from __future__ import annotations

import torch
from torch import nn

from training.pre_training.model.attention import CausalSelfAttention


class FeedForward(nn.Module):
    def __init__(self, d_model: int, multiplier: int = 4, dropout: float = 0.0):
        super().__init__()
        hidden_dim = d_model * multiplier
        self.network = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.network(hidden_states)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        ffn_multiplier: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = CausalSelfAttention(d_model, n_heads, max_seq_len, dropout)
        self.feed_forward_norm = nn.LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, ffn_multiplier, dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(
            self.attention_norm(hidden_states),
            attention_mask,
        )
        hidden_states = hidden_states + self.feed_forward(
            self.feed_forward_norm(hidden_states)
        )
        return hidden_states


if __name__ == "__main__":
    block = TransformerBlock(d_model=32, n_heads=4, max_seq_len=16)
    x = torch.randn(2, 7, 32)
    mask = torch.ones(2, 7, dtype=torch.long)
    y = block(x, mask)
    print(x.shape, "->", y.shape)

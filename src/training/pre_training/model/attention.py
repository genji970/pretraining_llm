from __future__ import annotations

import torch
from torch import nn

from training.pre_training.model.rope import RotaryPositionEmbedding


class CausalSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        if self.head_dim % 2 != 0:
            raise ValueError("The per-head dimension must be even for RoPE")

        self.query_projection = nn.Linear(d_model, d_model, bias=False)
        self.key_projection = nn.Linear(d_model, d_model, bias=False)
        self.value_projection = nn.Linear(d_model, d_model, bias=False)
        self.output_projection = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryPositionEmbedding(max_seq_len, self.head_dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.residual_dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = x.shape
        return x.view(batch_size, sequence_length, self.n_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("hidden_states must have shape [batch, sequence, d_model]")

        batch_size, sequence_length, _ = hidden_states.shape
        query = self._split_heads(self.query_projection(hidden_states))
        key = self._split_heads(self.key_projection(hidden_states))
        value = self._split_heads(self.value_projection(hidden_states))
        query, key = self.rope(query, key)

        scores = query @ key.transpose(-1, -2)
        scores = scores / (self.head_dim**0.5)

        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=hidden_states.device,
            ),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask[None, None, :, :], float("-inf"))

        if attention_mask is not None:
            expected_shape = (batch_size, sequence_length)
            if tuple(attention_mask.shape) != expected_shape:
                raise ValueError(
                    f"attention_mask must have shape {expected_shape}, got {tuple(attention_mask.shape)}"
                )
            key_padding_mask = attention_mask[:, None, None, :] == 0
            scores = scores.masked_fill(key_padding_mask, float("-inf"))

        probabilities = torch.softmax(scores, dim=-1)
        probabilities = self.attention_dropout(probabilities)
        context = probabilities @ value
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.d_model)
        )
        return self.residual_dropout(self.output_projection(context))


if __name__ == "__main__":
    attention = CausalSelfAttention(d_model=32, n_heads=4, max_seq_len=16)
    hidden_states = torch.randn(2, 7, 32)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 0, 0, 0],
        ]
    )
    output = attention(hidden_states, attention_mask)
    print("input:", hidden_states.shape)
    print("output:", output.shape)
    print("finite:", torch.isfinite(output).all().item())

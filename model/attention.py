from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .rope import RoPE_Positional_Embedding
except ImportError:  # Allows: python model/attention.py
    from rope import RoPE_Positional_Embedding


class Self_Attention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        context_length: int,
        num_head: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if embed_dim % num_head != 0:
            raise ValueError("embed_dim must be divisible by num_head")

        self.embed_dim = embed_dim
        self.context_length = context_length
        self.num_head = num_head
        self.head_dim = embed_dim // num_head

        self.positional_embedding = RoPE_Positional_Embedding(
            context_length=context_length,
            head_dim=self.head_dim,
        )

        self.query_embedding = nn.Parameter(
            torch.empty(embed_dim, embed_dim)
        )
        self.key_embedding = nn.Parameter(
            torch.empty(embed_dim, embed_dim)
        )
        self.value_embedding = nn.Parameter(
            torch.empty(embed_dim, embed_dim)
        )

        self.output_projection = nn.Linear(
            embed_dim,
            embed_dim,
            bias=False,
        )
        self.attention_dropout = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.query_embedding)
        nn.init.xavier_uniform_(self.key_embedding)
        nn.init.xavier_uniform_(self.value_embedding)
        nn.init.xavier_uniform_(self.output_projection.weight)

    def construct_q_k_v(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            x @ self.query_embedding,
            x @ self.key_embedding,
            x @ self.value_embedding,
        )

    def split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = tensor.shape
        return tensor.view(
            batch_size,
            sequence_length,
            self.num_head,
            self.head_dim,
        ).transpose(1, 2)

    @staticmethod
    def causal_masking(
        attention_matrix: torch.Tensor,
    ) -> torch.Tensor:
        sequence_length = attention_matrix.size(-1)
        mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                device=attention_matrix.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        )
        return attention_matrix.masked_fill(
            mask[None, None, :, :],
            -torch.inf,
        )

    @staticmethod
    def padding_masking(
        attention_logits: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_size: int,
        sequence_length: int,
    ) -> torch.Tensor:
        if attention_mask.shape != (batch_size, sequence_length):
            raise ValueError(
                "attention_mask must have shape "
                "[batch_size, sequence_length]"
            )

        key_padding_mask = attention_mask[:, None, None, :] == 0
        return attention_logits.masked_fill(
            key_padding_mask,
            -torch.inf,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, sequence_length, embed_dim = x.shape

        if embed_dim != self.embed_dim:
            raise ValueError(
                f"expected embed_dim={self.embed_dim}, received {embed_dim}"
            )
        if sequence_length > self.context_length:
            raise ValueError(
                f"sequence length {sequence_length} "
                f"exceeds {self.context_length}"
            )

        query, key, value = self.construct_q_k_v(x)
        query = self.split_heads(query)
        key = self.split_heads(key)
        value = self.split_heads(value)

        sin, cos = self.positional_embedding.transformation(
            sequence_length=sequence_length,
            device=x.device,
            dtype=x.dtype,
        )
        query = self.positional_embedding.construct_embedding(
            query,
            sin,
            cos,
        )
        key = self.positional_embedding.construct_embedding(
            key,
            sin,
            cos,
        )

        attention_logits = (
            query @ key.transpose(-1, -2)
        ) / (self.head_dim**0.5)
        attention_logits = self.causal_masking(attention_logits)

        if attention_mask is not None:
            attention_logits = self.padding_masking(
                attention_logits=attention_logits,
                attention_mask=attention_mask,
                batch_size=batch_size,
                sequence_length=sequence_length,
            )

        attention_weights = torch.softmax(
            attention_logits,
            dim=-1,
        )
        attention_weights = self.attention_dropout(attention_weights)

        attention_output = attention_weights @ value
        attention_output = (
            attention_output.transpose(1, 2)
            .contiguous()
            .view(batch_size, sequence_length, self.embed_dim)
        )

        return self.output_projection(attention_output)


if __name__ == "__main__":
    attention = Self_Attention(
        embed_dim=32,
        context_length=8,
        num_head=4,
        dropout=0.0,
    )
    x = torch.randn(3, 7, 32)
    attention_mask = torch.ones(3, 7, dtype=torch.long)
    attention_mask[0, -2:] = 0

    output = attention(x, attention_mask)
    print(f"output.shape={tuple(output.shape)}")

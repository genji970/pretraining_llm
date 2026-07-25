from __future__ import annotations

import torch
import torch.nn as nn

try:
    from .attention import Self_Attention
except ImportError:  # Allows: python model/transformer_block.py
    from attention import Self_Attention


class Transformer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        context_length: int,
        num_head: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.layernorm_list = nn.ModuleList(
            [nn.LayerNorm(embed_dim) for _ in range(2)]
        )
        self.causal_attention = Self_Attention(
            embed_dim=embed_dim,
            context_length=context_length,
            num_head=num_head,
            dropout=dropout,
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        normalized_x = self.layernorm_list[0](x)
        x = x + self.causal_attention(
            normalized_x,
            attention_mask,
        )

        normalized_x = self.layernorm_list[1](x)
        x = x + self.feed_forward(normalized_x)
        return x


if __name__ == "__main__":
    block = Transformer(
        embed_dim=32,
        context_length=8,
        num_head=4,
        dropout=0.0,
    )
    x = torch.randn(3, 7, 32)
    attention_mask = torch.ones(3, 7, dtype=torch.long)

    output = block(x, attention_mask)
    print(f"output.shape={tuple(output.shape)}")

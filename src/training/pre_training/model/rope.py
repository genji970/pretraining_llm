from __future__ import annotations

import torch
from torch import nn


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, max_seq_len: int, head_dim: int, base: float = 10000.0):
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")

        position = torch.arange(max_seq_len, dtype=torch.float32)[:, None]
        dimension = torch.arange(0, head_dim, 2, dtype=torch.float32)[None, :]
        inverse_frequency = base ** (-dimension / head_dim)
        angles = position * inverse_frequency

        self.max_seq_len = max_seq_len
        self.register_buffer(
            "cos_cache",
            torch.cos(angles)[None, None, :, :],
            persistent=False,
        )
        self.register_buffer(
            "sin_cache",
            torch.sin(angles)[None, None, :, :],
            persistent=False,
        )

    def forward(self, query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sequence_length = query.size(-2)
        if sequence_length > self.max_seq_len:
            raise ValueError(
                f"sequence_length={sequence_length} exceeds max_seq_len={self.max_seq_len}"
            )
        cos = self.cos_cache[:, :, :sequence_length, :].to(dtype=query.dtype)
        sin = self.sin_cache[:, :, :sequence_length, :].to(dtype=query.dtype)
        return self._rotate(query, cos, sin), self._rotate(key, cos, sin)

    @staticmethod
    def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


if __name__ == "__main__":
    rope = RotaryPositionEmbedding(max_seq_len=16, head_dim=8)
    query = torch.randn(2, 4, 7, 8)
    key = torch.randn(2, 4, 7, 8)
    rotated_query, rotated_key = rope(query, key)
    print(rotated_query.shape, rotated_key.shape)
    print("norm preserved:", torch.allclose(query.norm(dim=-1), rotated_query.norm(dim=-1)))

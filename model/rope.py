from __future__ import annotations

import torch
import torch.nn as nn


class RoPE_Positional_Embedding(nn.Module):
    """Dynamic RoPE without a fixed batch dimension or repeated frequency tensor."""

    def __init__(self, context_length: int, head_dim: int) -> None:
        super().__init__()
        if head_dim % 2 != 0:
            raise ValueError("head_dim must be even")
        if context_length <= 0:
            raise ValueError("context_length must be positive")

        self.context_length = context_length
        self.head_dim = head_dim

        inv_frequency = 10000 ** (
            -torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim
        )
        self.register_buffer(
            "inv_frequency",
            inv_frequency,
            persistent=False,
        )

    def transformation(
        self,
        sequence_length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence_length > self.context_length:
            raise ValueError(
                f"sequence_length={sequence_length} exceeds "
                f"context_length={self.context_length}"
            )

        positions = torch.arange(
            sequence_length,
            device=device,
            dtype=torch.float32,
        )
        frequency = torch.outer(
            positions,
            self.inv_frequency.to(device=device),
        )
        frequency = frequency[None, None, :, :]

        sin = torch.sin(frequency).to(dtype=dtype)
        cos = torch.cos(frequency).to(dtype=dtype)
        return sin, cos

    @staticmethod
    def construct_embedding(
        x: torch.Tensor,
        sin: torch.Tensor,
        cos: torch.Tensor,
    ) -> torch.Tensor:
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        rotate_even = x_even * cos - x_odd * sin
        rotate_odd = x_even * sin + x_odd * cos

        return torch.stack(
            [rotate_even, rotate_odd],
            dim=-1,
        ).flatten(-2)


if __name__ == "__main__":
    rope = RoPE_Positional_Embedding(
        context_length=8,
        head_dim=4,
    )
    x = torch.randn(2, 3, 7, 4)
    sin, cos = rope.transformation(
        sequence_length=x.size(-2),
        device=x.device,
        dtype=x.dtype,
    )
    output = rope.construct_embedding(x, sin, cos)

    print(f"sin.shape={tuple(sin.shape)}")
    print(f"cos.shape={tuple(cos.shape)}")
    print(f"output.shape={tuple(output.shape)}")

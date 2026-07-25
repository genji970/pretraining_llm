from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import nn


class TokenEmbeddingSpace(nn.Module, ABC):
    """Interface separating token IDs from their learned vector space."""

    @property
    @abstractmethod
    def weight(self) -> torch.Tensor:
        raise NotImplementedError


class LearnedTokenEmbedding(TokenEmbeddingSpace):
    def __init__(self, vocab_size: int, d_model: int, scale: bool = True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.scale = d_model**0.5 if scale else 1.0

    @property
    def weight(self) -> torch.Tensor:
        return self.embedding.weight

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding(input_ids) * self.scale


def build_embedding_space(kind: str, vocab_size: int, d_model: int) -> TokenEmbeddingSpace:
    if kind == "learned":
        return LearnedTokenEmbedding(vocab_size=vocab_size, d_model=d_model)
    raise ValueError(f"Unknown embedding_type: {kind}")


if __name__ == "__main__":
    embedding = build_embedding_space("learned", vocab_size=20, d_model=8)
    input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]])
    output = embedding(input_ids)
    print("input:", input_ids.shape)
    print("embedded:", output.shape)
    print("weight:", embedding.weight.shape)

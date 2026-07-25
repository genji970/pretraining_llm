"""Backward-compatible model entry point.

Existing code can continue to use::

    from model.model import DecoderLanguageModel

The actual implementations are separated by responsibility in this package.
"""

from __future__ import annotations

import torch

try:
    from .attention import Self_Attention
    from .decoder_lm import DecoderLanguageModel
    from .rope import RoPE_Positional_Embedding
    from .stacking import Stacking
    from .transformer_block import Transformer
except ImportError:  # Allows: python model/model.py
    from attention import Self_Attention
    from decoder_lm import DecoderLanguageModel
    from rope import RoPE_Positional_Embedding
    from stacking import Stacking
    from transformer_block import Transformer

__all__ = [
    "RoPE_Positional_Embedding",
    "Self_Attention",
    "Transformer",
    "Stacking",
    "DecoderLanguageModel",
]


if __name__ == "__main__":
    model = DecoderLanguageModel(
        vocab_size=32,
        block_num=2,
        embed_dim=32,
        context_length=8,
        num_head=4,
        dropout=0.0,
    )
    input_ids = torch.randint(0, 32, (3, 7))
    attention_mask = torch.ones_like(input_ids)

    logits = model(input_ids, attention_mask)
    print(f"logits.shape={tuple(logits.shape)}")

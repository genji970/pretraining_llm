from __future__ import annotations

import torch
from torch import nn

from training.pre_training.model.block import TransformerBlock
from training.pre_training.model.embedding import build_embedding_space


class DecoderLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        ffn_multiplier: int,
        max_seq_len: int,
        dropout: float = 0.0,
        tie_embeddings: bool = True,
        embedding_type: str = "learned",
    ):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.token_embedding = build_embedding_space(embedding_type, vocab_size, d_model)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    n_heads=n_heads,
                    max_seq_len=max_seq_len,
                    ffn_multiplier=ffn_multiplier,
                    dropout=dropout,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._initialize_weights)

        if tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, sequence]")
        if input_ids.size(1) > self.max_seq_len:
            raise ValueError(
                f"Input length {input_ids.size(1)} exceeds max_seq_len={self.max_seq_len}"
            )

        hidden_states = self.embedding_dropout(self.token_embedding(input_ids))
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask)
        hidden_states = self.final_norm(hidden_states)
        return self.lm_head(hidden_states)

    def parameter_count(self, trainable_only: bool = False) -> int:
        parameters = self.parameters()
        if trainable_only:
            parameters = (parameter for parameter in parameters if parameter.requires_grad)
        return sum(parameter.numel() for parameter in parameters)


if __name__ == "__main__":
    model = DecoderLanguageModel(
        vocab_size=50,
        d_model=32,
        n_heads=4,
        n_layers=2,
        ffn_multiplier=2,
        max_seq_len=16,
    )
    input_ids = torch.randint(0, 50, (2, 7))
    attention_mask = torch.ones_like(input_ids)
    logits = model(input_ids, attention_mask)
    print("logits:", logits.shape)
    print("parameters:", f"{model.parameter_count():,}")

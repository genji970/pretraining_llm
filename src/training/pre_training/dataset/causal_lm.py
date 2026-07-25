from __future__ import annotations

from collections.abc import Iterable

import torch
from torch.utils.data import Dataset

from training.pre_training.tokenization.base import TokenizerBase


class CausalLMDataset(Dataset[dict[str, torch.Tensor]]):
    """Creates shifted input/label pairs for decoder-only language modeling.

    ``document`` keeps each document independent and relies on dynamic padding.
    ``packed`` concatenates documents and creates fixed-length token blocks.
    """

    def __init__(
        self,
        texts: Iterable[str],
        tokenizer: TokenizerBase,
        max_seq_len: int,
        mode: str = "document",
    ):
        if max_seq_len < 2:
            raise ValueError("max_seq_len must be at least 2")
        if mode not in {"document", "packed"}:
            raise ValueError("mode must be 'document' or 'packed'")

        self.max_seq_len = max_seq_len
        self.mode = mode
        self.samples: list[torch.Tensor] = []

        if mode == "document":
            self._build_document_samples(texts, tokenizer)
        else:
            self._build_packed_samples(texts, tokenizer)

        if not self.samples:
            raise ValueError("No valid causal-LM samples were produced")

    def _build_document_samples(
        self,
        texts: Iterable[str],
        tokenizer: TokenizerBase,
    ) -> None:
        eos_token_id = getattr(tokenizer, "eos_token_id")
        for text in texts:
            token_ids = tokenizer.encode(str(text), add_bos=True, add_eos=True)
            if len(token_ids) > self.max_seq_len + 1:
                token_ids = token_ids[: self.max_seq_len]
                token_ids.append(eos_token_id)
            if len(token_ids) >= 2:
                self.samples.append(torch.tensor(token_ids, dtype=torch.long))

    def _build_packed_samples(
        self,
        texts: Iterable[str],
        tokenizer: TokenizerBase,
    ) -> None:
        all_token_ids: list[int] = []
        for text in texts:
            all_token_ids.extend(tokenizer.encode(str(text), add_bos=True, add_eos=True))

        block_size = self.max_seq_len + 1
        for start in range(0, len(all_token_ids), block_size):
            block = all_token_ids[start : start + block_size]
            if len(block) >= 2:
                self.samples.append(torch.tensor(block, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        tokens = self.samples[index]
        return {
            "input_ids": tokens[:-1],
            "labels": tokens[1:],
        }


if __name__ == "__main__":
    from training.pre_training.tokenization.regex_tokenizer import RegexTokenizer

    texts = ["one two three", "one two", "three"]
    tokenizer = RegexTokenizer()
    tokenizer.train(texts)

    document_dataset = CausalLMDataset(texts, tokenizer, max_seq_len=8, mode="document")
    packed_dataset = CausalLMDataset(texts, tokenizer, max_seq_len=8, mode="packed")
    print("document lengths:", [len(sample["input_ids"]) for sample in document_dataset])
    print("packed lengths:", [len(sample["input_ids"]) for sample in packed_dataset])

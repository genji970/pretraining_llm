from __future__ import annotations

import math
from collections.abc import Sequence
from functools import partial
from typing import Protocol

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset as TorchDataset


class TokenizerProtocol(Protocol):
    eos_token_id: int | None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...


class Train_Dataset(TorchDataset):
    """Pack one in-memory document chunk into a continuous next-token stream."""

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: TokenizerProtocol,
        context_length: int,
    ) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if not texts:
            raise ValueError("texts must contain at least one document")
        if tokenizer.eos_token_id is None:
            raise ValueError("The tokenizer must define eos_token_id")

        self.context_length = context_length
        eos_token_id = int(tokenizer.eos_token_id)

        all_token_ids: list[int] = []
        for text in texts:
            all_token_ids.extend(
                tokenizer.encode(str(text), add_special_tokens=False)
            )
            # GPT-2 uses <|endoftext|> as the boundary between documents.
            all_token_ids.append(eos_token_id)

        if len(all_token_ids) < 2:
            raise ValueError("At least two tokens are required")

        self.token_ids = torch.tensor(all_token_ids, dtype=torch.long)
        self.num_sequences = math.ceil(
            (len(self.token_ids) - 1) / self.context_length
        )

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if index < 0 or index >= self.num_sequences:
            raise IndexError("Dataset index out of range")

        start = index * self.context_length
        end = min(start + self.context_length + 1, len(self.token_ids))
        tokens = self.token_ids[start:end]
        return {"input_ids": tokens[:-1], "labels": tokens[1:]}


def collate_function(
    batch: list[dict[str, torch.Tensor]],
    pad_token_id: int,
) -> dict[str, torch.Tensor]:
    input_lengths = torch.tensor(
        [sample["input_ids"].numel() for sample in batch],
        dtype=torch.long,
    )

    input_ids = pad_sequence(
        [sample["input_ids"] for sample in batch],
        batch_first=True,
        padding_value=pad_token_id,
    )
    labels = pad_sequence(
        [sample["labels"] for sample in batch],
        batch_first=True,
        padding_value=-100,
    )

    # GPT-2 has no native pad token, so EOS is also used for padding. Building the
    # mask from token values would incorrectly hide real EOS document boundaries.
    positions = torch.arange(input_ids.size(1), dtype=torch.long).unsqueeze(0)
    attention_mask = (positions < input_lengths.unsqueeze(1)).long()

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def create_dataloader(
    dataset: Train_Dataset,
    batch_size: int,
    pad_token_id: int,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        collate_fn=partial(collate_function, pad_token_id=pad_token_id),
    )


if __name__ == "__main__":
    class TinyTokenizer:
        eos_token_id = 0

        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            del add_special_tokens
            return [ord(character) % 31 + 1 for character in text]

    texts = ["one two three four", "five six"]
    tokenizer = TinyTokenizer()
    dataset = Train_Dataset(texts, tokenizer, context_length=4)
    loader = create_dataloader(
        dataset,
        batch_size=2,
        pad_token_id=tokenizer.eos_token_id,
        shuffle=False,
    )
    batch = next(iter(loader))
    print({key: tuple(value.shape) for key, value in batch.items()})
    first = dataset[0]
    print(torch.equal(first["input_ids"][1:], first["labels"][:-1]))

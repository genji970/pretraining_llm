from __future__ import annotations

import math
from collections.abc import Sequence
from functools import partial

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset as TorchDataset

try:
    from .tokenizer import Tokenizer
except ImportError:  # Allows: python data/dataset.py
    from tokenizer import Tokenizer


class Train_Dataset(TorchDataset):
    """Pack documents into a single next-token stream, matching the notebook."""

    def __init__(
        self,
        texts: Sequence[str],
        tokenizer: Tokenizer,
        context_length: int,
    ) -> None:
        if context_length <= 0:
            raise ValueError("context_length must be positive")
        if not texts:
            raise ValueError("texts must contain at least one document")

        self.context_length = context_length
        all_token_ids: list[int] = []
        for text in texts:
            all_token_ids.extend(
                tokenizer.encode(str(text), add_bos=True, add_eos=True)
            )

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
    attention_mask = (input_ids != pad_token_id).long()
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
        collate_fn=partial(collate_function, pad_token_id=pad_token_id),
    )


if __name__ == "__main__":
    texts = ["one two three four", "five six"]
    tokenizer = Tokenizer(data_name="toy")
    tokenizer.build_vocab(texts, max_vocab_size=64, min_frequency=1)
    dataset = Train_Dataset(texts, tokenizer, context_length=4)
    loader = create_dataloader(
        dataset,
        batch_size=2,
        pad_token_id=tokenizer.dictionary["<pad>"],
        shuffle=False,
    )
    batch = next(iter(loader))
    print({key: tuple(value.shape) for key, value in batch.items()})
    first = dataset[0]
    print(torch.equal(first["input_ids"][1:], first["labels"][:-1]))

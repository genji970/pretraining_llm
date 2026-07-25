from __future__ import annotations

import torch
from torch.nn.utils.rnn import pad_sequence


class CausalLMCollator:
    """Dynamically right-pads each batch to its longest sequence."""

    def __init__(self, pad_token_id: int, label_pad_id: int = -100):
        self.pad_token_id = pad_token_id
        self.label_pad_id = label_pad_id

    def __call__(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("batch cannot be empty")

        input_ids = pad_sequence(
            [sample["input_ids"] for sample in batch],
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        labels = pad_sequence(
            [sample["labels"] for sample in batch],
            batch_first=True,
            padding_value=self.label_pad_id,
        )
        attention_mask = (input_ids != self.pad_token_id).long()
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


if __name__ == "__main__":
    collator = CausalLMCollator(pad_token_id=0)
    batch = collator(
        [
            {
                "input_ids": torch.tensor([1, 4, 5]),
                "labels": torch.tensor([4, 5, 2]),
            },
            {
                "input_ids": torch.tensor([1, 6]),
                "labels": torch.tensor([6, 2]),
            },
        ]
    )
    for name, tensor in batch.items():
        print(name, tensor.shape)
        print(tensor)

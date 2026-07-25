from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from training.pre_training.data.source.base import DataSource


class HuggingFaceSource(DataSource):
    """Lazy Hugging Face dataset source.

    ``datasets`` is imported only when this source is used, so local training does
    not require the optional dependency.
    """

    def __init__(
        self,
        dataset_name: str,
        split: str,
        config_name: str | None = None,
        streaming: bool = False,
    ):
        self.dataset_name = dataset_name
        self.split = split
        self.config_name = config_name
        self.streaming = streaming

    def records(self) -> Iterator[dict[str, Any]]:
        try:
            from datasets import load_dataset
        except ImportError as error:
            raise ImportError(
                "Install Hugging Face support with: pip install -e '.[huggingface]'"
            ) from error

        dataset = load_dataset(
            self.dataset_name,
            self.config_name,
            split=self.split,
            streaming=self.streaming,
        )
        for row in dataset:
            yield dict(row)


if __name__ == "__main__":
    print(
        "Example usage: HuggingFaceSource('m-a-p/SuperGPQA', split='train').records()"
    )
    print("The example is not downloaded automatically to keep this test offline-safe.")

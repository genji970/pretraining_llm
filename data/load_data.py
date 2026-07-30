from __future__ import annotations

from collections.abc import Iterable, Iterator
from itertools import cycle, islice


TOY_TEXTS = [
    "A language model predicts the next token from previous tokens.",
    "Wikipedia articles are commonly used as one component of pretraining data.",
    "Causal attention prevents a token from reading future tokens.",
    "The optimizer updates model parameters using gradients from the loss.",
]


def load_pretraining_stream(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    streaming: bool,
    seed: int,
    shuffle_buffer: int,
) -> Iterable[dict]:
    """Open the dataset once and return an iterable over source rows."""
    if dataset_name == "toy":
        return cycle({"text": text} for text in TOY_TEXTS)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for non-toy data. "
            "Install it with: pip install -r requirements.txt"
        ) from exc

    load_kwargs: dict[str, object] = {
        "split": split,
        "streaming": streaming,
    }
    if dataset_config is None:
        dataset = load_dataset(dataset_name, **load_kwargs)
    else:
        dataset = load_dataset(dataset_name, dataset_config, **load_kwargs)

    if shuffle_buffer > 0:
        if streaming:
            dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
        else:
            dataset = dataset.shuffle(seed=seed)

    return dataset


def skip_source_rows(rows: Iterable[dict], row_count: int) -> Iterable[dict]:
    """Resume at a source-row offset without materializing earlier rows."""
    if row_count <= 0:
        return rows

    skip_method = getattr(rows, "skip", None)
    if callable(skip_method):
        return skip_method(row_count)

    return islice(rows, row_count, None)


def load_next_text_chunk(
    row_iterator: Iterator[dict],
    text_column: str,
    chunk_size: int,
) -> tuple[list[str], int]:
    """Collect at most chunk_size non-empty documents from one shared iterator.

    Returns the accepted texts and the number of source rows consumed. The source
    row count can be larger than the text count when empty documents are skipped.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    texts: list[str] = []
    source_rows_consumed = 0

    while len(texts) < chunk_size:
        try:
            row = next(row_iterator)
        except StopIteration:
            break

        source_rows_consumed += 1

        if text_column not in row:
            available = ", ".join(sorted(row.keys()))
            raise KeyError(
                f"text column {text_column!r} was not found. "
                f"Available columns: {available}"
            )

        text = str(row[text_column]).strip()
        if text:
            texts.append(text)

    return texts, source_rows_consumed


if __name__ == "__main__":
    stream = load_pretraining_stream(
        dataset_name="toy",
        dataset_config=None,
        split="train",
        streaming=True,
        seed=42,
        shuffle_buffer=0,
    )
    iterator = iter(stream)
    first_chunk, consumed = load_next_text_chunk(iterator, "text", chunk_size=3)
    print(f"loaded={len(first_chunk)} consumed_rows={consumed}")
    print(first_chunk[0])

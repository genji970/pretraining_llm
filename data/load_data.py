from __future__ import annotations

from collections.abc import Iterable


TOY_TEXTS = [
    "A language model predicts the next token from previous tokens.",
    "Wikipedia articles are commonly used as one component of pretraining data.",
    "Causal attention prevents a token from reading future tokens.",
    "The optimizer updates model parameters using gradients from the loss.",
]


def _clean_texts(rows: Iterable[dict], text_column: str, max_samples: int) -> list[str]:
    texts: list[str] = []
    for row in rows:
        if text_column not in row:
            available = ", ".join(sorted(row.keys()))
            raise KeyError(
                f"text column {text_column!r} was not found. Available columns: {available}"
            )
        text = str(row[text_column]).strip()
        if text:
            texts.append(text)
        if len(texts) >= max_samples:
            break
    if not texts:
        raise ValueError("No non-empty text samples were loaded.")
    return texts


def load_pretraining_texts(
    dataset_name: str,
    dataset_config: str | None,
    split: str,
    text_column: str,
    max_samples: int,
    streaming: bool,
    seed: int,
    shuffle_buffer: int,
) -> list[str]:
    """Load a bounded list of documents for the current in-memory pipeline.

    `toy` is intentionally network-free and is used by module smoke tests.
    For Hugging Face datasets, imports happen lazily so the toy tests work even
    before `datasets` is installed.
    """
    if max_samples <= 0:
        raise ValueError("max_samples must be positive")

    if dataset_name == "toy":
        repeats = (max_samples + len(TOY_TEXTS) - 1) // len(TOY_TEXTS)
        return (TOY_TEXTS * repeats)[:max_samples]

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "The 'datasets' package is required for non-toy data. "
            "Install it with: pip install -r requirements.txt"
        ) from exc

    requested_split = split if streaming else f"{split}[:{max_samples}]"
    load_kwargs: dict[str, object] = {
        "split": requested_split,
        "streaming": streaming,
    }
    if dataset_config is None:
        dataset = load_dataset(dataset_name, **load_kwargs)
    else:
        dataset = load_dataset(dataset_name, dataset_config, **load_kwargs)

    if streaming:
        if shuffle_buffer > 0:
            dataset = dataset.shuffle(seed=seed, buffer_size=shuffle_buffer)
        return _clean_texts(dataset, text_column, max_samples)

    dataset = dataset.shuffle(seed=seed)
    row_count = min(max_samples, len(dataset))
    dataset = dataset.select(range(row_count))
    return _clean_texts(dataset, text_column, max_samples)


if __name__ == "__main__":
    samples = load_pretraining_texts(
        dataset_name="toy",
        dataset_config=None,
        split="train",
        text_column="text",
        max_samples=3,
        streaming=True,
        seed=42,
        shuffle_buffer=0,
    )
    print(f"loaded={len(samples)}")
    print(samples[0])

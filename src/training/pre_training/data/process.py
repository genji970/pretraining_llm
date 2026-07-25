from __future__ import annotations

import json
import tempfile
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from training.pre_training.data.format.base import RecordFormatter
from training.pre_training.data.format.mcqa import MCQAFormatter
from training.pre_training.data.format.plain_text import PlainTextFormatter
from training.pre_training.data.source.base import DataSource
from training.pre_training.data.source.huggingface import HuggingFaceSource
from training.pre_training.data.source.local_jsonl import LocalJsonlSource
from training.pre_training.data.storage.base import DataStorage
from training.pre_training.data.storage.local_jsonl import LocalJsonlStorage


def build_source(
    source_type: str,
    *,
    path: str | Path | None = None,
    dataset_name: str | None = None,
    split: str = "train",
    config_name: str | None = None,
    streaming: bool = False,
) -> DataSource:
    if source_type == "local_jsonl":
        if path is None:
            raise ValueError("path is required for local_jsonl")
        return LocalJsonlSource(path)
    if source_type == "huggingface":
        if not dataset_name:
            raise ValueError("dataset_name is required for huggingface")
        return HuggingFaceSource(dataset_name, split, config_name, streaming)
    raise ValueError(f"Unknown source_type: {source_type}")


def build_formatter(
    formatter_type: str,
    *,
    source_text_field: str = "text",
    target_text_field: str = "text",
) -> RecordFormatter:
    if formatter_type == "plain_text":
        return PlainTextFormatter(source_text_field, target_text_field)
    if formatter_type == "mcqa":
        return MCQAFormatter(target_field=target_text_field)
    raise ValueError(f"Unknown formatter_type: {formatter_type}")


def format_records(
    records: Iterable[dict[str, Any]],
    formatter: RecordFormatter,
    limit: int = 0,
) -> Iterator[dict[str, Any]]:
    for index, record in enumerate(records):
        if limit > 0 and index >= limit:
            break
        yield formatter.format(record)


def process_data(
    source: DataSource,
    formatter: RecordFormatter,
    storage: DataStorage,
    output_path: str | Path,
    limit: int = 0,
) -> int:
    return storage.write(
        output_path,
        format_records(source.records(), formatter, limit),
    )


if __name__ == "__main__":
    sample = {
        "question": "Which number is even?",
        "options": ["3", "4"],
        "answer_letter": "B",
        "answer": "4",
    }
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        raw_path = root / "raw.jsonl"
        processed_path = root / "process.jsonl"
        raw_path.write_text(json.dumps(sample) + "\n", encoding="utf-8")
        count = process_data(
            source=LocalJsonlSource(raw_path),
            formatter=MCQAFormatter(),
            storage=LocalJsonlStorage(),
            output_path=processed_path,
        )
        print("processed records:", count)
        print(processed_path.read_text(encoding="utf-8"))

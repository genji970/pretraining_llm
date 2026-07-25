from __future__ import annotations

import json
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from training.pre_training.data.source.base import DataSource


class LocalJsonlSource(DataSource):
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def records(self) -> Iterator[dict[str, Any]]:
        if not self.path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.path}")

        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid JSON at {self.path}:{line_number}: {error}"
                    ) from error
                if not isinstance(record, dict):
                    raise ValueError(
                        f"Each JSONL row must be an object: {self.path}:{line_number}"
                    )
                yield record


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "sample.jsonl"
        path.write_text(
            '{"text": "hello"}\n{"text": "world"}\n',
            encoding="utf-8",
        )
        print(list(LocalJsonlSource(path).records()))

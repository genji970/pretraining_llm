from __future__ import annotations

import json
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from training.pre_training.data.storage.base import DataStorage


class LocalJsonlStorage(DataStorage):
    def write(self, path: str | Path, records: Iterable[dict[str, Any]]) -> int:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with destination.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
        return count


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "sample.jsonl"
        count = LocalJsonlStorage().write(
            output,
            [{"text": "first"}, {"text": "second"}],
        )
        print("written:", count)
        print(output.read_text(encoding="utf-8"))

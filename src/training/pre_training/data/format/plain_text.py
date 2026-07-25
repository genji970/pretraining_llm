from __future__ import annotations

from typing import Any

from training.pre_training.data.format.base import RecordFormatter


class PlainTextFormatter(RecordFormatter):
    def __init__(self, source_field: str = "text", target_field: str = "text"):
        self.source_field = source_field
        self.target_field = target_field

    def format(self, record: dict[str, Any]) -> dict[str, Any]:
        if self.source_field not in record:
            raise KeyError(f"Missing text field: {self.source_field!r}")
        text = str(record[self.source_field]).strip()
        if not text:
            raise ValueError("The formatted text cannot be empty")
        return {self.target_field: text}


if __name__ == "__main__":
    formatter = PlainTextFormatter(source_field="body")
    print(formatter.format({"body": "A short pretraining document."}))

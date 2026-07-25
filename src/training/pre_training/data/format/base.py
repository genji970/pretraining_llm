from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class RecordFormatter(ABC):
    """Converts a source-specific row into a normalized pretraining record."""

    @abstractmethod
    def format(self, record: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError


if __name__ == "__main__":
    class IdentityFormatter(RecordFormatter):
        def format(self, record: dict[str, Any]) -> dict[str, Any]:
            return {"text": str(record["text"]).strip()}

    print(IdentityFormatter().format({"text": "  sample  "}))

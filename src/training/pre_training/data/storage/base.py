from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path
from typing import Any


class DataStorage(ABC):
    """Interface for processed-data destinations."""

    @abstractmethod
    def write(self, path: str | Path, records: Iterable[dict[str, Any]]) -> int:
        raise NotImplementedError


if __name__ == "__main__":
    class CountingStorage(DataStorage):
        def write(self, path: str | Path, records: Iterable[dict[str, Any]]) -> int:
            del path
            return sum(1 for _ in records)

    print(CountingStorage().write("unused", [{"text": "a"}, {"text": "b"}]))

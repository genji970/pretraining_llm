from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import Any


class DataSource(ABC):
    """Interface for local, remote, streaming, or database-backed datasets."""

    @abstractmethod
    def records(self) -> Iterator[dict[str, Any]]:
        raise NotImplementedError


if __name__ == "__main__":
    class InMemorySource(DataSource):
        def records(self) -> Iterator[dict[str, Any]]:
            yield {"text": "first example"}
            yield {"text": "second example"}

    print(list(InMemorySource().records()))

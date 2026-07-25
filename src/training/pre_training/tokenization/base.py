from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from pathlib import Path


class TokenizerBase(ABC):
    """Tokenizer interface kept independent from the model embedding space."""

    @abstractmethod
    def train(self, texts: Iterable[str]) -> None:
        raise NotImplementedError

    @abstractmethod
    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        raise NotImplementedError

    @abstractmethod
    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        raise NotImplementedError

    @abstractmethod
    def save(self, path: str | Path) -> None:
        raise NotImplementedError


if __name__ == "__main__":
    class CharacterTokenizer(TokenizerBase):
        def __init__(self) -> None:
            self.vocab = {"<unk>": 0}

        def train(self, texts: Iterable[str]) -> None:
            for character in sorted(set("".join(texts))):
                self.vocab.setdefault(character, len(self.vocab))

        def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
            del add_bos, add_eos
            return [self.vocab.get(character, 0) for character in text]

        def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
            del skip_special_tokens
            inverse = {value: key for key, value in self.vocab.items()}
            return "".join(inverse[index] for index in token_ids)

        def save(self, path: str | Path) -> None:
            Path(path).write_text(str(self.vocab), encoding="utf-8")

    tokenizer = CharacterTokenizer()
    tokenizer.train(["abc"])
    print(tokenizer.encode("cab"))
    print(tokenizer.decode(tokenizer.encode("cab")))

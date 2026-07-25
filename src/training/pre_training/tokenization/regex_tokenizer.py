from __future__ import annotations

import json
import re
import tempfile
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from training.pre_training.tokenization.base import TokenizerBase


SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<unk>"]


class RegexTokenizer(TokenizerBase):
    """Small extensible tokenizer used for the from-scratch implementation.

    This module intentionally owns only text-to-token-ID conversion. The model's
    vector-space representation lives in ``model/embedding.py``, allowing the
    tokenizer and embedding-space design to evolve independently.
    """

    TOKEN_PATTERN = re.compile(r"<[^>\s]+>|[^\W_]+(?:'[^\W_]+)?|[^\w\s]", re.UNICODE)

    def __init__(self, lowercase: bool = False):
        self.lowercase = lowercase
        self.token_to_id: dict[str, int] = {
            token: index for index, token in enumerate(SPECIAL_TOKENS)
        }
        self.id_to_token: list[str] = list(SPECIAL_TOKENS)
        self.is_trained = False

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_token)

    @property
    def pad_token_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_token_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_token_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def unk_token_id(self) -> int:
        return self.token_to_id["<unk>"]

    def tokenize(self, text: str) -> list[str]:
        normalized = str(text)
        if self.lowercase:
            normalized = normalized.lower()
        return self.TOKEN_PATTERN.findall(normalized)

    def train(
        self,
        texts: Iterable[str],
        min_frequency: int = 1,
        max_vocab_size: int = 0,
    ) -> None:
        if min_frequency < 1:
            raise ValueError("min_frequency must be positive")

        frequencies: Counter[str] = Counter()
        for text in texts:
            frequencies.update(self.tokenize(str(text)))

        candidates = [
            (token, frequency)
            for token, frequency in frequencies.items()
            if frequency >= min_frequency and token not in self.token_to_id
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))

        if max_vocab_size > 0:
            available = max(max_vocab_size - len(SPECIAL_TOKENS), 0)
            candidates = candidates[:available]

        self.token_to_id = {token: index for index, token in enumerate(SPECIAL_TOKENS)}
        self.id_to_token = list(SPECIAL_TOKENS)
        for token, _ in candidates:
            self.token_to_id[token] = len(self.id_to_token)
            self.id_to_token.append(token)
        self.is_trained = True

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True) -> list[int]:
        token_ids: list[int] = []
        if add_bos:
            token_ids.append(self.bos_token_id)
        token_ids.extend(
            self.token_to_id.get(token, self.unk_token_id)
            for token in self.tokenize(text)
        )
        if add_eos:
            token_ids.append(self.eos_token_id)
        return token_ids

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        tokens: list[str] = []
        for token_id in token_ids:
            index = int(token_id)
            if not 0 <= index < len(self.id_to_token):
                token = "<unk>"
            else:
                token = self.id_to_token[index]
            if skip_special_tokens and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)

        text = " ".join(tokens)
        text = re.sub(r"\s+([.,!?;:%)\]}])", r"\1", text)
        text = re.sub(r"([({\[])\s+", r"\1", text)
        return text.strip()

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "type": "regex",
            "lowercase": self.lowercase,
            "id_to_token": self.id_to_token,
        }
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "RegexTokenizer":
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(f"Tokenizer file not found: {source}")
        payload = json.loads(source.read_text(encoding="utf-8"))
        if payload.get("type") != "regex":
            raise ValueError(f"Unsupported tokenizer type: {payload.get('type')}")
        tokenizer = cls(lowercase=bool(payload.get("lowercase", False)))
        tokenizer.id_to_token = [str(token) for token in payload["id_to_token"]]
        tokenizer.token_to_id = {
            token: index for index, token in enumerate(tokenizer.id_to_token)
        }
        tokenizer.is_trained = True
        return tokenizer


if __name__ == "__main__":
    samples = [
        "Hello world!",
        "Hello from a tiny tokenizer.",
    ]
    tokenizer = RegexTokenizer(lowercase=True)
    tokenizer.train(samples)
    encoded = tokenizer.encode("Hello world!")
    print("vocab size:", tokenizer.vocab_size)
    print("encoded:", encoded)
    print("decoded:", tokenizer.decode(encoded))

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "tokenizer.json"
        tokenizer.save(path)
        restored = RegexTokenizer.load(path)
        print("restored:", restored.decode(encoded))

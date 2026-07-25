from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path


SPECIAL_TOKENS = [
    "<pad>",
    "<bos>",
    "<eos>",
    "<unk>",
    "<endoftext>",
    "<question>",
    "<choices>",
    "<answer>",
]


class Tokenizer:
    def __init__(
        self,
        data_name: str | None = None,
        dictionary: dict[str, int] | None = None,
        special_tokens: Sequence[str] | None = None,
    ) -> None:
        self.data_name = data_name
        self.dictionary = dict(dictionary) if dictionary is not None else {}
        self.special_tokens = list(special_tokens or SPECIAL_TOKENS)
        self.max_count = max(self.dictionary.values(), default=-1)
        self.special_token_pattern = "|".join(
            re.escape(token)
            for token in sorted(self.special_tokens, key=len, reverse=True)
        )

    def __len__(self) -> int:
        return len(self.dictionary)

    def tokenize(self, texts: Iterable[str]) -> list[str]:
        tokens_list: list[str] = []
        pattern = rf"""
            {self.special_token_pattern}
            |\\[A-Za-z]+
            |\n
            |[A-Za-z]+(?:'[A-Za-z]+)?
            |\d+(?:\.\d+)?(?:[eE][+-]?\d+)?
            |\S
        """
        for text in texts:
            text = re.sub(r"#+", " ", str(text))
            tokens_list.extend(re.findall(pattern, text, flags=re.VERBOSE))
        return tokens_list

    def tokenizer_encoding(self, text_list: Sequence[str]) -> dict[str, int]:
        """Compatibility method retained from the original notebook."""
        for token in [*self.special_tokens, *text_list]:
            if token not in self.dictionary:
                self.max_count += 1
                self.dictionary[token] = self.max_count
        return self.dictionary

    def build_vocab(
        self,
        texts: Iterable[str],
        max_vocab_size: int,
        min_frequency: int = 1,
    ) -> dict[str, int]:
        if max_vocab_size < len(self.special_tokens):
            raise ValueError(
                "max_vocab_size must be at least the number of special tokens"
            )
        if min_frequency < 0:
            raise ValueError("min_frequency must be non-negative")

        counts: Counter[str] = Counter()
        for text in texts:
            counts.update(self.tokenize([text]))

        self.dictionary = {}
        self.max_count = -1
        self.tokenizer_encoding(self.special_tokens)

        available_slots = max_vocab_size - len(self.dictionary)
        candidates = (
            (token, frequency)
            for token, frequency in counts.items()
            if frequency >= min_frequency and token not in self.dictionary
        )
        sorted_candidates = sorted(candidates, key=lambda item: (-item[1], item[0]))
        self.tokenizer_encoding([token for token, _ in sorted_candidates[:available_slots]])
        return self.dictionary

    def encode(
        self,
        text: str,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        if "<unk>" not in self.dictionary:
            raise ValueError("Build or load the vocabulary before calling encode().")

        unk_id = self.dictionary["<unk>"]
        token_ids = [
            self.dictionary.get(token, unk_id)
            for token in self.tokenize([text])
        ]
        if add_bos:
            token_ids.insert(0, self.dictionary["<bos>"])
        if add_eos:
            token_ids.append(self.dictionary["<eos>"])
        return token_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        inverse = {token_id: token for token, token_id in self.dictionary.items()}
        return " ".join(inverse.get(int(token_id), "<unk>") for token_id in token_ids)

    def inserting_special_token(self, special_token: str) -> None:
        if not isinstance(special_token, str):
            raise TypeError("special_token must be str")
        if special_token not in self.dictionary:
            self.max_count += 1
            self.dictionary[special_token] = self.max_count

    def print_max_num_encoder_dictionary(self) -> None:
        print(
            f"whole number of tokens: {len(self.dictionary)} / "
            f"max token index: {self.max_count}"
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "data_name": self.data_name,
            "dictionary": self.dictionary,
            "special_tokens": self.special_tokens,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            data_name=payload.get("data_name"),
            dictionary=payload["dictionary"],
            special_tokens=payload["special_tokens"],
        )


if __name__ == "__main__":
    toy_texts = ["A tiny tokenizer test.", "A second tiny test!"]
    tokenizer = Tokenizer(data_name="toy")
    tokenizer.build_vocab(toy_texts, max_vocab_size=64, min_frequency=1)
    encoded = tokenizer.encode(toy_texts[0], add_bos=True, add_eos=True)
    print(f"vocab_size={len(tokenizer)}")
    print(f"encoded={encoded}")
    print(f"decoded={tokenizer.decode(encoded)}")

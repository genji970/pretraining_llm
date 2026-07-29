from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

from tokenizers import Tokenizer as BackendTokenizer
from tokenizers import decoders, models, pre_tokenizers, trainers


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
        backend: BackendTokenizer,
        data_name: str | None = None,
    ) -> None:
        self.data_name = data_name
        self.backend = backend

    @classmethod
    def create_init_token(cls, data_name: str | None=None,) -> "Tokenizer":
        backend=BackendTokenizer(models.BPE(unk_token="<unk>"))

        backend.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False,use_regex=True,)
        # add_prefix_space=False means do not add imaginary blank space. It makes Hello from Hello world and world Hello different. Since first one is 'Hello' and second one is ' Hello'
        # use_regex=True, Before ByteLevel transformation, dividing text to word,number,blank,etc using GPT-2 normal expression.

        backend.decoder=decoders.ByteLevel()
        return cls(
            backend=backend,
            data_name=data_name,
        )

    @classmethod
    def train(cls,texts: Iterable[str] | Iterable[list[str]], vocab_size: int, min_frequency: int, data_name: str | None=None, length: int | None=None,) -> "Tokenizer":
        tokenizer=cls.create_init_token(data_name=data_name)

        trainer=trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency, # min_frequency : 
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=True,
        )

        tokenizer.backend.trainer(
            texts,
            trainer=trainer,
            length=length,
        )
        return tokenizer
    
    def __len__(self) -> int:
        return self.backend.get_vocab_size()

    @property
    def dictionary(self) -> dic[str, int]:
        return self.backend.get_vocab()
    
    def token_to_id(self, token: str) -> int:
        token_id = self.backend.token_to_id(token)
        if token_id is None:
            raise KeyError(f"Token not found: {token!r}")

        return int(token_id)
    
    @property
    def pad_token_id(self) -> int:
        return self.token_to_id()


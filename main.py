from __future__ import annotations

import json
import random
from pathlib import Path

import torch

from config import TrainConfig, parse_config
from data.dataset import Train_Dataset, create_dataloader
from data.load_data import load_pretraining_texts
from data.tokenizer import Tokenizer
from model.model import DecoderLanguageModel
from train.trainer import PretrainingTrainer


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_or_load_tokenizer(config: TrainConfig, texts: list[str]) -> Tokenizer:
    if config.vocab_path is not None and Path(config.vocab_path).exists():
        tokenizer = Tokenizer.load(config.vocab_path)
        print(f"Loaded tokenizer from {config.vocab_path}")
        return tokenizer

    tokenizer = Tokenizer(data_name=config.dataset_name)
    tokenizer.build_vocab(
        texts=texts,
        max_vocab_size=config.max_vocab_size,
        min_frequency=config.min_token_frequency,
    )
    save_path = (
        Path(config.vocab_path)
        if config.vocab_path is not None
        else Path(config.output_dir) / "tokenizer.json"
    )
    tokenizer.save(save_path)
    print(f"Built tokenizer with vocab_size={len(tokenizer)} and saved it to {save_path}")
    return tokenizer


def run(config: TrainConfig) -> dict[str, float | int | str]:
    set_seed(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(config.to_dict(), indent=2), encoding="utf-8"
    )

    texts = load_pretraining_texts(
        dataset_name=config.dataset_name,
        dataset_config=config.dataset_config,
        split=config.dataset_split,
        text_column=config.text_column,
        max_samples=config.max_samples,
        streaming=config.streaming,
        seed=config.seed,
        shuffle_buffer=config.shuffle_buffer,
    )
    print(f"Loaded {len(texts)} documents")

    tokenizer = build_or_load_tokenizer(config, texts)
    train_dataset = Train_Dataset(
        texts=texts,
        tokenizer=tokenizer,
        context_length=config.context_length,
    )
    train_loader = create_dataloader(
        dataset=train_dataset,
        batch_size=config.batch_size,
        pad_token_id=tokenizer.dictionary["<pad>"],
        shuffle=True,
        num_workers=config.num_workers,
    )

    # vocab_size is derived exactly once from the tokenizer; no separate config
    # value can silently disagree with the tokenizer dictionary.
    model = DecoderLanguageModel(
        vocab_size=len(tokenizer),
        block_num=config.block_num,
        embed_dim=config.embed_dim,
        context_length=config.context_length,
        num_head=config.num_heads,
        dropout=config.dropout,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"sequences={len(train_dataset)} batches={len(train_loader)} "
        f"parameters={parameter_count:,}"
    )

    trainer = PretrainingTrainer(model=model, train_loader=train_loader, config=config)
    if config.resume_from is not None:
        trainer.load_checkpoint(config.resume_from)
        print(f"Resumed from {config.resume_from} at step={trainer.global_step}")

    return trainer.train()


if __name__ == "__main__":
    run_config = parse_config()
    print(json.dumps(run_config.to_dict(), indent=2))
    result = run(run_config)
    print(json.dumps(result, indent=2))

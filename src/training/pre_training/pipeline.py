from __future__ import annotations

import json
import math
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.utils.data import DataLoader

from training.pre_training.config import ProjectConfig
from training.pre_training.data.format.mcqa import MCQAFormatter
from training.pre_training.data.format.plain_text import PlainTextFormatter
from training.pre_training.data.process import build_source, process_data
from training.pre_training.data.source.local_jsonl import LocalJsonlSource
from training.pre_training.data.storage.local_jsonl import LocalJsonlStorage
from training.pre_training.dataset.causal_lm import CausalLMDataset
from training.pre_training.dataset.collator import CausalLMCollator
from training.pre_training.model.language_model import DecoderLanguageModel
from training.pre_training.scheduler import build_scheduler
from training.pre_training.tokenization.regex_tokenizer import RegexTokenizer
from training.pre_training.trainer import PretrainingTrainer, TrainerState
from training.pre_training.utils import select_device, set_seed


@dataclass
class PretrainingResult:
    tokenizer: RegexTokenizer
    model: DecoderLanguageModel
    trainer: PretrainingTrainer
    state: TrainerState


def _formatter(config: ProjectConfig):
    if config.data.formatter == "plain_text":
        return PlainTextFormatter(
            source_field=config.data.source_text_field,
            target_field=config.data.processed_text_field,
        )
    if config.data.formatter == "mcqa":
        return MCQAFormatter(target_field=config.data.processed_text_field)
    raise ValueError(f"Unknown formatter: {config.data.formatter}")


def _source(config: ProjectConfig, evaluation: bool):
    if config.data.source_type == "local_jsonl":
        path = config.data.eval_path if evaluation else config.data.train_path
        if path is None:
            return None
        return build_source("local_jsonl", path=path)

    split = config.data.eval_split if evaluation else config.data.train_split
    if split is None:
        return None
    return build_source(
        "huggingface",
        dataset_name=config.data.dataset_name,
        split=split,
        config_name=config.data.dataset_config_name,
        streaming=config.data.streaming,
    )


def prepare_processed_data(config: ProjectConfig) -> tuple[int, int]:
    formatter = _formatter(config)
    storage = LocalJsonlStorage()

    train_source = _source(config, evaluation=False)
    if train_source is None:
        raise ValueError("A training source is required")
    if config.data.overwrite_processed or not config.data.processed_train_path.exists():
        train_count = process_data(
            train_source,
            formatter,
            storage,
            config.data.processed_train_path,
            config.data.max_train_samples,
        )
    else:
        train_count = sum(1 for _ in LocalJsonlSource(config.data.processed_train_path).records())

    eval_count = 0
    eval_source = _source(config, evaluation=True)
    if eval_source is not None:
        if config.data.overwrite_processed or not config.data.processed_eval_path.exists():
            eval_count = process_data(
                eval_source,
                formatter,
                storage,
                config.data.processed_eval_path,
                config.data.max_eval_samples,
            )
        else:
            eval_count = sum(1 for _ in LocalJsonlSource(config.data.processed_eval_path).records())
    return train_count, eval_count


def processed_texts(path: str | Path, text_field: str) -> Iterator[str]:
    for record in LocalJsonlSource(path).records():
        if text_field not in record:
            raise KeyError(f"Processed record is missing {text_field!r}")
        text = str(record[text_field]).strip()
        if text:
            yield text


def build_or_load_tokenizer(config: ProjectConfig) -> RegexTokenizer:
    tokenizer_path = config.tokenizer.path
    if not config.tokenizer.retrain and tokenizer_path.exists():
        return RegexTokenizer.load(tokenizer_path)
    if config.tokenizer.tokenizer_type != "regex":
        raise ValueError("Only the regex tokenizer is currently implemented")

    tokenizer = RegexTokenizer(lowercase=config.tokenizer.lowercase)
    tokenizer.train(
        processed_texts(
            config.data.processed_train_path,
            config.data.processed_text_field,
        ),
        min_frequency=config.tokenizer.min_frequency,
        max_vocab_size=config.tokenizer.max_vocab_size,
    )
    tokenizer.save(tokenizer_path)
    return tokenizer


def _serialize_config(config: ProjectConfig) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        return value

    return convert(asdict(config))


def build_training_components(
    config: ProjectConfig,
    device: torch.device | None = None,
) -> tuple[RegexTokenizer, DecoderLanguageModel, PretrainingTrainer]:
    config.validate()
    set_seed(config.trainer.seed)
    device = device or select_device()
    tokenizer = build_or_load_tokenizer(config)

    train_dataset = CausalLMDataset(
        processed_texts(
            config.data.processed_train_path,
            config.data.processed_text_field,
        ),
        tokenizer,
        max_seq_len=config.model.max_seq_len,
        mode=config.data.sequence_mode,
    )
    eval_dataset = None
    evaluation_is_configured = (
        config.data.eval_path is not None
        if config.data.source_type == "local_jsonl"
        else config.data.eval_split is not None
    )
    if evaluation_is_configured and config.data.processed_eval_path.exists():
        eval_texts = list(
            processed_texts(
                config.data.processed_eval_path,
                config.data.processed_text_field,
            )
        )
        if eval_texts:
            eval_dataset = CausalLMDataset(
                eval_texts,
                tokenizer,
                max_seq_len=config.model.max_seq_len,
                mode=config.data.sequence_mode,
            )

    collator = CausalLMCollator(tokenizer.pad_token_id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.trainer.batch_size,
        shuffle=True,
        num_workers=config.trainer.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collator,
    )
    eval_loader = (
        DataLoader(
            eval_dataset,
            batch_size=config.trainer.eval_batch_size,
            shuffle=False,
            num_workers=config.trainer.num_workers,
            pin_memory=device.type == "cuda",
            collate_fn=collator,
        )
        if eval_dataset is not None
        else None
    )

    model = DecoderLanguageModel(
        vocab_size=tokenizer.vocab_size,
        d_model=config.model.d_model,
        n_heads=config.model.n_heads,
        n_layers=config.model.n_layers,
        ffn_multiplier=config.model.ffn_multiplier,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout,
        tie_embeddings=config.model.tie_embeddings,
        embedding_type=config.model.embedding_type,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.trainer.learning_rate,
        weight_decay=config.trainer.weight_decay,
    )
    steps_per_epoch = math.ceil(
        len(train_loader) / config.trainer.gradient_accumulation_steps
    )
    total_steps = config.trainer.max_steps or steps_per_epoch * config.trainer.epochs
    scheduler = build_scheduler(
        optimizer,
        config.trainer.scheduler,
        config.trainer.warmup_steps,
        total_steps,
    )
    trainer = PretrainingTrainer(
        model,
        optimizer,
        scheduler,
        train_loader,
        eval_loader,
        device,
        config.trainer.output_dir,
        epochs=config.trainer.epochs,
        max_steps=config.trainer.max_steps,
        gradient_accumulation_steps=config.trainer.gradient_accumulation_steps,
        max_grad_norm=config.trainer.max_grad_norm,
        log_every_steps=config.trainer.log_every_steps,
        eval_every_steps=config.trainer.eval_every_steps,
        save_every_steps=config.trainer.save_every_steps,
        use_amp=config.trainer.use_amp,
    )
    return tokenizer, model, trainer


def run_pretraining(
    config: ProjectConfig,
    device: torch.device | None = None,
) -> PretrainingResult:
    config.validate()
    train_count, eval_count = prepare_processed_data(config)
    config.trainer.output_dir.mkdir(parents=True, exist_ok=True)
    (config.trainer.output_dir / "run_config.json").write_text(
        json.dumps(_serialize_config(config), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    tokenizer, model, trainer = build_training_components(config, device)
    print(
        json.dumps(
            {
                "train_records": train_count,
                "eval_records": eval_count,
                "device": str(trainer.device),
                "vocab_size": tokenizer.vocab_size,
                "parameters": model.parameter_count(),
            }
        )
    )
    if config.trainer.resume_from is not None:
        trainer.resume(config.trainer.resume_from)
    state = trainer.train()
    return PretrainingResult(tokenizer, model, trainer, state)


if __name__ == "__main__":
    from training.pre_training.config import DataConfig, ModelConfig, TokenizerConfig, TrainerConfig

    samples = [
        {"text": "the cat sits on the mat"},
        {"text": "the dog sits near the cat"},
        {"text": "a small model predicts the next token"},
        {"text": "pretraining uses causal language modeling"},
    ]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        raw = root / "train.jsonl"
        raw.write_text(
            "".join(json.dumps(row) + "\n" for row in samples),
            encoding="utf-8",
        )
        config = ProjectConfig(
            data=DataConfig(
                train_path=raw,
                eval_path=raw,
                processed_train_path=root / "process_train.jsonl",
                processed_eval_path=root / "process_eval.jsonl",
            ),
            tokenizer=TokenizerConfig(path=root / "tokenizer.json"),
            model=ModelConfig(
                d_model=16,
                n_heads=4,
                n_layers=1,
                ffn_multiplier=2,
                max_seq_len=16,
            ),
            trainer=TrainerConfig(
                output_dir=root / "artifacts",
                batch_size=2,
                eval_batch_size=2,
                epochs=2,
                max_steps=2,
                learning_rate=1e-3,
                log_every_steps=1,
                eval_every_steps=1,
            ),
        )
        result = run_pretraining(config, device=torch.device("cpu"))
        print("finished at step:", result.state.global_step)

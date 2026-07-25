from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import torch

from training.pre_training.config import (
    DataConfig,
    ModelConfig,
    ProjectConfig,
    TokenizerConfig,
    TrainerConfig,
)
from training.pre_training.generation import generate
from training.pre_training.pipeline import run_pretraining


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run decoder-only language-model pretraining from scratch."
    )

    data = parser.add_argument_group("data")
    data.add_argument("--source", choices=["local_jsonl", "huggingface"], default="local_jsonl")
    data.add_argument("--train-path", type=Path, default=Path("data/raw/train.jsonl"))
    data.add_argument("--eval-path", type=Path)
    data.add_argument("--dataset-name")
    data.add_argument("--dataset-config-name")
    data.add_argument("--train-split", default="train")
    data.add_argument("--eval-split")
    data.add_argument("--streaming", action="store_true")
    data.add_argument("--formatter", choices=["plain_text", "mcqa"], default="plain_text")
    data.add_argument("--source-text-field", default="text")
    data.add_argument("--processed-train-path", type=Path, default=Path("data/process/train.jsonl"))
    data.add_argument("--processed-eval-path", type=Path, default=Path("data/process/eval.jsonl"))
    data.add_argument("--max-train-samples", type=int, default=0)
    data.add_argument("--max-eval-samples", type=int, default=0)
    data.add_argument("--sequence-mode", choices=["document", "packed"], default="document")
    data.add_argument("--reuse-processed", action="store_true")

    tokenizer = parser.add_argument_group("tokenizer")
    tokenizer.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path("artifacts/pretraining/tokenizer.json"),
    )
    tokenizer.add_argument("--lowercase", action="store_true")
    tokenizer.add_argument("--min-frequency", type=int, default=1)
    tokenizer.add_argument("--max-vocab-size", type=int, default=0)
    tokenizer.add_argument("--reuse-tokenizer", action="store_true")

    model = parser.add_argument_group("model")
    model.add_argument("--d-model", type=int, default=128)
    model.add_argument("--n-heads", type=int, default=4)
    model.add_argument("--n-layers", type=int, default=4)
    model.add_argument("--ffn-multiplier", type=int, default=4)
    model.add_argument("--max-seq-len", type=int, default=256)
    model.add_argument("--dropout", type=float, default=0.0)
    model.add_argument("--no-tie-embeddings", action="store_true")

    trainer = parser.add_argument_group("trainer")
    trainer.add_argument("--output-dir", type=Path, default=Path("artifacts/pretraining"))
    trainer.add_argument("--seed", type=int, default=42)
    trainer.add_argument("--batch-size", type=int, default=8)
    trainer.add_argument("--eval-batch-size", type=int, default=8)
    trainer.add_argument("--epochs", type=int, default=1)
    trainer.add_argument("--max-steps", type=int, default=0)
    trainer.add_argument("--learning-rate", type=float, default=3e-4)
    trainer.add_argument("--weight-decay", type=float, default=0.1)
    trainer.add_argument("--warmup-steps", type=int, default=0)
    trainer.add_argument("--scheduler", choices=["constant", "linear", "cosine"], default="cosine")
    trainer.add_argument("--gradient-accumulation-steps", type=int, default=1)
    trainer.add_argument("--max-grad-norm", type=float, default=1.0)
    trainer.add_argument("--log-every-steps", type=int, default=10)
    trainer.add_argument("--eval-every-steps", type=int, default=0)
    trainer.add_argument("--save-every-steps", type=int, default=0)
    trainer.add_argument("--num-workers", type=int, default=0)
    trainer.add_argument("--amp", action="store_true")
    trainer.add_argument("--resume-from", type=Path)

    parser.add_argument("--generate-prompt")
    parser.add_argument("--generate-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=0)
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a temporary two-step end-to-end test without creating a toy script or dataset file.",
    )
    return parser


def config_from_args(args: argparse.Namespace) -> ProjectConfig:
    config = ProjectConfig(
        data=DataConfig(
            source_type=args.source,
            train_path=args.train_path,
            eval_path=args.eval_path,
            dataset_name=args.dataset_name,
            dataset_config_name=args.dataset_config_name,
            train_split=args.train_split,
            eval_split=args.eval_split,
            streaming=args.streaming,
            formatter=args.formatter,
            source_text_field=args.source_text_field,
            processed_train_path=args.processed_train_path,
            processed_eval_path=args.processed_eval_path,
            max_train_samples=args.max_train_samples,
            max_eval_samples=args.max_eval_samples,
            sequence_mode=args.sequence_mode,
            overwrite_processed=not args.reuse_processed,
        ),
        tokenizer=TokenizerConfig(
            path=args.tokenizer_path,
            lowercase=args.lowercase,
            min_frequency=args.min_frequency,
            max_vocab_size=args.max_vocab_size,
            retrain=not args.reuse_tokenizer,
        ),
        model=ModelConfig(
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            ffn_multiplier=args.ffn_multiplier,
            max_seq_len=args.max_seq_len,
            dropout=args.dropout,
            tie_embeddings=not args.no_tie_embeddings,
        ),
        trainer=TrainerConfig(
            output_dir=args.output_dir,
            seed=args.seed,
            batch_size=args.batch_size,
            eval_batch_size=args.eval_batch_size,
            epochs=args.epochs,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            scheduler=args.scheduler,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            max_grad_norm=args.max_grad_norm,
            log_every_steps=args.log_every_steps,
            eval_every_steps=args.eval_every_steps,
            save_every_steps=args.save_every_steps,
            num_workers=args.num_workers,
            use_amp=args.amp,
            resume_from=args.resume_from,
        ),
    )
    config.resolve_paths(PROJECT_ROOT)
    config.validate()
    return config


def run_smoke_test(args: argparse.Namespace) -> None:
    rows = [
        {"text": "the cat sits on the mat"},
        {"text": "the dog sits near the cat"},
        {"text": "a decoder predicts the next token"},
        {"text": "pretraining uses causal language modeling"},
    ]
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        raw_path = root / "train.jsonl"
        raw_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        args.source = "local_jsonl"
        args.train_path = raw_path
        args.eval_path = raw_path
        args.processed_train_path = root / "process" / "train.jsonl"
        args.processed_eval_path = root / "process" / "eval.jsonl"
        args.tokenizer_path = root / "artifacts" / "tokenizer.json"
        args.output_dir = root / "artifacts"
        args.d_model = 16
        args.n_heads = 4
        args.n_layers = 1
        args.ffn_multiplier = 2
        args.max_seq_len = 16
        args.batch_size = 2
        args.eval_batch_size = 2
        args.epochs = 2
        args.max_steps = 2
        args.learning_rate = 1e-3
        args.log_every_steps = 1
        args.eval_every_steps = 1
        args.save_every_steps = 1
        config = config_from_args(args)
        result = run_pretraining(config, device=torch.device("cpu"))
        print("smoke-test global step:", result.state.global_step)
        print("checkpoint exists:", (config.trainer.output_dir / "last.pt").exists())


def main() -> None:
    args = build_parser().parse_args()
    if args.smoke_test:
        run_smoke_test(args)
        return

    config = config_from_args(args)
    result = run_pretraining(config)
    if args.generate_prompt:
        text = generate(
            result.model,
            result.tokenizer,
            args.generate_prompt,
            result.trainer.device,
            config.model.max_seq_len,
            max_new_tokens=args.generate_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        print(text)


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from typing import Sequence

@dataclass(frozen=True)
class TrainConfig:
    dataset_name: str
    dataset_config: str | None
    dataset_split: str
    text_column: str

    total_documents: int
    chunk_size: int

    streaming: bool
    shuffle_buffer: int
    seed: int

    tokenizer_name: str
    output_dir: str

    context_length: int
    batch_size: int
    num_workers: int

    block_num: int
    embed_dim: int
    num_heads: int
    dropout: float

    # Training
    epochs: int
    learning_rate: float
    weight_decay: float
    max_grad_norm: float
    log_every_steps: int
    max_steps: int
    early_stop_step: int
    device: str
    resume_from: str | None

    # Automatic early stopping
    early_stop_metric: str
    early_stop_mode: str
    early_stop_threshold: float | None
    early_stop_patience_steps: int
    early_stop_min_delta: float
    early_stop_warmup_steps: int

    # Checkpointing
    save_every_steps: int
    save_each_chunk: bool
    save_optimizer_state: bool

    save_best_checkpoint: bool
    best_checkpoint_metric: str
    best_checkpoint_mode: str
    best_checkpoint_min_delta: float

    save_threshold_checkpoint: bool
    threshold_checkpoint_metric: str
    threshold_checkpoint_value: float | None
    threshold_checkpoint_mode: str
    threshold_checkpoint_once: bool

    # Metrics and visualization
    metrics_every_steps: int
    metric_smoothing_window: int
    jsonl_metrics: bool
    tensorboard: bool
    save_plots: bool
    plot_every_steps: int
    plot_metrics: str
    max_plot_points: int

    log_parameter_norm: bool
    histogram_every_steps: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the from-scratch decoder-only language model."
    )

    data = parser.add_argument_group("data")
    data.add_argument("--dataset_name", type=str, default="wikimedia/wikipedia")
    data.add_argument("--dataset_config", type=str, default="20231101.en")
    data.add_argument("--dataset_split", type=str, default="train")
    data.add_argument("--text_column", type=str, default="text")
    data.add_argument(
        "--max_samples",
        type=int,
        default=10_000,
        help="Number of documents to materialize from the dataset.",
    )
    data.add_argument(
        "--streaming",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use Hugging Face streaming. Disable with --no-streaming.",
    )
    data.add_argument("--shuffle_buffer", type=int, default=10_000)
    data.add_argument("--seed", type=int, default=42)

    tokenizer = parser.add_argument_group("tokenizer")
    tokenizer.add_argument("--vocab_path", type=str, default=None)
    tokenizer.add_argument("--max_vocab_size", type=int, default=30_000)
    tokenizer.add_argument("--min_token_frequency", type=int, default=2)

    model = parser.add_argument_group("model")
    model.add_argument("--context_length", type=int, default=256)
    model.add_argument("--block_num", type=int, default=8)
    model.add_argument("--embed_dim", type=int, default=512)
    model.add_argument("--num_heads", type=int, default=8)
    model.add_argument("--dropout", type=float, default=0.1)

        train = parser.add_argument_group("training")

    train.add_argument(
        "--output_dir",
        type=str,
        default="outputs/fineweb_pretrain",
    )
    train.add_argument("--batch_size", type=int, default=8)
    train.add_argument("--num_workers", type=int, default=0)
    train.add_argument("--epochs", type=int, default=1)
    train.add_argument("--learning_rate", type=float, default=2e-4)
    train.add_argument("--weight_decay", type=float, default=0.01)
    train.add_argument("--max_grad_norm", type=float, default=1.0)

    train.add_argument(
        "--log_every_steps",
        type=int,
        default=10,
        help="Print training status every N optimizer steps.",
    )

    train.add_argument(
        "--max_steps",
        type=int,
        default=0,
        help=(
            "Absolute maximum number of optimizer steps across all chunks. "
            "0 means no step limit."
        ),
    )

    train.add_argument(
        "--early_stop_step",
        type=int,
        default=0,
        help=(
            "Manually terminate training at this global step. "
            "Useful for short test runs. 0 disables it."
        ),
    )

    train.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
    )

    train.add_argument(
        "--resume_from",
        type=str,
        default=None,
    )

    return parser


def _validate(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    positive_names = (
        "max_samples",
        "max_vocab_size",
        "context_length",
        "batch_size",
        "block_num",
        "embed_dim",
        "num_heads",
        "epochs",
        "log_every",
    )
    for name in positive_names:
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")

    nonnegative_names = (
        "shuffle_buffer",
        "num_workers",
        "min_token_frequency",
        "save_every",
        "max_steps",
        "weight_decay",
        "dropout",
    )
    for name in nonnegative_names:
        if getattr(args, name) < 0:
            parser.error(f"--{name} must be non-negative")

    if args.embed_dim % args.num_heads != 0:
        parser.error("--embed_dim must be divisible by --num_heads")
    if (args.embed_dim // args.num_heads) % 2 != 0:
        parser.error("head_dim = embed_dim / num_heads must be even for RoPE")
    if not 0.0 <= args.dropout < 1.0:
        parser.error("--dropout must satisfy 0 <= dropout < 1")
    if args.learning_rate <= 0:
        parser.error("--learning_rate must be positive")
    if args.max_grad_norm <= 0:
        parser.error("--max_grad_norm must be positive")


def parse_config(argv: Sequence[str] | None = None) -> TrainConfig:
    parser = build_parser()
    namespace = parser.parse_args(argv)
    _validate(namespace, parser)

    # Empty string is treated as no config, but no later module mutates values.
    if namespace.dataset_config == "":
        namespace.dataset_config = None
    if namespace.vocab_path == "":
        namespace.vocab_path = None
    if namespace.resume_from == "":
        namespace.resume_from = None

    return TrainConfig(**vars(namespace))


if __name__ == "__main__":
    toy_config = parse_config(
        [
            "--dataset_name",
            "toy",
            "--dataset_config",
            "",
            "--max_samples",
            "4",
            "--block_num",
            "2",
            "--embed_dim",
            "32",
            "--num_heads",
            "4",
        ]
    )
    print(toy_config)

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DataConfig:
    source_type: str = "local_jsonl"
    train_path: Path = Path("data/raw/train.jsonl")
    eval_path: Path | None = None
    dataset_name: str | None = None
    dataset_config_name: str | None = None
    train_split: str = "train"
    eval_split: str | None = None
    streaming: bool = False
    formatter: str = "plain_text"
    source_text_field: str = "text"
    processed_text_field: str = "text"
    processed_train_path: Path = Path("data/process/train.jsonl")
    processed_eval_path: Path = Path("data/process/eval.jsonl")
    max_train_samples: int = 0
    max_eval_samples: int = 0
    sequence_mode: str = "document"
    overwrite_processed: bool = True


@dataclass
class TokenizerConfig:
    tokenizer_type: str = "regex"
    path: Path = Path("artifacts/pretraining/tokenizer.json")
    lowercase: bool = False
    min_frequency: int = 1
    max_vocab_size: int = 0
    retrain: bool = True


@dataclass
class ModelConfig:
    embedding_type: str = "learned"
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ffn_multiplier: int = 4
    max_seq_len: int = 256
    dropout: float = 0.0
    tie_embeddings: bool = True


@dataclass
class TrainerConfig:
    output_dir: Path = Path("artifacts/pretraining")
    seed: int = 42
    batch_size: int = 8
    eval_batch_size: int = 8
    epochs: int = 1
    max_steps: int = 0
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 0
    scheduler: str = "cosine"
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    log_every_steps: int = 10
    eval_every_steps: int = 0
    save_every_steps: int = 0
    num_workers: int = 0
    use_amp: bool = False
    resume_from: Path | None = None


@dataclass
class ProjectConfig:
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def resolve_paths(self, project_root: str | Path) -> None:
        root = Path(project_root).expanduser().resolve()
        path_fields = [
            (self.data, "train_path"),
            (self.data, "eval_path"),
            (self.data, "processed_train_path"),
            (self.data, "processed_eval_path"),
            (self.tokenizer, "path"),
            (self.trainer, "output_dir"),
            (self.trainer, "resume_from"),
        ]
        for owner, name in path_fields:
            value = getattr(owner, name)
            if value is None:
                continue
            path = Path(value).expanduser()
            setattr(owner, name, path if path.is_absolute() else (root / path).resolve())

    def validate(self) -> None:
        if self.data.source_type not in {"local_jsonl", "huggingface"}:
            raise ValueError("source_type must be 'local_jsonl' or 'huggingface'")
        if self.data.formatter not in {"plain_text", "mcqa"}:
            raise ValueError("formatter must be 'plain_text' or 'mcqa'")
        if self.data.sequence_mode not in {"document", "packed"}:
            raise ValueError("sequence_mode must be 'document' or 'packed'")
        if self.data.source_type == "huggingface" and not self.data.dataset_name:
            raise ValueError("dataset_name is required for a Hugging Face source")
        if self.model.d_model % self.model.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        head_dim = self.model.d_model // self.model.n_heads
        if head_dim % 2 != 0:
            raise ValueError("The per-head dimension must be even for RoPE")
        if self.model.max_seq_len < 2:
            raise ValueError("max_seq_len must be at least 2")
        if self.trainer.gradient_accumulation_steps < 1:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.trainer.batch_size < 1 or self.trainer.eval_batch_size < 1:
            raise ValueError("batch sizes must be positive")


if __name__ == "__main__":
    config = ProjectConfig()
    config.resolve_paths(Path.cwd())
    config.validate()
    print(config)

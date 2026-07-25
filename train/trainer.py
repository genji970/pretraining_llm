from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from config import TrainConfig
except ImportError:  # pragma: no cover
    TrainConfig = object  # type: ignore[misc,assignment]


class PretrainingTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        config: TrainConfig,
    ) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._resolve_device(config.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        self.loss_function = nn.CrossEntropyLoss(ignore_index=-100)
        self.global_step = 0
        self.start_epoch = 0

    @staticmethod
    def _resolve_device(requested: str) -> torch.device:
        if requested == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
        return torch.device(requested)

    def save_checkpoint(self, name: str) -> Path:
        checkpoint_path = self.output_dir / name
        payload = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self.global_step,
            "start_epoch": self.start_epoch,
            "config": self.config.to_dict(),
        }
        torch.save(payload, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str | Path) -> None:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.global_step = int(checkpoint.get("global_step", 0))
        self.start_epoch = int(checkpoint.get("start_epoch", 0))

    def train(self) -> dict[str, float | int | str]:
        self.model.train()
        last_loss = float("nan")
        stop_training = False

        for epoch in range(self.start_epoch, self.config.epochs):
            self.start_epoch = epoch
            for batch in self.train_loader:
                input_ids = batch["input_ids"].to(self.device)
                labels = batch["labels"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                logits = self.model(input_ids, attention_mask=attention_mask)
                loss = self.loss_function(
                    logits.reshape(-1, logits.size(-1)),
                    labels.reshape(-1),
                )
                loss.backward()
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=self.config.max_grad_norm,
                )
                self.optimizer.step()

                self.global_step += 1
                last_loss = float(loss.item())

                if self.global_step % self.config.log_every == 0:
                    print(
                        f"epoch={epoch + 1}/{self.config.epochs} "
                        f"step={self.global_step} "
                        f"loss={last_loss:.4f} "
                        f"grad_norm={float(gradient_norm):.4f}"
                    )

                if (
                    self.config.save_every > 0
                    and self.global_step % self.config.save_every == 0
                ):
                    self.save_checkpoint(f"checkpoint-step-{self.global_step}.pt")

                if (
                    self.config.max_steps > 0
                    and self.global_step >= self.config.max_steps
                ):
                    stop_training = True
                    break

            if stop_training:
                break

        self.start_epoch += 1
        final_checkpoint = self.save_checkpoint("final_checkpoint.pt")
        summary = {
            "global_step": self.global_step,
            "last_loss": last_loss,
            "device": str(self.device),
            "final_checkpoint": str(final_checkpoint),
        }
        (self.output_dir / "train_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        return summary


if __name__ == "__main__":
    import tempfile

    from config import parse_config
    from data.dataset import Train_Dataset, create_dataloader
    from data.tokenizer import Tokenizer
    from model.model import DecoderLanguageModel

    texts = ["a b c d e", "a b c f g"]
    tokenizer = Tokenizer(data_name="toy")
    tokenizer.build_vocab(texts, max_vocab_size=32, min_frequency=1)
    dataset = Train_Dataset(texts, tokenizer, context_length=4)
    loader = create_dataloader(
        dataset,
        batch_size=2,
        pad_token_id=tokenizer.dictionary["<pad>"],
        shuffle=False,
    )
    with tempfile.TemporaryDirectory() as output_dir:
        config = parse_config(
            [
                "--dataset_name",
                "toy",
                "--dataset_config",
                "",
                "--max_samples",
                "2",
                "--output_dir",
                output_dir,
                "--context_length",
                "4",
                "--batch_size",
                "2",
                "--block_num",
                "1",
                "--embed_dim",
                "16",
                "--num_heads",
                "2",
                "--epochs",
                "1",
                "--max_steps",
                "2",
                "--log_every",
                "1",
                "--device",
                "cpu",
            ]
        )
        model = DecoderLanguageModel(
            vocab_size=len(tokenizer),
            block_num=config.block_num,
            embed_dim=config.embed_dim,
            context_length=config.context_length,
            num_head=config.num_heads,
            dropout=config.dropout,
        )
        summary = PretrainingTrainer(model, loader, config).train()
        print(summary)

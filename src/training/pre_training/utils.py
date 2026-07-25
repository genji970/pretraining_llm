from __future__ import annotations

import json
import math
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from training.pre_training.checkpoint import load_checkpoint, save_checkpoint
from training.pre_training.loss import causal_lm_loss


@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0
    best_eval_loss: float = math.inf


class PretrainingTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        train_loader: DataLoader,
        eval_loader: DataLoader | None,
        device: torch.device,
        output_dir: str | Path,
        *,
        epochs: int,
        max_steps: int = 0,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        log_every_steps: int = 10,
        eval_every_steps: int = 0,
        save_every_steps: int = 0,
        use_amp: bool = False,
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.device = device
        self.output_dir = Path(output_dir)
        self.epochs = epochs
        self.max_steps = max_steps
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.log_every_steps = log_every_steps
        self.eval_every_steps = eval_every_steps
        self.save_every_steps = save_every_steps
        self.use_amp = use_amp and device.type == "cuda"
        self.amp_dtype = (
            torch.bfloat16
            if self.use_amp and torch.cuda.is_bf16_supported()
            else torch.float16
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=self.use_amp and self.amp_dtype == torch.float16,
        )
        self.state = TrainerState()
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def resume(self, path: str | Path) -> None:
        loaded_state = load_checkpoint(
            path,
            self.model,
            self.optimizer,
            self.scheduler,
            map_location=self.device,
        )
        self.state = TrainerState(**loaded_state)

    def train(self) -> TrainerState:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        running_loss_sum = 0.0
        running_tokens = 0
        interval_start = time.perf_counter()
        stop = False

        for epoch in range(self.state.epoch, self.epochs):
            for batch_index, batch in enumerate(self.train_loader):
                batch = self._move_batch(batch)
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                    enabled=self.use_amp,
                ):
                    logits = self.model(batch["input_ids"], batch["attention_mask"])
                    loss = causal_lm_loss(logits, batch["labels"])
                    scaled_loss = loss / self.gradient_accumulation_steps

                self.scaler.scale(scaled_loss).backward()
                valid_tokens = int((batch["labels"] != -100).sum().item())
                running_loss_sum += float(loss.detach()) * valid_tokens
                running_tokens += valid_tokens

                is_accumulation_boundary = (
                    (batch_index + 1) % self.gradient_accumulation_steps == 0
                    or batch_index + 1 == len(self.train_loader)
                )
                if not is_accumulation_boundary:
                    continue

                self.scaler.unscale_(self.optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()
                self.state.global_step += 1

                if self.log_every_steps > 0 and self.state.global_step % self.log_every_steps == 0:
                    elapsed = max(time.perf_counter() - interval_start, 1e-9)
                    average_loss = running_loss_sum / max(running_tokens, 1)
                    payload = {
                        "step": self.state.global_step,
                        "epoch": epoch,
                        "train_loss": average_loss,
                        "perplexity": math.exp(min(average_loss, 20.0)),
                        "gradient_norm": float(gradient_norm),
                        "tokens_per_second": running_tokens / elapsed,
                        "learning_rate": self.optimizer.param_groups[0]["lr"],
                    }
                    print(json.dumps(payload))
                    running_loss_sum = 0.0
                    running_tokens = 0
                    interval_start = time.perf_counter()

                if (
                    self.eval_loader is not None
                    and self.eval_every_steps > 0
                    and self.state.global_step % self.eval_every_steps == 0
                ):
                    self._evaluate_and_maybe_save_best()
                    self.model.train()

                if (
                    self.save_every_steps > 0
                    and self.state.global_step % self.save_every_steps == 0
                ):
                    self._save(f"step_{self.state.global_step}.pt")

                if self.max_steps > 0 and self.state.global_step >= self.max_steps:
                    stop = True
                    break

            self.state.epoch = epoch + 1
            if stop:
                break

        if self.eval_loader is not None:
            self._evaluate_and_maybe_save_best(final=True)
        self._save("last.pt")
        return self.state

    @torch.no_grad()
    def evaluate(self) -> float:
        if self.eval_loader is None:
            raise RuntimeError("No evaluation loader was provided")
        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        for batch in self.eval_loader:
            batch = self._move_batch(batch)
            with torch.autocast(
                device_type=self.device.type,
                dtype=self.amp_dtype,
                enabled=self.use_amp,
            ):
                logits = self.model(batch["input_ids"], batch["attention_mask"])
                loss_sum = causal_lm_loss(logits, batch["labels"], reduction="sum")
            valid_tokens = int((batch["labels"] != -100).sum().item())
            total_loss += float(loss_sum)
            total_tokens += valid_tokens
        return total_loss / max(total_tokens, 1)

    def _evaluate_and_maybe_save_best(self, final: bool = False) -> float:
        eval_loss = self.evaluate()
        payload = {
            "step": self.state.global_step,
            "eval_loss": eval_loss,
            "eval_perplexity": math.exp(min(eval_loss, 20.0)),
        }
        if final:
            payload["final"] = True
        print(json.dumps(payload))
        if eval_loss < self.state.best_eval_loss:
            self.state.best_eval_loss = eval_loss
            self._save("best.pt")
        return eval_loss

    def _move_batch(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {
            name: tensor.to(self.device, non_blocking=self.device.type == "cuda")
            for name, tensor in batch.items()
        }

    def _save(self, filename: str) -> Path:
        return save_checkpoint(
            self.output_dir / filename,
            self.model,
            self.optimizer,
            self.scheduler,
            asdict(self.state),
        )


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from training.pre_training.dataset.causal_lm import CausalLMDataset
    from training.pre_training.dataset.collator import CausalLMCollator
    from training.pre_training.model.language_model import DecoderLanguageModel
    from training.pre_training.scheduler import build_scheduler
    from training.pre_training.tokenization.regex_tokenizer import RegexTokenizer

    texts = ["one two three", "one two", "three two one", "two three"]
    tokenizer = RegexTokenizer()
    tokenizer.train(texts)
    dataset = CausalLMDataset(texts, tokenizer, max_seq_len=16)
    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=True,
        collate_fn=CausalLMCollator(tokenizer.pad_token_id),
    )
    model = DecoderLanguageModel(
        tokenizer.vocab_size,
        d_model=16,
        n_heads=4,
        n_layers=1,
        ffn_multiplier=2,
        max_seq_len=16,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_scheduler(optimizer, "constant", warmup_steps=0, total_steps=2)

    with tempfile.TemporaryDirectory() as directory:
        trainer = PretrainingTrainer(
            model,
            optimizer,
            scheduler,
            loader,
            loader,
            torch.device("cpu"),
            directory,
            epochs=2,
            max_steps=2,
            log_every_steps=1,
            eval_every_steps=1,
            save_every_steps=1,
        )
        print(trainer.train())

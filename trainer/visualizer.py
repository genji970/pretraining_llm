from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from typing import TYPE_CHECKING

import torch.nn as nn

if TYPE_CHECKING:
    from config import TrainConfig


class TrainingVisualizer:
    """Persist scalar metrics to JSONL, TensorBoard, and separate PNG files."""

    def __init__(self, config: TrainConfig) -> None:
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.metrics_dir = self.output_dir / "metrics"
        self.plots_dir = self.output_dir / "plots"
        self.tensorboard_dir = self.output_dir / "tensorboard"

        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        if config.save_plots:
            self.plots_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_path = self.metrics_dir / "train_metrics.jsonl"
        self.history: dict[str, list[tuple[int, float]]] = defaultdict(list)
        self.recent: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=config.metric_smoothing_window)
        )

        self.writer = None
        if config.tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except ImportError as exc:
                raise ImportError(
                    "TensorBoard logging is enabled but tensorboard is not installed. "
                    "Install requirements.txt or run with --no-tensorboard."
                ) from exc
            self.tensorboard_dir.mkdir(parents=True, exist_ok=True)
            self.writer = SummaryWriter(log_dir=str(self.tensorboard_dir))

        self._load_existing_history()

    def _load_existing_history(self) -> None:
        if not self.metrics_path.exists():
            return

        for line in self.metrics_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            step = record.get("global_step")
            if not isinstance(step, int):
                continue
            for name, value in record.items():
                if name == "global_step" or isinstance(value, bool):
                    continue
                if isinstance(value, (int, float)):
                    numeric_value = float(value)
                    self.history[name].append((step, numeric_value))
                    self.recent[name].append(numeric_value)

    def record(
        self,
        metrics: dict[str, float | int],
        model: nn.Module | None = None,
    ) -> dict[str, float]:
        step = int(metrics["global_step"])
        record = dict(metrics)

        smoothed: dict[str, float] = {}
        for name, value in metrics.items():
            if name == "global_step" or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                numeric_value = float(value)
                self.history[name].append((step, numeric_value))
                self.recent[name].append(numeric_value)
                smoothed[name] = sum(self.recent[name]) / len(self.recent[name])

                if self.writer is not None:
                    self.writer.add_scalar(f"train/{name}", numeric_value, step)
                    self.writer.add_scalar(
                        f"train_smoothed/{name}", smoothed[name], step
                    )

        if self.config.jsonl_metrics:
            with self.metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

        if (
            self.writer is not None
            and model is not None
            and self.config.histogram_every_steps > 0
            and step % self.config.histogram_every_steps == 0
        ):
            for name, parameter in model.named_parameters():
                self.writer.add_histogram(f"parameters/{name}", parameter.detach(), step)
                if parameter.grad is not None:
                    self.writer.add_histogram(
                        f"gradients/{name}", parameter.grad.detach(), step
                    )

        if (
            self.config.save_plots
            and self.config.plot_every_steps > 0
            and step % self.config.plot_every_steps == 0
        ):
            self.save_plots()

        return smoothed

    def save_plots(self) -> None:
        if not self.config.save_plots:
            return

        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        metric_names = [
            name.strip()
            for name in self.config.plot_metrics.split(",")
            if name.strip()
        ]

        for metric_name in metric_names:
            points = self.history.get(metric_name, [])
            if not points:
                continue

            if len(points) > self.config.max_plot_points:
                stride = max(1, len(points) // self.config.max_plot_points)
                points = points[::stride]
                if points[-1] != self.history[metric_name][-1]:
                    points.append(self.history[metric_name][-1])

            steps = [step for step, _ in points]
            values = [value for _, value in points]

            figure, axis = plt.subplots(figsize=(8, 5))
            axis.plot(steps, values)
            axis.set_xlabel("Global step")
            axis.set_ylabel(metric_name.replace("_", " ").title())
            axis.set_title(metric_name.replace("_", " ").title())
            axis.grid(True, alpha=0.3)
            figure.tight_layout()
            figure.savefig(
                self.plots_dir / f"{metric_name}.png",
                dpi=160,
                bbox_inches="tight",
            )
            plt.close(figure)

    def close(self) -> None:
        if self.config.save_plots:
            self.save_plots()
        if self.writer is not None:
            self.writer.flush()
            self.writer.close()

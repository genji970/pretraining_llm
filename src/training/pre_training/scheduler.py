from __future__ import annotations

import math

import torch
from torch.optim.lr_scheduler import LambdaLR


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    warmup_steps: int,
    total_steps: int,
) -> LambdaLR:
    if total_steps < 1:
        raise ValueError("total_steps must be positive")
    if not 0 <= warmup_steps <= total_steps:
        raise ValueError("warmup_steps must be between 0 and total_steps")
    if scheduler_type not in {"constant", "linear", "cosine"}:
        raise ValueError("scheduler_type must be constant, linear, or cosine")

    def multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)

        if scheduler_type == "constant":
            return 1.0

        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        progress = min(max(progress, 0.0), 1.0)
        if scheduler_type == "linear":
            return 1.0 - progress
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda=multiplier)


if __name__ == "__main__":
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=1e-3)
    scheduler = build_scheduler(optimizer, "cosine", warmup_steps=2, total_steps=6)
    for step in range(6):
        optimizer.step()
        scheduler.step()
        print(step + 1, optimizer.param_groups[0]["lr"])

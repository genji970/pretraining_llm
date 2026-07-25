from __future__ import annotations

import torch
import torch.nn.functional as F


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    if logits.ndim != 3:
        raise ValueError("logits must have shape [batch, sequence, vocab]")
    if labels.ndim != 2:
        raise ValueError("labels must have shape [batch, sequence]")
    if logits.shape[:2] != labels.shape:
        raise ValueError("The batch and sequence dimensions of logits and labels must match")

    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        labels.reshape(-1),
        ignore_index=-100,
        reduction=reduction,
    )


if __name__ == "__main__":
    logits = torch.randn(2, 4, 10)
    labels = torch.tensor([[1, 2, 3, 4], [4, 5, -100, -100]])
    print("loss:", float(causal_lm_loss(logits, labels)))

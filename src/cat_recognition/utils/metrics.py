from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass
class AverageMeter:
    total: float = 0.0
    count: int = 0

    @property
    def avg(self) -> float:
        return self.total / max(self.count, 1)

    def update(self, value: float, n: int = 1) -> None:
        self.total += value * n
        self.count += n


def topk_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    topk: Iterable[int] = (1,),
) -> list[float]:
    max_k = max(topk)
    _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    results: list[float] = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        results.append(correct_k.mul_(100.0 / targets.size(0)).item())
    return results

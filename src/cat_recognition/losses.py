from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin: float = 0.2) -> None:
        super().__init__()
        self.margin = float(margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected embeddings with shape (B, D), got: {tuple(embeddings.shape)}")
        if labels.ndim != 1:
            labels = labels.view(-1)

        dist_mat = torch.cdist(embeddings, embeddings, p=2)
        labels = labels.view(-1, 1)
        positive_mask = labels.eq(labels.t())
        negative_mask = ~positive_mask
        positive_mask.fill_diagonal_(False)

        valid_positive = positive_mask.any(dim=1)
        valid_negative = negative_mask.any(dim=1)
        valid_anchor = valid_positive & valid_negative
        if not valid_anchor.any():
            return embeddings.new_zeros(())

        hardest_positive = dist_mat.masked_fill(~positive_mask, float("-inf")).max(dim=1).values
        hardest_negative = dist_mat.masked_fill(~negative_mask, float("inf")).min(dim=1).values

        loss = F.relu(hardest_positive - hardest_negative + self.margin)
        return loss[valid_anchor].mean()


def build_aux_loss(cfg):
    if not cfg:
        return None

    name = str(cfg.get("name", "none")).lower()
    if name in {"none", "", "null"}:
        return None
    if name == "batch_hard_triplet":
        return BatchHardTripletLoss(margin=float(cfg.get("margin", 0.2)))
    raise ValueError(f"Unsupported auxiliary loss: {cfg.get('name')}")

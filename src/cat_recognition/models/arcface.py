from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcMarginProduct(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale: float = 64.0,
        margin: float = 0.5,
        easy_margin: bool = False,
        ls_eps: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.scale = scale
        self.margin = margin
        self.easy_margin = easy_margin
        self.ls_eps = ls_eps

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def _apply_margin(self, cosine: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=0.0))
        phi = cosine * self.cos_m - sine * self.sin_m

        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        if self.ls_eps > 0:
            one_hot = (1 - self.ls_eps) * one_hot + self.ls_eps / self.out_features

        logits = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        return logits * self.scale

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        return self._apply_margin(cosine, labels)


class SubCenterArcMarginProduct(ArcMarginProduct):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        scale: float = 64.0,
        margin: float = 0.5,
        easy_margin: bool = False,
        ls_eps: float = 0.0,
        num_subcenters: int = 3,
    ) -> None:
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            scale=scale,
            margin=margin,
            easy_margin=easy_margin,
            ls_eps=ls_eps,
        )
        self.num_subcenters = int(num_subcenters)
        if self.num_subcenters <= 0:
            raise ValueError("num_subcenters must be > 0.")
        self.weight = nn.Parameter(torch.empty(out_features, self.num_subcenters, in_features))
        nn.init.xavier_uniform_(self.weight.view(out_features * self.num_subcenters, in_features))

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        normalized_embeddings = F.normalize(embeddings)
        normalized_weight = F.normalize(self.weight, dim=-1).view(
            self.out_features * self.num_subcenters,
            self.in_features,
        )
        cosine = F.linear(normalized_embeddings, normalized_weight)
        cosine = cosine.view(-1, self.out_features, self.num_subcenters).max(dim=-1).values
        return self._apply_margin(cosine, labels)

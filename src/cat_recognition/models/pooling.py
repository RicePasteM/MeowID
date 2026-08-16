from __future__ import annotations

import torch
import torch.nn as nn


def _normalize_layout(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 2:
        return x
    if x.ndim == 3:
        if x.shape[-1] >= x.shape[1]:
            return x.transpose(1, 2)
        return x
    if x.ndim == 4:
        if x.shape[-1] >= x.shape[1]:
            return x.permute(0, 3, 1, 2)
        return x
    raise ValueError(f"Unsupported feature tensor shape: {tuple(x.shape)}")


def _avg_pool_tokens(tokens: torch.Tensor) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"Expected token tensor with shape (B, N, C), got: {tuple(tokens.shape)}")
    return tokens.mean(dim=1)


def _gem_pool_tokens(tokens: torch.Tensor, p: torch.Tensor, eps: float) -> torch.Tensor:
    if tokens.ndim != 3:
        raise ValueError(f"Expected token tensor with shape (B, N, C), got: {tuple(tokens.shape)}")
    x = tokens.transpose(1, 2).clamp(min=eps).pow(p)
    return x.mean(dim=-1).pow(1.0 / p)


class GeMPooling(nn.Module):
    def __init__(self, p: float = 3.0, eps: float = 1.0e-6, trainable_p: bool = True) -> None:
        super().__init__()
        if trainable_p:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer("p", torch.ones(1) * p)
        self.eps = eps

    def get_output_dim(self, input_dim: int) -> int:
        return int(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _normalize_layout(x)
        if x.ndim == 2:
            return x
        x = x.clamp(min=self.eps).pow(self.p)
        if x.ndim == 3:
            return x.mean(dim=-1).pow(1.0 / self.p)
        return x.mean(dim=(-1, -2)).pow(1.0 / self.p)


class AvgPooling(nn.Module):
    def get_output_dim(self, input_dim: int) -> int:
        return int(input_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = _normalize_layout(x)
        if x.ndim == 2:
            return x
        if x.ndim == 3:
            return x.mean(dim=-1)
        return x.mean(dim=(-1, -2))


class DinoV3TokenPooling(nn.Module):
    def __init__(
        self,
        patch_pool: str = "gem",
        combine: str = "concat",
        include_cls: bool = True,
        include_patch: bool = True,
        p: float = 3.0,
        eps: float = 1.0e-6,
        trainable_p: bool = True,
        input_dim: int | None = None,
        gate_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.patch_pool = patch_pool.lower()
        self.combine = combine.lower()
        self.include_cls = include_cls
        self.include_patch = include_patch
        self.input_dim = input_dim
        self.gate_hidden_dim = gate_hidden_dim
        if trainable_p:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer("p", torch.ones(1) * p)
        self.eps = eps
        self.gate = None

        if self.patch_pool not in {"avg", "gem"}:
            raise ValueError(f"Unsupported DINOv3 patch_pool: {patch_pool}")
        if self.combine not in {"concat", "mean", "sum", "gated"}:
            raise ValueError(f"Unsupported DINOv3 combine mode: {combine}")
        if self.combine == "gated":
            if not self.include_cls or not self.include_patch:
                raise ValueError("DinoV3TokenPooling combine='gated' requires include_cls=true and include_patch=true.")
            if input_dim is None:
                raise ValueError("DinoV3TokenPooling combine='gated' requires input_dim.")
            hidden_dim = int(gate_hidden_dim or input_dim)
            self.gate = nn.Sequential(
                nn.Linear(int(input_dim) * 2, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 2),
            )

    def get_output_dim(self, input_dim: int) -> int:
        parts = int(self.include_cls) + int(self.include_patch)
        if parts <= 0:
            raise ValueError("DinoV3TokenPooling requires at least one of include_cls/include_patch.")
        if self.combine == "concat":
            return int(input_dim) * parts
        return int(input_dim)

    def _pool_patch_tokens(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        if self.patch_pool == "avg":
            return _avg_pool_tokens(patch_tokens)
        return _gem_pool_tokens(patch_tokens, p=self.p, eps=self.eps)

    def forward(self, x) -> torch.Tensor:
        if not isinstance(x, dict):
            raise TypeError("DinoV3TokenPooling expects a dict with cls_token and patch_tokens.")

        parts = []

        cls_token = x.get("cls_token")
        if self.include_cls and cls_token is not None:
            if cls_token.ndim == 3:
                cls_token = cls_token.squeeze(1)
            parts.append(cls_token)

        patch_tokens = x.get("patch_tokens")
        if self.include_patch and patch_tokens is not None:
            if patch_tokens.ndim != 3 or patch_tokens.shape[1] == 0:
                raise ValueError("DINOv3 patch_tokens must have shape (B, N, C) with N > 0.")
            parts.append(self._pool_patch_tokens(patch_tokens))

        if not parts:
            raise ValueError("DinoV3TokenPooling did not receive any usable token features.")

        if len(parts) == 1:
            return parts[0]
        if self.combine == "concat":
            return torch.cat(parts, dim=-1)
        if self.combine == "gated":
            gate_logits = self.gate(torch.cat(parts, dim=-1))
            gate_weights = torch.softmax(gate_logits, dim=-1)
            return (gate_weights[:, 0:1] * parts[0]) + (gate_weights[:, 1:2] * parts[1])

        stacked = torch.stack(parts, dim=0)
        if self.combine == "mean":
            return stacked.mean(dim=0)
        return stacked.sum(dim=0)


def build_pool(cfg, input_dim: int | None = None) -> nn.Module:
    name = cfg.name.lower()
    if name == "gem":
        return GeMPooling(
            p=float(cfg.get("p", 3.0)),
            eps=float(cfg.get("eps", 1.0e-6)),
            trainable_p=bool(cfg.get("trainable_p", True)),
        )
    if name == "avg":
        return AvgPooling()
    if name == "dinov3_token":
        return DinoV3TokenPooling(
            patch_pool=str(cfg.get("patch_pool", "gem")),
            combine=str(cfg.get("combine", "concat")),
            include_cls=bool(cfg.get("include_cls", True)),
            include_patch=bool(cfg.get("include_patch", True)),
            p=float(cfg.get("p", 3.0)),
            eps=float(cfg.get("eps", 1.0e-6)),
            trainable_p=bool(cfg.get("trainable_p", True)),
            input_dim=input_dim,
            gate_hidden_dim=int(cfg.get("gate_hidden_dim", input_dim or 0) or (input_dim or 0)),
        )
    raise ValueError(f"Unsupported pooling type: {cfg.name}")

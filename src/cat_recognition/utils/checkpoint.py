from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def save_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    epoch: int = 0,
    best_metric: float | None = None,
    best_epoch: int | None = None,
    best_summary: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model": unwrap_model(model).state_dict(),
        "epoch": epoch,
        "best_metric": best_metric,
        "best_epoch": best_epoch,
        "best_summary": best_summary,
        "config": config,
        "extra": extra or {},
    }
    if optimizer is not None:
        checkpoint["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler"] = scheduler.state_dict()
    if scaler is not None:
        checkpoint["scaler"] = scaler.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(
    path: str | Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    unwrapped = unwrap_model(model)
    if bool((checkpoint.get("extra") or {}).get("single_body_deployment", False)):
        prepare = getattr(unwrapped, "prepare_for_single_body_deployment", None)
        if prepare is None:
            raise TypeError("Checkpoint requires a model with single-body deployment support")
        prepare()
    incompatible = unwrapped.load_state_dict(checkpoint["model"], strict=strict)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    checkpoint["missing_keys"] = list(getattr(incompatible, "missing_keys", []))
    checkpoint["unexpected_keys"] = list(getattr(incompatible, "unexpected_keys", []))
    return checkpoint

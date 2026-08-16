from __future__ import annotations

import torch


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def build_optimizer(cfg, model: torch.nn.Module) -> torch.optim.Optimizer:
    unwrapped = _unwrap_model(model)
    if hasattr(unwrapped, "get_optimizer_param_groups"):
        params = unwrapped.get_optimizer_param_groups(cfg)
    else:
        params = list(model.parameters())
    name = cfg.optimizer.name.lower()

    if name == "adamw":
        return torch.optim.AdamW(
            params,
            lr=float(cfg.optimizer.lr),
            weight_decay=float(cfg.optimizer.get("weight_decay", 0.0)),
            betas=tuple(cfg.optimizer.get("betas", [0.9, 0.999])),
        )
    if name == "sgd":
        return torch.optim.SGD(
            params,
            lr=float(cfg.optimizer.lr),
            momentum=float(cfg.optimizer.get("momentum", 0.9)),
            weight_decay=float(cfg.optimizer.get("weight_decay", 0.0)),
            nesterov=bool(cfg.optimizer.get("nesterov", True)),
        )
    raise ValueError(f"Unsupported optimizer: {cfg.optimizer.name}")


def build_scheduler(cfg, optimizer: torch.optim.Optimizer):
    name = cfg.scheduler.name.lower()
    if name == "none":
        return None

    if name == "cosine":
        warmup_epochs = int(cfg.scheduler.get("warmup_epochs", 0))
        total_epochs = int(cfg.scheduler.get("total_epochs", cfg.train.epochs))
        min_lr = float(cfg.scheduler.get("min_lr", 0.0))
        cosine_epochs = max(total_epochs - warmup_epochs, 1)

        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cosine_epochs,
            eta_min=min_lr,
        )

        if warmup_epochs <= 0:
            return cosine

        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=float(cfg.scheduler.get("warmup_start_factor", 0.1)),
            total_iters=warmup_epochs,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )

    if name == "multistep":
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(cfg.scheduler.get("milestones", [10, 20])),
            gamma=float(cfg.scheduler.get("gamma", 0.1)),
        )

    raise ValueError(f"Unsupported scheduler: {cfg.scheduler.name}")

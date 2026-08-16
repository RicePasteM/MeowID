from __future__ import annotations

import os
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_world_size() -> int:
    if not is_distributed():
        return 1
    return dist.get_world_size()


def get_rank() -> int:
    if not is_distributed():
        return 0
    return dist.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def init_distributed(backend: str = "nccl", timeout_minutes: int = 30) -> bool:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        torch.cuda.set_device(local_rank)
    else:
        backend = "gloo"

    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            timeout=timedelta(minutes=timeout_minutes),
        )
    return True


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def destroy_distributed() -> None:
    if is_distributed():
        dist.destroy_process_group()


def all_gather_object(obj: Any) -> list[Any]:
    if not is_distributed():
        return [obj]
    gathered: list[Any] = [None for _ in range(get_world_size())]
    dist.all_gather_object(gathered, obj)
    return gathered


def reduce_dict_sum(metrics: dict[str, float], device: torch.device) -> dict[str, float]:
    if not is_distributed():
        return metrics

    keys = sorted(metrics.keys())
    values = torch.tensor([float(metrics[key]) for key in keys], device=device)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {key: value.item() for key, value in zip(keys, values)}

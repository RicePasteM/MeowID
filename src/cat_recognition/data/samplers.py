from __future__ import annotations

import math
import random
from collections import defaultdict

from torch.utils.data import Sampler


class DistributedEvalSampler(Sampler[int]):
    """Shard evaluation samples across ranks without padding duplicates."""

    def __init__(self, dataset_size: int, rank: int = 0, world_size: int = 1) -> None:
        if dataset_size < 0:
            raise ValueError(f"dataset_size must be non-negative, got {dataset_size}")
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}")
        if rank < 0 or rank >= world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        self.dataset_size = int(dataset_size)
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, self.dataset_size, self.world_size))

    def __len__(self) -> int:
        remaining = self.dataset_size - self.rank
        if remaining <= 0:
            return 0
        return (remaining + self.world_size - 1) // self.world_size


class PKBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        labels: list[int],
        p: int,
        k: int,
        drop_last: bool = True,
        seed: int = 42,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        if p <= 0 or k <= 0:
            raise ValueError(f"PK sampler expects positive p/k, got p={p}, k={k}")

        self.labels = list(labels)
        self.p = int(p)
        self.k = int(k)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.epoch = 0

        label_to_indices: dict[int, list[int]] = defaultdict(list)
        for index, label in enumerate(self.labels):
            if int(label) < 0:
                continue
            label_to_indices[int(label)].append(index)

        self.label_to_indices = dict(label_to_indices)
        self.class_labels = sorted(self.label_to_indices.keys())
        if len(self.class_labels) < self.p:
            raise ValueError(
                f"PK sampler requires at least p={self.p} classes, found {len(self.class_labels)} valid classes."
            )

        self.batch_size = self.p * self.k
        total_batches = len(self.labels) / max(self.batch_size, 1)
        if self.drop_last:
            self.num_batches_per_rank = max(int(total_batches // max(self.world_size, 1)), 1)
        else:
            self.num_batches_per_rank = max(int(math.ceil(total_batches / max(self.world_size, 1))), 1)

    def __len__(self) -> int:
        return self.num_batches_per_rank

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _sample_class_indices(self, rng: random.Random, label: int) -> list[int]:
        indices = self.label_to_indices[label]
        if len(indices) >= self.k:
            return rng.sample(indices, self.k)
        return [rng.choice(indices) for _ in range(self.k)]

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        all_batches: list[list[int]] = []
        total_batches = self.num_batches_per_rank * self.world_size

        for _ in range(total_batches):
            selected_labels = rng.sample(self.class_labels, self.p)
            batch = []
            for label in selected_labels:
                batch.extend(self._sample_class_indices(rng, label))
            rng.shuffle(batch)
            all_batches.append(batch)

        yield from all_batches[self.rank :: self.world_size]

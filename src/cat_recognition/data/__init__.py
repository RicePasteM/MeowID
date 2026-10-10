from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .dataset import ManifestDataset, PairedFaceHintSplitDataset, SingleImageDataset, SplitDataset
from .samplers import DistributedEvalSampler, PKBatchSampler
from .transforms import build_eval_transform, build_train_transform


def _get(cfg, key: str, default=None):
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def resolve_path(root_dir: str | Path, value: str | None) -> str | None:
    if not value:
        return None
    path = Path(value)
    if path.is_absolute():
        return str(path)
    return str((Path(root_dir) / path).resolve())


def resolve_path_mapping(root_dir: str | Path, mapping) -> dict[str, str]:
    if not mapping:
        return {}
    if hasattr(mapping, "to_dict"):
        mapping = mapping.to_dict()
    return {
        str(key): str(Path(resolve_path(root_dir, str(value)) or str(value)).resolve())
        for key, value in dict(mapping).items()
    }


def build_manifest_dataset(
    split_cfg,
    data_cfg,
    is_train: bool,
    root_dir: str | Path,
    class_to_idx: dict[str, int] | None = None,
    build_class_to_idx: bool = False,
) -> ManifestDataset:
    transform_cfg = data_cfg.transforms.train if is_train else data_cfg.transforms.eval
    transform = build_train_transform(transform_cfg) if is_train else build_eval_transform(transform_cfg)
    manifest_path = resolve_path(root_dir, _get(split_cfg, "manifest"))
    data_root = resolve_path(root_dir, _get(data_cfg, "root"))
    data_roots = resolve_path_mapping(root_dir, _get(data_cfg, "roots", {}))

    return ManifestDataset(
        manifest_path=manifest_path,
        root=data_root,
        roots=data_roots,
        transform=transform,
        image_key=_get(data_cfg, "image_key", "image_path"),
        label_key=_get(data_cfg, "label_key", "label"),
        cat_id_key=_get(data_cfg, "cat_id_key", "cat_id"),
        dataset_name_key=_get(data_cfg, "dataset_name_key", "dataset_name"),
        dataset_root_key=_get(data_cfg, "dataset_root_key", "dataset_root"),
        class_to_idx=class_to_idx,
        build_class_to_idx=build_class_to_idx,
    )


def build_image_dataset(image_paths: list[str], transform_cfg) -> SingleImageDataset:
    return SingleImageDataset(image_paths=image_paths, transform=build_eval_transform(transform_cfg))


def build_split_dataset(
    data_cfg,
    split_name: str,
    is_train: bool,
    root_dir: str | Path,
    class_to_idx: dict[str, int] | None = None,
    build_class_to_idx: bool = False,
) -> SplitDataset:
    transform_cfg = data_cfg.transforms.train if is_train else data_cfg.transforms.eval
    transform = build_train_transform(transform_cfg) if is_train else build_eval_transform(transform_cfg)
    split = _get(data_cfg.get(split_name), "split", split_name) if data_cfg.get(split_name) else split_name
    data_root = resolve_path(root_dir, _get(data_cfg, "root"))

    return SplitDataset(
        root=data_root,
        split=split,
        transform=transform,
        class_to_idx=class_to_idx,
        build_class_to_idx=build_class_to_idx,
    )


def build_paired_face_hint_split_dataset(
    data_cfg,
    split_name: str,
    is_train: bool,
    root_dir: str | Path,
    class_to_idx: dict[str, int] | None = None,
    build_class_to_idx: bool = False,
) -> PairedFaceHintSplitDataset:
    whole_transform_cfg = data_cfg.transforms.train if is_train else data_cfg.transforms.eval
    face_transform_cfg = data_cfg.face_transforms.train if is_train else data_cfg.face_transforms.eval
    whole_transform = (
        build_train_transform(whole_transform_cfg)
        if is_train
        else build_eval_transform(whole_transform_cfg)
    )
    face_transform = (
        build_train_transform(face_transform_cfg)
        if is_train
        else build_eval_transform(face_transform_cfg)
    )
    split = _get(data_cfg.get(split_name), "split", split_name) if data_cfg.get(split_name) else split_name
    data_root = resolve_path(root_dir, _get(data_cfg, "root"))
    face_root = resolve_path(root_dir, _get(data_cfg, "face_root"))
    return PairedFaceHintSplitDataset(
        root=data_root,
        face_root=face_root,
        split=split,
        transform=whole_transform,
        face_transform=face_transform,
        class_to_idx=class_to_idx,
        build_class_to_idx=build_class_to_idx,
    )


def build_dataloader(dataset, split_cfg, is_train: bool, distributed: bool):
    sampler = None
    batch_sampler = None
    sampler_cfg = _get(split_cfg, "sampler", None)

    if is_train and sampler_cfg and str(_get(sampler_cfg, "name", "default")).lower() == "pk":
        batch_size = int(_get(split_cfg, "batch_size"))
        p = int(_get(sampler_cfg, "p"))
        k = int(_get(sampler_cfg, "k"))
        if batch_size != p * k:
            raise ValueError(f"PK sampler requires batch_size == p*k, got batch_size={batch_size}, p={p}, k={k}")
        rank = 0
        world_size = 1
        if distributed:
            import torch.distributed as dist

            rank = dist.get_rank()
            world_size = dist.get_world_size()
        batch_sampler = PKBatchSampler(
            labels=getattr(dataset, "labels", []),
            p=p,
            k=k,
            drop_last=bool(_get(split_cfg, "drop_last", False)),
            seed=int(_get(sampler_cfg, "seed", 42)),
            rank=rank,
            world_size=world_size,
        )
        sampler = batch_sampler

    if distributed:
        if batch_sampler is None:
            if is_train:
                sampler = DistributedSampler(dataset, shuffle=True)
            else:
                import torch.distributed as dist

                sampler = DistributedEvalSampler(
                    dataset_size=len(dataset),
                    rank=dist.get_rank(),
                    world_size=dist.get_world_size(),
                )

    if batch_sampler is not None:
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            num_workers=_get(split_cfg, "num_workers", 0),
            pin_memory=_get(split_cfg, "pin_memory", True),
            persistent_workers=_get(split_cfg, "num_workers", 0) > 0,
        )

    return DataLoader(
        dataset,
        batch_size=_get(split_cfg, "batch_size"),
        shuffle=is_train and sampler is None,
        sampler=sampler,
        num_workers=_get(split_cfg, "num_workers", 0),
        pin_memory=_get(split_cfg, "pin_memory", True),
        drop_last=_get(split_cfg, "drop_last", False) if is_train else False,
        persistent_workers=_get(split_cfg, "num_workers", 0) > 0,
    )


__all__ = [
    "ManifestDataset",
    "PairedFaceHintSplitDataset",
    "SingleImageDataset",
    "SplitDataset",
    "DistributedEvalSampler",
    "PKBatchSampler",
    "build_dataloader",
    "build_image_dataset",
    "build_manifest_dataset",
    "build_paired_face_hint_split_dataset",
    "build_split_dataset",
    "resolve_path_mapping",
    "resolve_path",
]

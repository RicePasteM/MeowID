from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageFile
import torch
from torch.utils.data import Dataset

ImageFile.LOAD_TRUNCATED_IMAGES = True


def _read_manifest(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    if path.suffix.lower() == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, list):
            raise ValueError(f"JSON manifest must be a list: {path}")
        return data
    raise ValueError(f"Unsupported manifest format: {path}")


def _resolve_image_path(
    image_path: str,
    manifest_path: Path,
    root: str | None,
    roots: dict[str, str] | None = None,
    dataset_name: str | None = None,
    dataset_root: str | None = None,
) -> str:
    path = Path(image_path)
    if path.is_absolute():
        return str(path)
    if dataset_root:
        return str((Path(dataset_root) / path).resolve())
    if dataset_name and roots and dataset_name in roots:
        return str((Path(roots[dataset_name]) / path).resolve())
    if root:
        return str((Path(root) / path).resolve())
    return str((manifest_path.parent / path).resolve())


class ManifestDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        root: str | None = None,
        transform=None,
        image_key: str = "image_path",
        label_key: str = "label",
        cat_id_key: str = "cat_id",
        dataset_name_key: str = "dataset_name",
        dataset_root_key: str = "dataset_root",
        roots: dict[str, str] | None = None,
        class_to_idx: dict[str, int] | None = None,
        build_class_to_idx: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path).resolve()
        self.root = root
        self.transform = transform
        self.image_key = image_key
        self.label_key = label_key
        self.cat_id_key = cat_id_key
        self.dataset_name_key = dataset_name_key
        self.dataset_root_key = dataset_root_key
        self.roots = dict(roots or {})

        rows = _read_manifest(self.manifest_path)
        if not rows:
            raise ValueError(f"Empty manifest: {self.manifest_path}")

        class_keys = []
        for row in rows:
            class_key = row.get(self.cat_id_key) or row.get(self.label_key)
            if class_key not in (None, ""):
                class_keys.append(str(class_key))

        if class_to_idx is not None:
            self.class_to_idx = dict(class_to_idx)
        elif build_class_to_idx:
            self.class_to_idx = {
                key: idx for idx, key in enumerate(sorted(set(class_keys)))
            }
        else:
            self.class_to_idx = {}

        self.samples = []
        for row in rows:
            image_path = row.get(self.image_key)
            if not image_path:
                raise ValueError(
                    f"Manifest row missing image key '{self.image_key}': {self.manifest_path}"
                )

            raw_label = row.get(self.label_key)
            cat_id = row.get(self.cat_id_key)
            dataset_name = row.get(self.dataset_name_key)
            dataset_root = row.get(self.dataset_root_key)
            class_key = str(cat_id or raw_label or "")
            target = self.class_to_idx.get(class_key, -1)

            self.samples.append(
                {
                    "path": _resolve_image_path(
                        image_path,
                        self.manifest_path,
                        self.root,
                        roots=self.roots,
                        dataset_name=str(dataset_name) if dataset_name not in (None, "") else None,
                        dataset_root=str(dataset_root) if dataset_root not in (None, "") else None,
                    ),
                    "cat_id": str(cat_id) if cat_id not in (None, "") else class_key,
                    "dataset_name": str(dataset_name) if dataset_name not in (None, "") else "",
                    "label": target,
                }
            )

        self.labels = [int(sample["label"]) for sample in self.samples]
        self.num_classes = len(self.class_to_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = Image.open(sample["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": sample["label"],
            "cat_id": sample["cat_id"],
            "dataset_name": sample["dataset_name"],
            "path": sample["path"],
        }


class SplitDataset(Dataset):
    def __init__(
        self,
        root: str | Path,
        split: str = "train",
        transform=None,
        class_to_idx: dict[str, int] | None = None,
        build_class_to_idx: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
        self.transform = transform

        if not self.split_dir.is_dir():
            raise ValueError(f"Split directory not found: {self.split_dir}")

        cat_dirs = sorted(
            [d for d in self.split_dir.iterdir() if d.is_dir()],
            key=lambda d: d.name,
        )

        if class_to_idx is not None:
            self.class_to_idx = dict(class_to_idx)
        elif build_class_to_idx:
            self.class_to_idx = {
                d.name: idx for idx, d in enumerate(cat_dirs)
            }
        else:
            self.class_to_idx = {}

        self.samples: list[dict[str, Any]] = []
        _img_exts = {".jpg", ".jpeg", ".png"}
        for cat_dir in cat_dirs:
            cat_id = cat_dir.name
            images = sorted(
                [f for f in cat_dir.iterdir() if f.suffix.lower() in _img_exts],
                key=lambda f: f.name,
            )
            target = self.class_to_idx.get(cat_id, -1)
            for img_path in images:
                self.samples.append(
                    {
                        "path": str(img_path),
                        "cat_id": cat_id,
                        "label": target,
                        "dataset_name": "",
                    }
                )

        self.labels = [int(s["label"]) for s in self.samples]
        self.num_classes = len(self.class_to_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        image = Image.open(sample["path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": sample["label"],
            "cat_id": sample["cat_id"],
            "dataset_name": sample["dataset_name"],
            "path": sample["path"],
        }


class PairedFaceHintSplitDataset(Dataset):
    """ICW split dataset returning a whole image and an optional aligned face.

    The whole-image split is authoritative.  A face is available only when the
    same ``<split>/<cat_id>/<filename>`` exists below ``face_root``.  Missing
    faces are represented by a zero tensor plus an explicit boolean mask; the
    model must never infer availability from tensor values or detector scores.
    """

    def __init__(
        self,
        root: str | Path,
        face_root: str | Path,
        split: str = "train",
        transform=None,
        face_transform=None,
        class_to_idx: dict[str, int] | None = None,
        build_class_to_idx: bool = False,
    ) -> None:
        self.root = Path(root)
        self.face_root = Path(face_root)
        self.split = split
        self.split_dir = self.root / split
        self.face_split_dir = self.face_root / split
        self.transform = transform
        self.face_transform = face_transform

        if not self.split_dir.is_dir():
            raise ValueError(f"Split directory not found: {self.split_dir}")
        if not self.face_split_dir.is_dir():
            raise ValueError(f"Face split directory not found: {self.face_split_dir}")

        cat_dirs = sorted(
            [directory for directory in self.split_dir.iterdir() if directory.is_dir()],
            key=lambda directory: directory.name,
        )
        if class_to_idx is not None:
            self.class_to_idx = dict(class_to_idx)
        elif build_class_to_idx:
            self.class_to_idx = {
                directory.name: index for index, directory in enumerate(cat_dirs)
            }
        else:
            self.class_to_idx = {}

        self.samples: list[dict[str, Any]] = []
        image_extensions = {".jpg", ".jpeg", ".png"}
        for cat_dir in cat_dirs:
            cat_id = cat_dir.name
            label = self.class_to_idx.get(cat_id, -1)
            image_paths = sorted(
                [path for path in cat_dir.iterdir() if path.suffix.lower() in image_extensions],
                key=lambda path: path.name,
            )
            for image_path in image_paths:
                face_path = self.face_split_dir / cat_id / image_path.name
                self.samples.append(
                    {
                        "path": str(image_path),
                        "face_path": str(face_path),
                        "face_exists": face_path.is_file(),
                        "cat_id": cat_id,
                        "label": label,
                        "dataset_name": "",
                    }
                )

        self.labels = [int(sample["label"]) for sample in self.samples]
        self.num_classes = len(self.class_to_idx)
        self.face_count = sum(bool(sample["face_exists"]) for sample in self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.samples[index]
        whole_image = Image.open(sample["path"]).convert("RGB")
        if self.transform is not None:
            whole_image = self.transform(whole_image)

        if sample["face_exists"]:
            face_image = Image.open(sample["face_path"]).convert("RGB")
            if self.face_transform is not None:
                face_image = self.face_transform(face_image)
        else:
            if not isinstance(whole_image, torch.Tensor):
                raise TypeError("PairedFaceHintSplitDataset requires tensor transforms")
            face_image = torch.zeros_like(whole_image)

        if not isinstance(face_image, torch.Tensor):
            raise TypeError("PairedFaceHintSplitDataset requires tensor face transforms")
        if face_image.shape != whole_image.shape:
            raise ValueError(
                "Whole and face transforms must produce equal tensor shapes, got "
                f"whole={tuple(whole_image.shape)} face={tuple(face_image.shape)}"
            )

        return {
            "image": whole_image,
            "face_image": face_image,
            "face_exists": bool(sample["face_exists"]),
            "label": sample["label"],
            "cat_id": sample["cat_id"],
            "dataset_name": sample["dataset_name"],
            "path": sample["path"],
            "face_path": sample["face_path"] if sample["face_exists"] else "",
        }


class SingleImageDataset(Dataset):
    def __init__(self, image_paths: list[str], transform=None) -> None:
        self.image_paths = [str(Path(path).resolve()) for path in image_paths]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.image_paths[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": -1,
            "cat_id": "",
            "path": path,
        }

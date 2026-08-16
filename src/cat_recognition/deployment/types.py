from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class LoadedImage:
    image: Image.Image
    source: str


@dataclass
class PoseDetection:
    """Best cat-face pose returned by ECPose for one image."""

    keypoints: np.ndarray
    score: float
    label: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "score": float(self.score),
            "label": int(self.label),
            "keypoints": np.asarray(self.keypoints, dtype=np.float32).tolist(),
        }


@dataclass
class EmbeddingResult:
    """Hard-routed MeowID embedding and its diagnostic branch outputs."""

    source: str
    embedding: np.ndarray
    body_embedding: np.ndarray
    face_embedding: np.ndarray | None
    route: str
    face: PoseDetection | None = None
    aligned_face: Image.Image | None = field(default=None, repr=False)

    @property
    def face_detected(self) -> bool:
        return self.face is not None

    def to_dict(self, include_embedding: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "source": self.source,
            "route": self.route,
            "face_detected": self.face_detected,
            "face": None if self.face is None else self.face.to_dict(),
        }
        if include_embedding:
            payload["embedding"] = self.embedding.astype(np.float32).tolist()
            payload["body_embedding"] = self.body_embedding.astype(np.float32).tolist()
            payload["face_embedding"] = (
                None
                if self.face_embedding is None
                else self.face_embedding.astype(np.float32).tolist()
            )
        return payload


@dataclass
class SearchMatch:
    cat_id: str
    score: float
    support: int
    route: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cat_id": self.cat_id,
            "score": float(self.score),
            "support": int(self.support),
            "route": self.route,
            "metadata": self.metadata,
        }


@dataclass
class PredictionResult:
    embedding: EmbeddingResult
    matches: list[SearchMatch]
    unknown: bool = False

    def to_dict(self, include_embedding: bool = False) -> dict[str, Any]:
        return {
            **self.embedding.to_dict(include_embedding=include_embedding),
            "matches": [match.to_dict() for match in self.matches],
            "unknown": bool(self.unknown),
        }


@dataclass
class CatCrop:
    """One ECSeg cat instance in original-image coordinates."""

    score: float
    label: int
    box: np.ndarray
    crop_box: np.ndarray
    mask: np.ndarray = field(repr=False)
    crop: Image.Image = field(repr=False)

    def to_dict(self, include_mask: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "score": float(self.score),
            "label": int(self.label),
            "box": np.asarray(self.box, dtype=np.float32).tolist(),
            "crop_box": np.asarray(self.crop_box, dtype=np.int64).tolist(),
            "crop_size": [int(self.crop.width), int(self.crop.height)],
            "mask_shape": list(np.asarray(self.mask).shape),
        }
        if include_mask:
            payload["mask"] = np.asarray(self.mask, dtype=bool).tolist()
        return payload


@dataclass
class CatCropResult:
    """All whole-cat crops found in one source image."""

    source: str
    original_size: tuple[int, int]
    cats: list[CatCrop]

    def to_dict(self, include_mask: bool = False) -> dict[str, Any]:
        return {
            "source": self.source,
            "original_size": list(self.original_size),
            "count": len(self.cats),
            "cats": [cat.to_dict(include_mask=include_mask) for cat in self.cats],
        }


ImageSource = str | Path | Image.Image | np.ndarray

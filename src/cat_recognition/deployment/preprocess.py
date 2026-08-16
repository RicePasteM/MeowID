from __future__ import annotations

import glob
import math
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T
from torchvision.transforms import InterpolationMode

from .types import ImageSource, LoadedImage

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
KEYPOINT_NAMES = (
    "left_eye",
    "right_eye",
    "mouth",
    "left_ear1",
    "left_ear2",
    "left_ear3",
    "right_ear1",
    "right_ear2",
    "right_ear3",
)

ALIGNMENT_PROFILES = {
    "balanced_3pt": np.asarray(
        [[0.36, 0.43], [0.64, 0.43], [0.50, 0.64]],
        dtype=np.float64,
    ),
    "petface_tight_3pt": np.asarray(
        [
            [56.0 / 224.0, 114.75322978 / 224.0],
            [168.0 / 224.0, 114.58009847 / 224.0],
            [
                ((76.15839386 + 147.32220459) / 2.0) / 224.0,
                ((183.4698995 + 183.47365316) / 2.0) / 224.0,
            ],
        ],
        dtype=np.float64,
    ),
}


def _path_sources(path_text: str) -> list[Path]:
    path = Path(path_text).expanduser()
    if path.is_dir():
        return sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        )
    if path.is_file():
        return [path]
    matches = [Path(item) for item in sorted(glob.glob(str(path)))]
    return [item for item in matches if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS]


def load_images(source: ImageSource | Iterable[ImageSource]) -> list[LoadedImage]:
    """Load a file, directory, glob, PIL/numpy image, or a list of them."""

    if isinstance(source, (str, Path, Image.Image, np.ndarray)):
        sources: list[ImageSource] = [source]
    else:
        sources = list(source)

    loaded: list[LoadedImage] = []
    memory_index = 0
    for item in sources:
        if isinstance(item, (str, Path)):
            paths = _path_sources(str(item))
            if not paths:
                raise FileNotFoundError(f"No supported images found: {item}")
            for path in paths:
                with Image.open(path) as handle:
                    image = handle.convert("RGB").copy()
                loaded.append(LoadedImage(image=image, source=str(path.resolve())))
            continue
        if isinstance(item, Image.Image):
            loaded.append(
                LoadedImage(image=item.convert("RGB").copy(), source=f"memory:{memory_index}")
            )
            memory_index += 1
            continue
        array = np.asarray(item)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError(f"Expected an HWC RGB/RGBA numpy image, got {array.shape}")
        if array.dtype != np.uint8:
            if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0, 255).astype(np.uint8)
        loaded.append(
            LoadedImage(
                image=Image.fromarray(array[..., :3], mode="RGB"),
                source=f"memory:{memory_index}",
            )
        )
        memory_index += 1
    if not loaded:
        raise ValueError("No images were supplied")
    return loaded


def build_recognition_transform(
    image_size: int,
    resize_size: int,
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
):
    return T.Compose(
        [
            T.Resize(int(resize_size), interpolation=InterpolationMode.BICUBIC),
            T.CenterCrop(int(image_size)),
            T.ToTensor(),
            T.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def build_pose_transform(
    image_size: tuple[int, int],
    mean: tuple[float, float, float],
    std: tuple[float, float, float],
):
    return T.Compose(
        [
            T.Resize(tuple(int(value) for value in image_size)),
            T.ToTensor(),
            T.Normalize(mean=list(mean), std=list(std)),
        ]
    )


def estimate_similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("source and target must both have shape [N, 2]")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("landmarks contain non-finite values")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if source_variance < 1.0e-8:
        raise ValueError("degenerate core landmarks")
    covariance = target_centered.T @ source_centered / source.shape[0]
    left, singular_values, right_t = np.linalg.svd(covariance)
    signs = np.ones(2, dtype=np.float64)
    if np.linalg.det(left) * np.linalg.det(right_t) < 0:
        signs[-1] = -1.0
    rotation = left @ np.diag(signs) @ right_t
    scale = float(np.sum(singular_values * signs) / source_variance)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid similarity scale")
    linear = scale * rotation
    translation = target_mean - linear @ source_mean
    return np.column_stack([linear, translation]).astype(np.float64)


def estimate_keypoint_bbox_transform(keypoints: np.ndarray, output_size: int) -> np.ndarray:
    finite = keypoints[np.isfinite(keypoints).all(axis=1)]
    if len(finite) < 2:
        raise ValueError("fewer than two finite keypoints")
    minimum = finite.min(axis=0)
    maximum = finite.max(axis=0)
    center = (minimum + maximum) / 2.0
    side = float(max(maximum - minimum)) * 1.25
    if not math.isfinite(side) or side < 1.0:
        raise ValueError("degenerate keypoint bounding box")
    scale = output_size / side
    top_left = center - side / 2.0
    return np.asarray(
        [[scale, 0.0, -top_left[0] * scale], [0.0, scale, -top_left[1] * scale]],
        dtype=np.float64,
    )


def align_face(
    image: Image.Image,
    keypoints: np.ndarray,
    output_size: int = 256,
    profile: str = "petface_tight_3pt",
) -> Image.Image:
    """Align two eyes and mouth to the training template with black borders."""

    if profile not in ALIGNMENT_PROFILES:
        raise ValueError(f"Unknown alignment profile: {profile}")
    points = np.asarray(keypoints, dtype=np.float64).reshape(-1, 2)
    try:
        core = points[:3]
        eye_distance = float(np.linalg.norm(core[1] - core[0]))
        if not math.isfinite(eye_distance) or eye_distance < 1.0:
            raise ValueError("invalid eye distance")
        matrix = estimate_similarity(core, ALIGNMENT_PROFILES[profile] * output_size)
    except (ValueError, np.linalg.LinAlgError):
        matrix = estimate_keypoint_bbox_transform(points, output_size)

    forward = np.vstack([matrix, [0.0, 0.0, 1.0]])
    inverse = np.linalg.inv(forward)[:2]
    return image.convert("RGB").transform(
        (int(output_size), int(output_size)),
        Image.Transform.AFFINE,
        tuple(float(value) for value in inverse.reshape(-1)),
        resample=Image.Resampling.BICUBIC,
        fillcolor=(0, 0, 0),
    )


def stack_transformed(images: list[Image.Image], transform) -> torch.Tensor:
    return torch.stack([transform(image) for image in images], dim=0).contiguous()

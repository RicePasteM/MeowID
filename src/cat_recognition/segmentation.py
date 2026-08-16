from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from .deployment.preprocess import build_pose_transform, load_images, stack_transformed
from .deployment.types import CatCrop, CatCropResult, ImageSource, LoadedImage

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)
_ECSEG_ENGINE_PACKAGE = "_meowid_ecseg_engine"


class ECSegDeploymentGraph(nn.Module):
    def __init__(self, model: nn.Module, postprocessor: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.postprocessor = postprocessor

    def forward(self, images: torch.Tensor, original_sizes: torch.Tensor):
        return self.postprocessor(self.model(images), original_sizes)


def _load_isolated_engine(root: Path) -> ModuleType:
    """Load ECSeg's ``engine`` under a private name to coexist with ECPose."""

    existing = sys.modules.get(_ECSEG_ENGINE_PACKAGE)
    if existing is not None:
        return existing
    init_path = root / "engine/__init__.py"
    spec = importlib.util.spec_from_file_location(
        _ECSEG_ENGINE_PACKAGE,
        init_path,
        submodule_search_locations=[str(init_path.parent)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create an ECSeg module spec from {init_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[_ECSEG_ENGINE_PACKAGE] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(_ECSEG_ENGINE_PACKAGE, None)
        raise
    return module


def load_ecseg_graph(
    ecseg_root: str | Path,
    config_path: str | Path,
    checkpoint_path: str | Path,
) -> tuple[ECSegDeploymentGraph, tuple[int, int]]:
    root = Path(ecseg_root).resolve()
    if not (root / "engine").is_dir():
        raise FileNotFoundError(f"ECSeg source tree does not exist: {root}")
    _load_isolated_engine(root)
    core = importlib.import_module(f"{_ECSEG_ENGINE_PACKAGE}.core")
    cfg = core.YAMLConfig(str(config_path), resume=str(checkpoint_path))
    if "ViTAdapter" in cfg.yaml_cfg:
        cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.suffix.lower() == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(checkpoint_path), device="cpu")
    else:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
        state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    incompatible = cfg.model.load_state_dict(state, strict=False)
    # The official Hub safetensors intentionally omits these two derived
    # decoder constants; the YAML constructor has already initialized them.
    allowed_missing = {"decoder.up", "decoder.reg_scale"}
    unexpected = set(incompatible.unexpected_keys)
    missing = set(incompatible.missing_keys) - allowed_missing
    if missing or unexpected:
        raise RuntimeError(
            "ECSeg checkpoint is incompatible: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    graph = ECSegDeploymentGraph(cfg.model.deploy(), cfg.postprocessor.deploy()).eval()
    image_size = tuple(int(value) for value in cfg.yaml_cfg["eval_spatial_size"])
    return graph, image_size


def _output_size(value: int | tuple[int, int] | None) -> tuple[int, int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        if value <= 0:
            raise ValueError("output_size must be positive")
        return value, value
    if len(value) != 2 or min(value) <= 0:
        raise ValueError("output_size must be a positive int or (width, height)")
    return int(value[0]), int(value[1])


def _mask_bbox(mask: np.ndarray) -> np.ndarray | None:
    rows, columns = np.nonzero(mask)
    if rows.size == 0:
        return None
    return np.asarray(
        [columns.min(), rows.min(), columns.max() + 1, rows.max() + 1],
        dtype=np.float32,
    )


def make_cat_crop(
    image: Image.Image,
    mask: np.ndarray,
    box: np.ndarray,
    *,
    padding: float = 0.06,
    output_size: int | tuple[int, int] | None = 512,
    mask_background: bool = True,
) -> tuple[Image.Image, np.ndarray]:
    """Create a black-background, aspect-preserving whole-cat crop."""

    if padding < 0:
        raise ValueError("padding must be non-negative")
    image = image.convert("RGB")
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != (image.height, image.width):
        raise ValueError(
            f"mask shape {mask.shape} does not match image {(image.height, image.width)}"
        )
    detected = np.asarray(box, dtype=np.float32).reshape(4)
    segmented = _mask_bbox(mask)
    if segmented is None:
        raise ValueError("cannot crop an empty segmentation mask")
    # The union prevents a thin or disconnected predicted limb/tail from being
    # clipped by either the segmentation-derived or detector-derived box.
    bounds = np.asarray(
        [
            min(detected[0], segmented[0]),
            min(detected[1], segmented[1]),
            max(detected[2], segmented[2]),
            max(detected[3], segmented[3]),
        ],
        dtype=np.float32,
    )
    width = max(float(bounds[2] - bounds[0]), 1.0)
    height = max(float(bounds[3] - bounds[1]), 1.0)
    pad_x, pad_y = width * float(padding), height * float(padding)
    crop_box = np.asarray(
        [
            max(0, int(np.floor(bounds[0] - pad_x))),
            max(0, int(np.floor(bounds[1] - pad_y))),
            min(image.width, int(np.ceil(bounds[2] + pad_x))),
            min(image.height, int(np.ceil(bounds[3] + pad_y))),
        ],
        dtype=np.int64,
    )
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        raise ValueError(f"invalid crop box: {crop_box.tolist()}")

    pixels = np.asarray(image, dtype=np.uint8)
    if mask_background:
        pixels = np.where(mask[..., None], pixels, 0).astype(np.uint8)
    left, top, right, bottom = (int(value) for value in crop_box)
    crop = Image.fromarray(pixels[top:bottom, left:right], mode="RGB")

    target = _output_size(output_size)
    if target is None:
        return crop, crop_box
    target_width, target_height = target
    scale = min(target_width / crop.width, target_height / crop.height)
    resized_size = (
        max(1, round(crop.width * scale)),
        max(1, round(crop.height * scale)),
    )
    resized = crop.resize(resized_size, resample=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", target, color=(0, 0, 0))
    canvas.paste(
        resized,
        ((target_width - resized.width) // 2, (target_height - resized.height) // 2),
    )
    return canvas, crop_box


class CatCropper:
    """ECSeg whole-cat segmentation and contain-style crop SDK."""

    def __init__(
        self,
        model: str | Path | None = None,
        *,
        config: str | Path | None = None,
        ecseg_root: str | Path | None = None,
        device: str | None = None,
        threshold: float | None = None,
        cat_label: int | None = None,
        batch_size: int = 1,
        half: bool = False,
    ) -> None:
        package_root = Path(__file__).resolve().parents[2]
        artifact_root = (
            Path(model).expanduser().resolve()
            if model is not None and Path(model).expanduser().is_dir()
            else package_root / "artifacts/ECSeg"
        )
        manifest_path = artifact_root / "deployment.json"
        manifest: dict[str, Any] = {}
        if manifest_path.is_file():
            with manifest_path.open("r", encoding="utf-8") as handle:
                manifest = json.load(handle)
        self.repo_root = self._resolve_repo_root(package_root, artifact_root)
        self.ecseg_root = (
            Path(ecseg_root).expanduser().resolve()
            if ecseg_root is not None
            else self.repo_root / "third_party/EdgeCrafter/ecdetseg"
        )
        configured_model = manifest.get("model", "ecseg_x.safetensors")
        self.model_path = (
            Path(model).expanduser().resolve()
            if model is not None and Path(model).expanduser().is_file()
            else (artifact_root / configured_model).resolve()
        )
        configured_config = manifest.get(
            "config", "third_party/EdgeCrafter/ecdetseg/configs/ecseg/ecseg_x.yml"
        )
        self.config_path = (
            Path(config).expanduser().resolve()
            if config is not None
            else (self.repo_root / configured_config).resolve()
        )
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ECSeg checkpoint does not exist: {self.model_path}")
        if not self.config_path.is_file():
            raise FileNotFoundError(f"ECSeg config does not exist: {self.config_path}")

        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.threshold = float(
            threshold if threshold is not None else manifest.get("threshold", 0.4)
        )
        self.cat_label = int(
            cat_label if cat_label is not None else manifest.get("cat_label", 15)
        )
        self.batch_size = max(1, int(batch_size))
        self.half = bool(half and self.device.type == "cuda")
        self.graph, self.image_size = load_ecseg_graph(
            self.ecseg_root, self.config_path, self.model_path
        )
        self.graph = self.graph.to(self.device)
        self.graph.eval()
        preprocessing = manifest.get("preprocessing", {})
        self.transform = build_pose_transform(
            tuple(int(value) for value in preprocessing.get("image_size", self.image_size)),
            tuple(preprocessing.get("mean", _IMAGENET_MEAN)),
            tuple(preprocessing.get("std", _IMAGENET_STD)),
        )

    @staticmethod
    def _resolve_repo_root(package_root: Path, artifact_root: Path) -> Path:
        for candidate in (artifact_root.parent.parent, artifact_root.parent, package_root):
            if (candidate / "third_party/EdgeCrafter/ecdetseg").is_dir():
                return candidate.resolve()
        return package_root.resolve()

    @torch.inference_mode()
    def _infer_chunk(
        self,
        loaded: list[LoadedImage],
        *,
        threshold: float,
        top_k: int | None,
        padding: float,
        output_size: int | tuple[int, int] | None,
        mask_background: bool,
    ) -> list[CatCropResult]:
        tensors = stack_transformed([item.image for item in loaded], self.transform)
        tensors = tensors.to(self.device, dtype=torch.float32, non_blocking=True)
        original_sizes = torch.as_tensor(
            [[item.image.width, item.image.height] for item in loaded],
            device=self.device,
            dtype=torch.float32,
        )
        precision_context = (
            torch.autocast(device_type="cuda", dtype=torch.float16)
            if self.half
            else nullcontext()
        )
        with precision_context:
            labels, boxes, scores, masks = self.graph(tensors, original_sizes)
        outputs: list[CatCropResult] = []
        for index, item in enumerate(loaded):
            valid = torch.nonzero(
                (labels[index] == self.cat_label)
                & torch.isfinite(scores[index])
                & (scores[index] >= float(threshold)),
                as_tuple=False,
            ).flatten()
            if valid.numel():
                order = torch.argsort(scores[index, valid], descending=True)
                valid = valid[order]
                if top_k is not None:
                    valid = valid[: max(0, int(top_k))]
            cats: list[CatCrop] = []
            if valid.numel():
                resized_masks = F.interpolate(
                    masks[index, valid].unsqueeze(1),
                    size=(item.image.height, item.image.width),
                    mode="bilinear",
                    align_corners=False,
                )[:, 0]
                resized_masks = resized_masks > 0.0
                for offset, query_index in enumerate(valid.tolist()):
                    mask = resized_masks[offset].cpu().numpy().astype(bool, copy=False)
                    if not mask.any():
                        continue
                    box = boxes[index, query_index].float().cpu().numpy()
                    crop, crop_box = make_cat_crop(
                        item.image,
                        mask,
                        box,
                        padding=padding,
                        output_size=output_size,
                        mask_background=mask_background,
                    )
                    cats.append(
                        CatCrop(
                            score=float(scores[index, query_index].item()),
                            label=int(labels[index, query_index].item()),
                            box=box,
                            crop_box=crop_box,
                            mask=mask,
                            crop=crop,
                        )
                    )
            outputs.append(
                CatCropResult(
                    source=item.source,
                    original_size=(item.image.width, item.image.height),
                    cats=cats,
                )
            )
        return outputs

    def crop(
        self,
        source: ImageSource | list[ImageSource],
        *,
        threshold: float | None = None,
        top_k: int | None = 1,
        padding: float = 0.06,
        output_size: int | tuple[int, int] | None = 512,
        mask_background: bool = True,
    ) -> list[CatCropResult]:
        """Segment and crop cats; one result object is returned per input image."""

        loaded = load_images(source)
        resolved_threshold = self.threshold if threshold is None else float(threshold)
        results: list[CatCropResult] = []
        for start in range(0, len(loaded), self.batch_size):
            results.extend(
                self._infer_chunk(
                    loaded[start : start + self.batch_size],
                    threshold=resolved_threshold,
                    top_k=top_k,
                    padding=padding,
                    output_size=output_size,
                    mask_background=mask_background,
                )
            )
        return results

    segment = crop
    __call__ = crop

    def info(self) -> dict[str, Any]:
        return {
            "model": str(self.model_path),
            "config": str(self.config_path),
            "device": str(self.device),
            "threshold": self.threshold,
            "cat_label": self.cat_label,
            "image_size": list(self.image_size),
            "half": self.half,
        }

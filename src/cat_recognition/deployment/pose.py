from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .runtime import OnnxRuntimeSession, TensorRTRuntimeSession
from .types import PoseDetection


class ECPoseDeploymentGraph(nn.Module):
    def __init__(self, model: nn.Module, postprocessor: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.postprocessor = postprocessor

    def forward(
        self,
        images: torch.Tensor,
        orig_target_sizes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.postprocessor(self.model(images), orig_target_sizes)


def load_ecpose_graph(
    ecpose_root: str | Path,
    config_path: str | Path,
    checkpoint_path: str | Path,
) -> tuple[ECPoseDeploymentGraph, tuple[int, int]]:
    root = Path(ecpose_root).resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from engine.core import YAMLConfig
    except ImportError as exc:
        raise ImportError(f"Could not import ECPose from {root}") from exc

    cfg = YAMLConfig(str(config_path), resume=str(checkpoint_path))
    if "ViTAdapter" in cfg.yaml_cfg:
        cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state, strict=True)
    graph = ECPoseDeploymentGraph(
        cfg.model.deploy(),
        cfg.postprocessor.deploy(),
    ).eval()
    image_size = tuple(int(value) for value in cfg.yaml_cfg["eval_spatial_size"])
    return graph, image_size


def select_detections(
    scores: np.ndarray,
    labels: np.ndarray,
    keypoints: np.ndarray,
    threshold: float,
    face_label: int = 1,
) -> list[PoseDetection | None]:
    detections: list[PoseDetection | None] = []
    for row_scores, row_labels, row_keypoints in zip(scores, labels, keypoints):
        valid = np.flatnonzero(np.asarray(row_labels) == int(face_label))
        if valid.size == 0:
            detections.append(None)
            continue
        best_index = int(valid[np.argmax(np.asarray(row_scores)[valid])])
        score = float(row_scores[best_index])
        if not np.isfinite(score) or score < float(threshold):
            detections.append(None)
            continue
        points = np.asarray(row_keypoints[best_index], dtype=np.float32).reshape(-1, 2)
        if points.shape[0] < 3 or not np.isfinite(points[:3]).all():
            detections.append(None)
            continue
        detections.append(
            PoseDetection(keypoints=points, score=score, label=int(row_labels[best_index]))
        )
    return detections


class TorchPoseBackend:
    def __init__(
        self,
        ecpose_root: str | Path,
        config_path: str | Path,
        checkpoint_path: str | Path,
        device: str,
        half: bool = False,
    ) -> None:
        self.graph, self.image_size = load_ecpose_graph(ecpose_root, config_path, checkpoint_path)
        self.device = torch.device(device)
        self.half = bool(half and self.device.type == "cuda")
        self.graph = self.graph.to(self.device)
        if self.half:
            self.graph.half()
        self.graph.eval()

    @torch.inference_mode()
    def run(self, images: torch.Tensor, original_sizes: np.ndarray):
        dtype = torch.float16 if self.half else torch.float32
        images = images.to(self.device, dtype=dtype, non_blocking=True)
        sizes = torch.as_tensor(original_sizes, device=self.device, dtype=dtype)
        scores, labels, keypoints = self.graph(images, sizes)
        return (
            scores.float().cpu().numpy(),
            labels.cpu().numpy(),
            keypoints.float().cpu().numpy(),
        )


class OnnxPoseBackend:
    image_size = (640, 640)

    def __init__(self, model_path: str | Path, device: str) -> None:
        self.session = OnnxRuntimeSession(model_path, device=device)

    def run(self, images: torch.Tensor, original_sizes: np.ndarray):
        outputs = self.session.run(
            {
                "images": images.numpy().astype(np.float32, copy=False),
                "orig_target_sizes": np.asarray(original_sizes, dtype=np.float32),
            }
        )
        return outputs["scores"], outputs["labels"], outputs["keypoints"]


class TensorRTPoseBackend:
    image_size = (640, 640)

    def __init__(self, engine_path: str | Path, device: str) -> None:
        self.session = TensorRTRuntimeSession(engine_path, device=device)

    def run(self, images: torch.Tensor, original_sizes: np.ndarray):
        outputs = self.session.run(
            {
                "images": images.numpy().astype(np.float32, copy=False),
                "orig_target_sizes": np.asarray(original_sizes, dtype=np.float32),
            }
        )
        return outputs["scores"], outputs["labels"], outputs["keypoints"]


def build_pose_backend(
    backend: str,
    *,
    model_path: str | Path,
    device: str,
    ecpose_root: str | Path | None = None,
    config_path: str | Path | None = None,
    half: bool = False,
):
    backend = backend.lower()
    if backend == "torch":
        if ecpose_root is None or config_path is None:
            raise ValueError("Torch ECPose requires ecpose_root and config_path")
        return TorchPoseBackend(ecpose_root, config_path, model_path, device=device, half=half)
    if backend == "onnx":
        return OnnxPoseBackend(model_path, device=device)
    if backend in {"tensorrt", "trt"}:
        return TensorRTPoseBackend(model_path, device=device)
    raise ValueError(f"Unsupported pose backend: {backend}")

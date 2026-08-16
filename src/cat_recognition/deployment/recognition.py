from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from cat_recognition.config import ConfigNode, load_config
from cat_recognition.models import build_meowid_deployment_model

from .runtime import OnnxRuntimeSession, TensorRTRuntimeSession


class MeowIDDeploymentGraph(nn.Module):
    """Static two-input/two-output graph shared by all deployment backends."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        body_images: torch.Tensor,
        face_images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.model.encode_deployment(body_images, face_images)


def load_deployment_graph(
    checkpoint_path: str | Path,
    config_path: str | Path | None = None,
) -> tuple[MeowIDDeploymentGraph, ConfigNode, dict]:
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    if "model" not in checkpoint:
        raise ValueError(f"Checkpoint has no model state: {checkpoint_path}")
    if config_path is not None:
        cfg = load_config(config_path)
    elif isinstance(checkpoint.get("config"), dict):
        cfg = ConfigNode(checkpoint["config"])
    else:
        raise ValueError("A config path is required when the checkpoint has no embedded config")
    model = build_meowid_deployment_model(cfg, checkpoint["model"])
    return MeowIDDeploymentGraph(model).eval(), cfg, checkpoint


class TorchRecognitionBackend:
    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path | None,
        device: str,
        half: bool = False,
    ) -> None:
        graph, self.cfg, self.checkpoint = load_deployment_graph(checkpoint_path, config_path)
        self.device = torch.device(device)
        self.half = bool(half and self.device.type == "cuda")
        self.graph = graph.to(self.device)
        if self.half:
            self.graph.half()
        self.graph.eval()

    @torch.inference_mode()
    def run(self, body_images: torch.Tensor, face_images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        dtype = torch.float16 if self.half else torch.float32
        body_images = body_images.to(self.device, dtype=dtype, non_blocking=True)
        face_images = face_images.to(self.device, dtype=dtype, non_blocking=True)
        body_embeddings, face_embeddings = self.graph(body_images, face_images)
        return (
            body_embeddings.float().cpu().numpy(),
            face_embeddings.float().cpu().numpy(),
        )


class OnnxRecognitionBackend:
    def __init__(self, model_path: str | Path, device: str) -> None:
        self.session = OnnxRuntimeSession(model_path, device=device)

    def run(self, body_images: torch.Tensor, face_images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        outputs = self.session.run(
            {
                "body_images": body_images.numpy().astype(np.float32, copy=False),
                "face_images": face_images.numpy().astype(np.float32, copy=False),
            }
        )
        return outputs["body_embeddings"], outputs["face_embeddings"]


class TensorRTRecognitionBackend:
    def __init__(self, engine_path: str | Path, device: str) -> None:
        self.session = TensorRTRuntimeSession(engine_path, device=device)

    def run(self, body_images: torch.Tensor, face_images: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
        outputs = self.session.run(
            {
                "body_images": body_images.numpy().astype(np.float32, copy=False),
                "face_images": face_images.numpy().astype(np.float32, copy=False),
            }
        )
        return outputs["body_embeddings"], outputs["face_embeddings"]


def build_recognition_backend(
    backend: str,
    *,
    model_path: str | Path,
    device: str,
    config_path: str | Path | None = None,
    half: bool = False,
):
    backend = backend.lower()
    if backend == "torch":
        return TorchRecognitionBackend(model_path, config_path, device=device, half=half)
    if backend == "onnx":
        return OnnxRecognitionBackend(model_path, device=device)
    if backend in {"tensorrt", "trt"}:
        return TensorRTRecognitionBackend(model_path, device=device)
    raise ValueError(f"Unsupported recognition backend: {backend}")

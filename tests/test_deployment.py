from __future__ import annotations

import inspect

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from cat_recognition.deployment.export import (
    export_deployment,
    export_ecpose_onnx,
    export_meowid_onnx,
)
from cat_recognition.deployment.pose import select_detections
from cat_recognition.deployment.preprocess import align_face
from cat_recognition.deployment.registry import EmbeddingRegistry
from cat_recognition.deployment.types import EmbeddingResult, PoseDetection
from cat_recognition.models.meowid_base import MeowIDBase
from cat_recognition.segmentation import make_cat_crop


class TinyExpert(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(dim, dim, bias=False)
        self.calls = 0

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return F.normalize(self.projection(images), dim=1)


def _result(route: str, seed: int) -> EmbeddingResult:
    generator = np.random.default_rng(seed)
    body = generator.normal(size=8).astype(np.float32)
    body /= np.linalg.norm(body)
    face = None
    detection = None
    if route == "face":
        face = generator.normal(size=8).astype(np.float32)
        face /= np.linalg.norm(face)
        detection = PoseDetection(np.zeros((9, 2), dtype=np.float32), score=0.9, label=1)
    return EmbeddingResult(
        source=f"{seed}.jpg",
        embedding=face if face is not None else body,
        body_embedding=body,
        face_embedding=face,
        route=route,
        face=detection,
    )


def test_deployment_graph_reuses_body_embedding() -> None:
    body = TinyExpert(4)
    face = TinyExpert(4)
    model = MeowIDBase(
        body_expert=body,
        face_expert=face,
        best_body_hint_expert=None,
        body_head=None,
        face_head=None,
        face_label_lookup=torch.empty(0, dtype=torch.long),
        embedding_dim=4,
        hint_hidden_dim=4,
        dynamic_best_body_hint=False,
    ).eval()
    body_output, face_output = model.encode_deployment(
        torch.randn(3, 4),
        torch.randn(3, 4),
    )
    assert body.calls == 1
    assert face.calls == 1
    assert body_output.shape == face_output.shape == (3, 4)
    assert torch.allclose(body_output.norm(dim=1), torch.ones(3), atol=1e-6)
    assert torch.allclose(face_output.norm(dim=1), torch.ones(3), atol=1e-6)


def test_onnxslim_is_enabled_by_default() -> None:
    for function in (export_meowid_onnx, export_ecpose_onnx, export_deployment):
        parameter = inspect.signature(function).parameters["use_onnxslim"]
        assert parameter.default is True


def test_registry_keeps_route_spaces_separate_and_roundtrips(tmp_path) -> None:
    registry = EmbeddingRegistry(dim=8, path=tmp_path)
    face_result = _result("face", 1)
    body_result = _result("body", 2)
    assert registry.add("mimi", [face_result, body_result]) == 3
    assert registry.counts() == {
        "cats": 1,
        "vectors": 3,
        "body_vectors": 2,
        "face_vectors": 1,
    }
    assert registry.search(face_result.face_embedding, "face")[0].cat_id == "mimi"
    assert registry.search(body_result.body_embedding, "body")[0].cat_id == "mimi"
    registry.save()
    restored = EmbeddingRegistry(dim=8, path=tmp_path)
    assert restored.counts() == registry.counts()
    assert restored.search(face_result.face_embedding, "face")[0].score > 0.999


def test_pose_selection_filters_class_and_threshold() -> None:
    scores = np.asarray([[0.99, 0.8, 0.2]], dtype=np.float32)
    labels = np.asarray([[0, 1, 1]], dtype=np.int64)
    keypoints = np.zeros((1, 3, 9, 2), dtype=np.float32)
    detection = select_detections(scores, labels, keypoints, threshold=0.4, face_label=1)[0]
    assert detection is not None
    assert detection.label == 1
    assert np.isclose(detection.score, 0.8)


def test_petface_alignment_uses_black_border() -> None:
    image = Image.new("RGB", (80, 80), color=(255, 255, 255))
    # A large source triangle maps to the canonical face at a small scale, so
    # the requested 256-pixel output crop extends beyond this 80-pixel source.
    source = np.asarray([[10, 30], [70, 30], [40, 65]], dtype=np.float32)
    keypoints = np.concatenate([source, np.tile(source[-1], (6, 1))], axis=0)
    aligned = np.asarray(align_face(image, keypoints, 256, "petface_tight_3pt"))
    assert aligned.shape == (256, 256, 3)
    assert np.any(np.all(aligned == 0, axis=2))


def test_whole_cat_crop_masks_background_and_uses_contain() -> None:
    pixels = np.full((60, 120, 3), 255, dtype=np.uint8)
    pixels[..., 0] = 120
    image = Image.fromarray(pixels)
    mask = np.zeros((60, 120), dtype=bool)
    mask[15:45, 20:100] = True
    crop, crop_box = make_cat_crop(
        image,
        mask,
        np.asarray([20, 15, 100, 45], dtype=np.float32),
        padding=0.0,
        output_size=64,
    )
    array = np.asarray(crop)
    assert crop.size == (64, 64)
    assert crop_box.tolist() == [20, 15, 100, 45]
    assert np.all(array[0] == 0) and np.all(array[-1] == 0)
    assert np.any(array[24:40] != 0)


def test_whole_cat_crop_can_preserve_native_crop_size() -> None:
    image = Image.new("RGB", (40, 30), color=(10, 20, 30))
    mask = np.zeros((30, 40), dtype=bool)
    mask[5:25, 8:32] = True
    crop, _ = make_cat_crop(
        image,
        mask,
        np.asarray([8, 5, 32, 25], dtype=np.float32),
        padding=0.0,
        output_size=None,
    )
    assert crop.size == (24, 20)

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from .deployment.pose import build_pose_backend, select_detections
from .deployment.preprocess import (
    align_face,
    build_pose_transform,
    build_recognition_transform,
    load_images,
    stack_transformed,
)
from .deployment.recognition import build_recognition_backend
from .deployment.registry import EmbeddingRegistry
from .deployment.types import (
    CatCropResult,
    EmbeddingResult,
    ImageSource,
    LoadedImage,
    PoseDetection,
    PredictionResult,
)

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def _normalize_embedding(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float32)
    return value / max(float(np.linalg.norm(value)), 1.0e-12)


class MeowID:
    """Unified cat-face identification API.

    The object owns ECPose, alignment, the MeowID-Base dual expert, hard-route
    selection, and a persistent registration gallery.  The public methods are
    backend-independent; choosing ``torch``, ``onnx`` or ``tensorrt`` changes
    only the two inference runtimes.
    """

    def __init__(
        self,
        model: str | Path | None = None,
        *,
        backend: str | None = None,
        pose_model: str | Path | None = None,
        device: str | None = None,
        registry: str | Path | EmbeddingRegistry | None = None,
        config: str | Path | None = None,
        pose_config: str | Path | None = None,
        ecpose_root: str | Path | None = None,
        face_threshold: float | None = None,
        face_label: int | None = None,
        alignment: str | None = None,
        batch_size: int = 4,
        half: bool = False,
        segmenter_model: str | Path | None = None,
        segmenter_threshold: float = 0.4,
    ) -> None:
        package_root = Path(__file__).resolve().parents[2]
        manifest, artifact_root = self._load_manifest(model)
        self.repo_root = self._resolve_repo_root(package_root, artifact_root)
        backend = self._resolve_backend(backend, model)
        self.backend = backend
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.half = bool(half)
        self.batch_size = max(1, int(batch_size))
        self.segmenter_model = segmenter_model
        self.segmenter_threshold = float(segmenter_threshold)
        self._cat_cropper = None
        self.face_label = int(
            face_label if face_label is not None else manifest.get("face_label", 1)
        )
        self.manifest = manifest
        self.artifact_root = artifact_root

        model_path = self._resolve_model_path(
            explicit=model if model is not None and Path(model).is_file() else None,
            manifest=manifest,
            artifact_root=artifact_root,
            backend=backend,
            kind="recognition",
            fallback=self._default_recognition_path(backend),
        )
        pose_model_path = self._resolve_model_path(
            explicit=pose_model,
            manifest=manifest,
            artifact_root=artifact_root,
            backend=backend,
            kind="pose",
            fallback=self._default_pose_path(backend),
        )
        self.model_path = model_path
        self.pose_model_path = pose_model_path

        config_path = Path(config) if config is not None else None
        if config_path is None:
            default_config = self.repo_root / "configs/experiments/meowid_base.yaml"
            config_path = default_config if default_config.exists() else None
        pose_config_path = (
            Path(pose_config)
            if pose_config is not None
            else self.repo_root
            / "third_party/EdgeCrafter/ecpose/configs/ecpose/ecpose_x_cat_face.yml"
        )
        ecpose_path = (
            Path(ecpose_root)
            if ecpose_root is not None
            else self.repo_root / "third_party/EdgeCrafter/ecpose"
        )

        self.recognition_backend = build_recognition_backend(
            backend,
            model_path=model_path,
            device=self.device,
            config_path=config_path if backend == "torch" else None,
            half=half,
        )
        self.pose_backend = build_pose_backend(
            backend,
            model_path=pose_model_path,
            device=self.device,
            ecpose_root=ecpose_path if backend == "torch" else None,
            config_path=pose_config_path if backend == "torch" else None,
            half=half,
        )

        preprocessing = manifest.get("preprocessing", {})
        pose_preprocess = preprocessing.get("pose", {})
        body_preprocess = preprocessing.get("body", {})
        face_preprocess = preprocessing.get("face", {})
        pose_size = tuple(
            int(value)
            for value in pose_preprocess.get("image_size", self.pose_backend.image_size)
        )
        self.body_image_size = int(body_preprocess.get("image_size", 256))
        self.face_image_size = int(face_preprocess.get("image_size", 256))
        self.pose_transform = build_pose_transform(
            pose_size,
            tuple(pose_preprocess.get("mean", _IMAGENET_MEAN)),
            tuple(pose_preprocess.get("std", _IMAGENET_STD)),
        )
        self.body_transform = build_recognition_transform(
            self.body_image_size,
            int(body_preprocess.get("resize_size", 292)),
            tuple(body_preprocess.get("mean", _IMAGENET_MEAN)),
            tuple(body_preprocess.get("std", _IMAGENET_STD)),
        )
        self.face_transform = build_recognition_transform(
            self.face_image_size,
            int(face_preprocess.get("resize_size", 256)),
            tuple(face_preprocess.get("mean", _IMAGENET_MEAN)),
            tuple(face_preprocess.get("std", _IMAGENET_STD)),
        )
        self.face_threshold = float(
            face_threshold
            if face_threshold is not None
            else manifest.get("face_threshold", 0.4)
        )
        self.alignment = str(
            alignment if alignment is not None else manifest.get("alignment", "petface_tight_3pt")
        )

        if isinstance(registry, EmbeddingRegistry):
            self.registry = registry
        else:
            self.registry = EmbeddingRegistry(
                dim=int(manifest.get("embedding_dim", 512)),
                path=registry,
            )

    @staticmethod
    def _resolve_repo_root(package_root: Path, artifact_root: Path | None) -> Path:
        candidates = [package_root]
        if artifact_root is not None:
            candidates = [artifact_root.parent.parent, artifact_root.parent, *candidates]
        for candidate in candidates:
            if (candidate / "third_party/EdgeCrafter/ecpose").is_dir():
                return candidate.resolve()
        return package_root.resolve()

    @staticmethod
    def _resolve_backend(backend: str | None, model: str | Path | None) -> str:
        if backend is None and model is not None and Path(model).is_file():
            suffix = Path(model).suffix.lower()
            backend = {".onnx": "onnx", ".engine": "tensorrt", ".plan": "tensorrt"}.get(
                suffix,
                "torch",
            )
        backend = (backend or "torch").lower()
        if backend == "trt":
            backend = "tensorrt"
        if backend not in {"torch", "onnx", "tensorrt"}:
            raise ValueError(f"Unsupported backend: {backend}")
        return backend

    @staticmethod
    def _load_manifest(model: str | Path | None) -> tuple[dict[str, Any], Path | None]:
        if model is None:
            return {}, None
        path = Path(model).expanduser()
        if not path.is_dir():
            return {}, None
        manifest_path = path / "deployment.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Deployment directory has no deployment.json: {path}")
        with manifest_path.open("r", encoding="utf-8") as handle:
            return json.load(handle), path.resolve()

    @staticmethod
    def _resolve_model_path(
        *,
        explicit: str | Path | None,
        manifest: dict[str, Any],
        artifact_root: Path | None,
        backend: str,
        kind: str,
        fallback: Path,
    ) -> Path:
        if explicit is not None:
            path = Path(explicit).expanduser()
        else:
            configured = manifest.get("models", {}).get(backend, {}).get(kind)
            if configured is not None:
                path = Path(configured)
                if not path.is_absolute() and artifact_root is not None:
                    path = artifact_root / path
            else:
                path = fallback
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{kind.capitalize()} model does not exist: {path}")
        return path

    def _default_recognition_path(self, backend: str) -> Path:
        paths = {
            "torch": self.repo_root / "artifacts/MeowID-Base/meowid_base.pth",
            "onnx": self.repo_root / "artifacts/MeowID-Base/meowid_base.onnx",
            "tensorrt": self.repo_root / "artifacts/MeowID-Base/meowid_base.engine",
        }
        return paths[backend]

    def _default_pose_path(self, backend: str) -> Path:
        paths = {
            "torch": self.repo_root / "artifacts/MeowID-Base/ecpose.pth",
            "onnx": self.repo_root / "artifacts/MeowID-Base/ecpose.onnx",
            "tensorrt": self.repo_root / "artifacts/MeowID-Base/ecpose.engine",
        }
        return paths[backend]

    def _detect_loaded(self, loaded: list[LoadedImage]) -> list[PoseDetection | None]:
        detections: list[PoseDetection | None] = []
        for start in range(0, len(loaded), self.batch_size):
            chunk = loaded[start : start + self.batch_size]
            tensors = stack_transformed([item.image for item in chunk], self.pose_transform)
            original_sizes = np.asarray(
                [[item.image.width, item.image.height] for item in chunk],
                dtype=np.float32,
            )
            scores, labels, keypoints = self.pose_backend.run(tensors, original_sizes)
            detections.extend(
                select_detections(
                    scores,
                    labels,
                    keypoints,
                    threshold=self.face_threshold,
                    face_label=self.face_label,
                )
            )
        return detections

    def detect(self, source: ImageSource | Iterable[ImageSource]) -> list[PoseDetection | None]:
        """Detect the best cat-face keypoint instance in each input image."""

        return self._detect_loaded(load_images(source))

    def align(self, source: ImageSource | Iterable[ImageSource]) -> list[Image.Image | None]:
        """Detect and align faces with the training profile and black borders."""

        loaded = load_images(source)
        detections = self._detect_loaded(loaded)
        return [
            None
            if detection is None
            else align_face(
                item.image,
                detection.keypoints,
                output_size=self.face_image_size,
                profile=self.alignment,
            )
            for item, detection in zip(loaded, detections)
        ]

    def embed(
        self,
        source: ImageSource | Iterable[ImageSource],
        *,
        return_aligned: bool = False,
    ) -> list[EmbeddingResult]:
        """Extract body/face diagnostics and the final hard-routed embedding."""

        loaded = load_images(source)
        detections = self._detect_loaded(loaded)
        aligned_faces: list[Image.Image | None] = []
        for item, detection in zip(loaded, detections):
            aligned_faces.append(
                None
                if detection is None
                else align_face(
                    item.image,
                    detection.keypoints,
                    output_size=self.face_image_size,
                    profile=self.alignment,
                )
            )

        results: list[EmbeddingResult] = []
        placeholder = Image.new("RGB", (self.face_image_size, self.face_image_size), (0, 0, 0))
        for start in range(0, len(loaded), self.batch_size):
            chunk = loaded[start : start + self.batch_size]
            chunk_detections = detections[start : start + self.batch_size]
            chunk_faces = aligned_faces[start : start + self.batch_size]
            body_tensors = stack_transformed([item.image for item in chunk], self.body_transform)
            face_tensors = stack_transformed(
                [face if face is not None else placeholder for face in chunk_faces],
                self.face_transform,
            )
            body_embeddings, face_embeddings = self.recognition_backend.run(
                body_tensors,
                face_tensors,
            )
            for index, (item, detection, aligned) in enumerate(
                zip(chunk, chunk_detections, chunk_faces)
            ):
                has_face = detection is not None
                body_embedding = _normalize_embedding(body_embeddings[index])
                face_embedding = (
                    _normalize_embedding(face_embeddings[index]) if has_face else None
                )
                route = "face" if has_face else "body"
                routed = face_embedding if has_face else body_embedding
                results.append(
                    EmbeddingResult(
                        source=item.source,
                        embedding=np.asarray(routed, dtype=np.float32),
                        body_embedding=body_embedding,
                        face_embedding=face_embedding,
                        route=route,
                        face=detection,
                        aligned_face=aligned if return_aligned else None,
                    )
                )
        return results

    extract_embeddings = embed

    def register(
        self,
        cat_id: str,
        source: ImageSource | Iterable[ImageSource],
        *,
        metadata: dict[str, Any] | None = None,
        replace: bool = False,
        save: bool = True,
    ) -> dict[str, Any]:
        """Register one cat from one or more images in both valid route spaces."""

        results = self.embed(source)
        if replace:
            self.registry.remove(str(cat_id))
        added = self.registry.add(str(cat_id), results, metadata=metadata)
        if save and self.registry.path is not None:
            self.registry.save()
        return {
            "cat_id": str(cat_id),
            "images": len(results),
            "vectors_added": added,
            "face_images": sum(result.face_detected for result in results),
            "registry": self.registry.counts(),
        }

    enroll = register

    def search(
        self,
        source: ImageSource | Iterable[ImageSource],
        *,
        top_k: int = 5,
        aggregation: str = "max",
        threshold: float | None = None,
    ) -> list[PredictionResult]:
        """Search the gallery in the route space selected independently per image."""

        outputs = []
        for result in self.embed(source):
            matches = self.registry.search(
                result.embedding,
                route=result.route,
                top_k=top_k,
                aggregation=aggregation,
            )
            unknown = bool(threshold is not None and (not matches or matches[0].score < threshold))
            outputs.append(PredictionResult(embedding=result, matches=matches, unknown=unknown))
        return outputs

    retrieve = search

    def crop_cats(
        self,
        source: ImageSource | Iterable[ImageSource],
        *,
        threshold: float | None = None,
        top_k: int | None = 1,
        padding: float = 0.06,
        output_size: int | tuple[int, int] | None = 512,
        mask_background: bool = True,
    ) -> list[CatCropResult]:
        """Return ECSeg whole-cat masks and contain-style crops.

        ECSeg is loaded lazily so recognition-only services pay no additional
        startup time or GPU-memory cost.  Use :class:`CatCropper` directly when
        the recognition and ECPose models are not needed.
        """

        if self._cat_cropper is None:
            from .segmentation import CatCropper

            self._cat_cropper = CatCropper(
                self.segmenter_model,
                device=self.device,
                threshold=self.segmenter_threshold,
                batch_size=self.batch_size,
                half=self.half,
            )
        return self._cat_cropper.crop(
            source,
            threshold=threshold,
            top_k=top_k,
            padding=padding,
            output_size=output_size,
            mask_background=mask_background,
        )

    def predict(
        self,
        source: ImageSource | Iterable[ImageSource],
        *,
        top_k: int = 5,
        aggregation: str = "max",
        threshold: float | None = None,
    ) -> list[PredictionResult]:
        return self.search(
            source,
            top_k=top_k,
            aggregation=aggregation,
            threshold=threshold,
        )

    def __call__(self, source: ImageSource | Iterable[ImageSource], **kwargs):
        return self.predict(source, **kwargs)

    def remove(self, cat_id: str, *, save: bool = True) -> int:
        removed = self.registry.remove(str(cat_id))
        if save and self.registry.path is not None:
            self.registry.save()
        return removed

    def save_registry(self, path: str | Path | None = None) -> Path:
        return self.registry.save(path)

    def load_registry(self, path: str | Path) -> MeowID:
        self.registry.load(path)
        return self

    def info(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "device": self.device,
            "recognition_model": str(self.model_path),
            "pose_model": str(self.pose_model_path),
            "alignment": self.alignment,
            "face_threshold": self.face_threshold,
            "face_label": self.face_label,
            "segmenter_model": str(
                self.segmenter_model
                if self.segmenter_model is not None
                else self.repo_root / "artifacts/ECSeg"
            ),
            "segmenter_loaded": self._cat_cropper is not None,
            "registry": self.registry.counts(),
        }

    def export(self, format: str = "onnx", **kwargs) -> dict[str, Any]:
        """Export from the Torch source artifacts used by this repository."""

        from .deployment.export import export_deployment

        return export_deployment(format=format, repo_root=self.repo_root, **kwargs)

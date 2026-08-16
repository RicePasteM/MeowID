from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import torch

from .pose import load_ecpose_graph
from .recognition import load_deployment_graph


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_info(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": os.path.relpath(path, root),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _finalize_onnx(
    raw_path: Path,
    output_path: Path,
    *,
    use_onnxslim: bool,
) -> None:
    """Optionally slim, validate, and atomically publish an ONNX model."""

    import onnx

    temporary = output_path.with_name(f".{output_path.name}.tmp")
    try:
        model = onnx.load(str(raw_path), load_external_data=True)
        if use_onnxslim:
            try:
                import onnxslim
            except ImportError as exc:
                raise ImportError(
                    "ONNXSlim is enabled by default; install it with "
                    "`pip install -e .[onnx]` or pass use_onnxslim=False"
                ) from exc
            model = onnxslim.slim(model)
            if model is None:
                raise RuntimeError(f"ONNXSlim did not return a model for {raw_path}")
        onnx.checker.check_model(model)
        onnx.save(model, str(temporary))
        os.replace(temporary, output_path)
    finally:
        raw_path.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


def export_meowid_onnx(
    checkpoint: str | Path,
    output: str | Path,
    config: str | Path | None = None,
    opset: int = 18,
    dynamic_batch: bool = True,
    use_onnxslim: bool = True,
) -> Path:
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = output.with_name(f".{output.stem}.raw.onnx")
    raw_output.unlink(missing_ok=True)
    graph, cfg, _ = load_deployment_graph(checkpoint, config)
    graph.eval().cpu()
    # PyTorch 2.4's legacy ONNX symbolic has a known scalar-scale failure for
    # scaled_dot_product_attention.  Hugging Face's mathematically equivalent
    # eager attention exports as primitive MatMul/Softmax operators and is also
    # easier for TensorRT to optimize.
    for expert in (graph.model.body_expert, graph.model.face_expert):
        backbone_model = getattr(getattr(expert, "backbone", None), "model", None)
        if backbone_model is not None and hasattr(backbone_model, "set_attn_implementation"):
            backbone_model.set_attn_implementation("eager")
    image_size = int(
        cfg.data.transforms.train.get(
            "image_size",
            cfg.model.backbone.get("image_size", 256),
        )
    )
    body = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    face = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "body_images": {0: "batch"},
            "face_images": {0: "batch"},
            "body_embeddings": {0: "batch"},
            "face_embeddings": {0: "batch"},
        }
    try:
        torch.onnx.export(
            graph,
            (body, face),
            str(raw_output),
            input_names=["body_images", "face_images"],
            output_names=["body_embeddings", "face_embeddings"],
            dynamic_axes=dynamic_axes,
            opset_version=int(opset),
            do_constant_folding=True,
            export_params=True,
        )
        _finalize_onnx(raw_output, output, use_onnxslim=use_onnxslim)
    finally:
        raw_output.unlink(missing_ok=True)
    return output


def export_ecpose_onnx(
    ecpose_root: str | Path,
    config: str | Path,
    checkpoint: str | Path,
    output: str | Path,
    opset: int = 18,
    dynamic_batch: bool = True,
    use_onnxslim: bool = True,
) -> Path:
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_output = output.with_name(f".{output.stem}.raw.onnx")
    raw_output.unlink(missing_ok=True)
    graph, image_size = load_ecpose_graph(ecpose_root, config, checkpoint)
    graph.eval().cpu()
    images = torch.randn(1, 3, *image_size, dtype=torch.float32)
    original_sizes = torch.tensor([[image_size[1], image_size[0]]], dtype=torch.float32)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {
            "images": {0: "batch"},
            "orig_target_sizes": {0: "batch"},
            "scores": {0: "batch"},
            "labels": {0: "batch"},
            "keypoints": {0: "batch"},
        }
    try:
        torch.onnx.export(
            graph,
            (images, original_sizes),
            str(raw_output),
            input_names=["images", "orig_target_sizes"],
            output_names=["scores", "labels", "keypoints"],
            dynamic_axes=dynamic_axes,
            opset_version=int(opset),
            do_constant_folding=True,
            export_params=True,
        )
        _finalize_onnx(raw_output, output, use_onnxslim=use_onnxslim)
    finally:
        raw_output.unlink(missing_ok=True)
    return output


def build_tensorrt_engine(
    onnx_path: str | Path,
    engine_path: str | Path,
    *,
    min_batch: int = 1,
    opt_batch: int = 4,
    max_batch: int = 16,
    fp16: bool = True,
    workspace_gib: float = 8.0,
) -> Path:
    try:
        import tensorrt as trt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Install TensorRT with `pip install -e .[tensorrt]`") from exc

    onnx_path = Path(onnx_path).resolve()
    engine_path = Path(engine_path).resolve()
    engine_path.parent.mkdir(parents=True, exist_ok=True)
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(str(onnx_path)):
        messages = [str(parser.get_error(index)) for index in range(parser.num_errors)]
        raise RuntimeError("TensorRT ONNX parsing failed:\n" + "\n".join(messages))

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE,
        int(float(workspace_gib) * 1024**3),
    )
    if fp16:
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("This TensorRT platform has no fast FP16 support")
        config.set_flag(trt.BuilderFlag.FP16)
        # Pure FP16 changes retrieval directions more than desired. Keep the
        # high-throughput convolutions/GEMMs in FP16 while evaluating the
        # numerically sensitive normalization, attention softmax and reduction
        # layers in FP32. TensorRT inserts the required casts automatically.
        config.set_flag(trt.BuilderFlag.PREFER_PRECISION_CONSTRAINTS)
        sensitive_types = {
            trt.LayerType.NORMALIZATION,
            trt.LayerType.SOFTMAX,
            trt.LayerType.REDUCE,
        }
        for layer_index in range(network.num_layers):
            layer = network.get_layer(layer_index)
            if layer.type not in sensitive_types:
                continue
            layer.precision = trt.float32
            for output_index in range(layer.num_outputs):
                output_tensor = layer.get_output(output_index)
                if output_tensor.dtype in {trt.float16, trt.float32}:
                    layer.set_output_type(output_index, trt.float32)

    profile = builder.create_optimization_profile()
    for index in range(network.num_inputs):
        input_tensor = network.get_input(index)
        shape = tuple(int(value) for value in input_tensor.shape)
        if not shape or shape[0] != -1:
            continue
        minimum = (int(min_batch), *shape[1:])
        optimum = (int(opt_batch), *shape[1:])
        maximum = (int(max_batch), *shape[1:])
        if any(value < 0 for value in minimum):
            raise ValueError(f"Only a dynamic batch dimension is supported, got {shape} for {input_tensor.name}")
        # TensorRT 10.8's Python binding returns ``None`` on success here,
        # while newer releases document a boolean return. Validation happens
        # when the profile is added and the network is built.
        profile.set_shape(input_tensor.name, minimum, optimum, maximum)
    config.add_optimization_profile(profile)

    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError(f"TensorRT failed to build {onnx_path}")
    temporary = engine_path.with_name(f".{engine_path.name}.tmp")
    temporary.write_bytes(bytes(serialized))
    os.replace(temporary, engine_path)
    return engine_path


def _write_manifest(
    output_dir: Path,
    repo_root: Path,
    source_paths: dict[str, Path],
    artifacts: dict[str, Path],
    build: dict[str, Any],
) -> Path:
    models: dict[str, dict[str, str]] = {
        "torch": {
            "recognition": os.path.relpath(source_paths["recognition"], output_dir),
            "pose": os.path.relpath(source_paths["pose"], output_dir),
        }
    }
    if "recognition_onnx" in artifacts:
        models["onnx"] = {
            "recognition": artifacts["recognition_onnx"].name,
            "pose": artifacts["pose_onnx"].name,
        }
    if "recognition_tensorrt" in artifacts:
        models["tensorrt"] = {
            "recognition": artifacts["recognition_tensorrt"].name,
            "pose": artifacts["pose_tensorrt"].name,
        }
    manifest = {
        "schema_version": 1,
        "name": "MeowID-Base",
        "embedding_dim": 512,
        "created_at_unix": int(time.time()),
        "models": models,
        "alignment": "petface_tight_3pt",
        "face_threshold": 0.4,
        "face_label": 1,
        "preprocessing": {
            "pose": {
                "image_size": [640, 640],
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "body": {
                "image_size": 256,
                "resize_size": 292,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
            "face": {
                "image_size": 256,
                "resize_size": 256,
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225],
            },
        },
        "routing": {
            "face_present": "face_embeddings",
            "face_absent": "body_embeddings",
            "confidence_fusion": False,
            "single_body_encoder": True,
        },
        "artifacts": {
            name: _artifact_info(path, output_dir) for name, path in artifacts.items()
        },
        "build": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "repo_root": str(repo_root),
            **build,
        },
    }
    path = output_dir / "deployment.json"
    temporary = output_dir / ".deployment.json.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)
    return path


def export_deployment(
    format: str = "all",
    *,
    repo_root: str | Path,
    output_dir: str | Path | None = None,
    recognition_checkpoint: str | Path | None = None,
    recognition_config: str | Path | None = None,
    pose_checkpoint: str | Path | None = None,
    pose_config: str | Path | None = None,
    opset: int = 18,
    min_batch: int = 1,
    opt_batch: int = 4,
    max_batch: int = 16,
    fp16: bool = True,
    workspace_gib: float = 8.0,
    use_onnxslim: bool = True,
) -> dict[str, Any]:
    format = format.lower()
    if format == "trt":
        format = "tensorrt"
    if format not in {"onnx", "tensorrt", "all"}:
        raise ValueError(f"Unsupported export format: {format}")
    repo_root = Path(repo_root).resolve()
    output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else repo_root / "artifacts/MeowID-Base"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    recognition_checkpoint = Path(
        recognition_checkpoint
        or repo_root / "artifacts/MeowID-Base/meowid_base.pth"
    ).resolve()
    recognition_config = (
        Path(recognition_config).resolve() if recognition_config is not None else None
    )
    pose_checkpoint = Path(
        pose_checkpoint
        or repo_root / "artifacts/MeowID-Base/ecpose.pth"
    ).resolve()
    pose_config = Path(
        pose_config
        or repo_root
        / "third_party/EdgeCrafter/ecpose/configs/ecpose/ecpose_x_cat_face.yml"
    ).resolve()
    ecpose_root = repo_root / "third_party/EdgeCrafter/ecpose"
    for path in (recognition_checkpoint, pose_checkpoint, pose_config, ecpose_root):
        if not path.exists():
            raise FileNotFoundError(path)

    artifacts: dict[str, Path] = {}
    recognition_onnx = output_dir / "meowid_base.onnx"
    pose_onnx = output_dir / "ecpose.onnx"
    if format in {"onnx", "all"} or not (recognition_onnx.exists() and pose_onnx.exists()):
        export_meowid_onnx(
            recognition_checkpoint,
            recognition_onnx,
            config=recognition_config,
            opset=opset,
            use_onnxslim=use_onnxslim,
        )
        export_ecpose_onnx(
            ecpose_root,
            pose_config,
            pose_checkpoint,
            pose_onnx,
            opset=opset,
            use_onnxslim=use_onnxslim,
        )
    artifacts["recognition_onnx"] = recognition_onnx
    artifacts["pose_onnx"] = pose_onnx

    import onnx
    import onnxruntime as ort

    build: dict[str, Any] = {
        "onnx": onnx.__version__,
        "onnx_opset": int(opset),
        "onnxruntime": ort.__version__,
        "onnxslim": None,
        "onnxslim_enabled": bool(use_onnxslim),
    }
    if use_onnxslim:
        import onnxslim

        build["onnxslim"] = onnxslim.__version__
    if format in {"tensorrt", "all"}:
        recognition_engine = build_tensorrt_engine(
            recognition_onnx,
            output_dir / "meowid_base.engine",
            min_batch=min_batch,
            opt_batch=opt_batch,
            max_batch=max_batch,
            fp16=fp16,
            workspace_gib=workspace_gib,
        )
        pose_engine = build_tensorrt_engine(
            pose_onnx,
            output_dir / "ecpose.engine",
            min_batch=min_batch,
            opt_batch=opt_batch,
            max_batch=max_batch,
            fp16=fp16,
            workspace_gib=workspace_gib,
        )
        artifacts["recognition_tensorrt"] = recognition_engine
        artifacts["pose_tensorrt"] = pose_engine
        import tensorrt as trt

        build.update(
            {
                "tensorrt": trt.__version__,
                "tensorrt_fp16": bool(fp16),
                "tensorrt_sensitive_layers_fp32": bool(fp16),
                "batch_profile": [int(min_batch), int(opt_batch), int(max_batch)],
                "workspace_gib": float(workspace_gib),
            }
        )
        if torch.cuda.is_available():
            build["gpu"] = torch.cuda.get_device_name(torch.cuda.current_device())
            build["cuda_capability"] = list(torch.cuda.get_device_capability())

    manifest_path = _write_manifest(
        output_dir,
        repo_root,
        source_paths={"recognition": recognition_checkpoint, "pose": pose_checkpoint},
        artifacts=artifacts,
        build=build,
    )
    return {
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "artifacts": {name: str(path) for name, path in artifacts.items()},
    }

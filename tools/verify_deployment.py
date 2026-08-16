from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from cat_recognition import MeowID


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    return float(left @ right / max(np.linalg.norm(left) * np.linalg.norm(right), 1.0e-12))


def run_backend(args, backend: str):
    model_arg = None if backend == "torch" else args.artifacts
    model = MeowID(
        model_arg,
        backend=backend,
        device=args.device,
        batch_size=len(args.images),
    )
    for _ in range(args.warmup):
        model.embed(args.images)
    timings = []
    results = None
    for _ in range(args.repeat):
        start = time.perf_counter()
        results = model.embed(args.images)
        timings.append((time.perf_counter() - start) * 1000.0)
    assert results is not None
    info = model.info()
    providers = {}
    if backend == "onnx":
        providers = {
            "recognition": model.recognition_backend.session.session.get_providers(),
            "pose": model.pose_backend.session.session.get_providers(),
        }
    payload = {
        "backend": backend,
        "info": info,
        "providers": providers,
        "latency_ms": {
            "median_batch": statistics.median(timings),
            "median_per_image": statistics.median(timings) / len(args.images),
            "samples": timings,
        },
        "results": results,
    }
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description="Verify Torch, ONNX and TensorRT end to end")
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeat", type=int, default=5)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.artifacts = args.artifacts.resolve()
    output = args.output or args.artifacts / "verification.json"
    runs = {}
    raw = {}
    for backend in ("torch", "onnx", "tensorrt"):
        result = run_backend(args, backend)
        raw[backend] = result.pop("results")
        runs[backend] = result

    comparisons = {}
    reference = raw["torch"]
    passed = True
    for backend in ("onnx", "tensorrt"):
        items = []
        for torch_result, candidate in zip(reference, raw[backend]):
            route_match = torch_result.route == candidate.route
            body_cosine = cosine(torch_result.body_embedding, candidate.body_embedding)
            if torch_result.face_embedding is not None and candidate.face_embedding is not None:
                face_cosine = cosine(torch_result.face_embedding, candidate.face_embedding)
            else:
                face_cosine = None
            if torch_result.face is not None and candidate.face is not None:
                keypoint_max_abs = float(
                    np.max(np.abs(torch_result.face.keypoints - candidate.face.keypoints))
                )
            else:
                keypoint_max_abs = None
            threshold = 0.999 if backend == "onnx" else 0.98
            item_passed = route_match and body_cosine >= threshold and (
                face_cosine is None or face_cosine >= threshold
            )
            passed &= item_passed
            items.append(
                {
                    "source": torch_result.source,
                    "route_match": route_match,
                    "body_cosine": body_cosine,
                    "face_cosine": face_cosine,
                    "keypoint_max_abs_px": keypoint_max_abs,
                    "passed": item_passed,
                }
            )
        comparisons[backend] = items

    report = {
        "passed": bool(passed),
        "images": [str(Path(path).resolve()) for path in args.images],
        "backends": runs,
        "comparisons_to_torch": comparisons,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.strict and not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

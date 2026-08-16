from __future__ import annotations

import argparse
import gc
import json
import os
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from cat_recognition import MeowID
from cat_recognition.deployment.preprocess import IMAGE_EXTENSIONS


@dataclass(frozen=True)
class BackendSpec:
    name: str
    backend: str
    device: str
    half: bool = False


def _image_paths(root: Path) -> list[Path]:
    paths = sorted(
        path.resolve()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No supported images found under {root}")
    return paths


def _warm_file_cache(paths: list[Path]) -> tuple[int, float]:
    total_bytes = 0
    started = time.perf_counter()
    for path in paths:
        with path.open("rb") as handle:
            while chunk := handle.read(8 * 1024 * 1024):
                total_bytes += len(chunk)
    return total_bytes, time.perf_counter() - started


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile))


def _synchronize(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def _providers(model: MeowID) -> dict[str, list[str]]:
    if model.backend != "onnx":
        return {}
    return {
        "recognition": model.recognition_backend.session.session.get_providers(),
        "pose": model.pose_backend.session.session.get_providers(),
    }


def _write_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def _run_backend(
    spec: BackendSpec,
    *,
    artifacts: Path,
    paths: list[Path],
    warmup: int,
    progress_every: int,
) -> dict:
    started = time.perf_counter()
    model = MeowID(
        artifacts,
        backend=spec.backend,
        device=spec.device,
        batch_size=1,
        half=spec.half,
    )
    load_seconds = time.perf_counter() - started
    providers = _providers(model)

    for index in range(warmup):
        model.embed(paths[index % len(paths)])
    _synchronize(spec.device)

    timings: list[float] = []
    routes = {"face": 0, "body": 0}
    run_started = time.perf_counter()
    for index, path in enumerate(paths, start=1):
        _synchronize(spec.device)
        sample_started = time.perf_counter()
        result = model.embed(path)[0]
        _synchronize(spec.device)
        timings.append((time.perf_counter() - sample_started) * 1000.0)
        routes[result.route] += 1
        if progress_every > 0 and (index % progress_every == 0 or index == len(paths)):
            elapsed = time.perf_counter() - run_started
            eta = elapsed / index * (len(paths) - index)
            print(
                f"[{spec.name}] {index}/{len(paths)} "
                f"mean={statistics.fmean(timings):.3f} ms "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )

    total_seconds = time.perf_counter() - run_started
    result = {
        "backend": spec.backend,
        "device": spec.device,
        "precision": "fp16" if spec.half or spec.backend == "tensorrt" else "fp32",
        "batch_size": 1,
        "images": len(timings),
        "warmup_images": warmup,
        "model_load_seconds": load_seconds,
        "total_timed_seconds": total_seconds,
        "latency_ms": {
            "mean": statistics.fmean(timings),
            "median": statistics.median(timings),
            "p90": _percentile(timings, 90),
            "p95": _percentile(timings, 95),
            "p99": _percentile(timings, 99),
            "min": min(timings),
            "max": max(timings),
        },
        "throughput_images_per_second": len(timings) / total_seconds,
        "routes": routes,
        "onnx_providers": providers,
        "timing_scope": (
            "MeowID.embed(path): cached file read/decode, preprocessing, ECPose, "
            "face alignment, body+face embedding, hard routing; excludes model load"
        ),
    }

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MeowID deployment backends")
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=("torch", "onnx_cpu", "onnx_gpu", "tensorrt"),
        default=("torch", "onnx_cpu", "onnx_gpu", "tensorrt"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--torch-half",
        action="store_true",
        help="Use FP16 for Torch instead of the default FP32 baseline",
    )
    parser.add_argument(
        "--no-warm-file-cache",
        action="store_true",
        help="Do not read the dataset once before timing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = args.artifacts.resolve()
    dataset = args.dataset.resolve()
    paths = _image_paths(dataset)
    if args.limit is not None:
        paths = paths[: max(1, int(args.limit))]

    cache = {"enabled": not args.no_warm_file_cache}
    if not args.no_warm_file_cache:
        cache_bytes, cache_seconds = _warm_file_cache(paths)
        cache.update({"bytes": cache_bytes, "seconds": cache_seconds})
        print(
            f"Warmed file cache: {cache_bytes / (1024**2):.1f} MiB in {cache_seconds:.2f}s",
            flush=True,
        )

    gpu_device = str(args.device)
    specs = {
        "torch": BackendSpec("torch", "torch", gpu_device, half=args.torch_half),
        "onnx_cpu": BackendSpec("onnx_cpu", "onnx", "cpu"),
        "onnx_gpu": BackendSpec("onnx_gpu", "onnx", gpu_device),
        "tensorrt": BackendSpec("tensorrt", "tensorrt", gpu_device, half=True),
    }
    report = {
        "artifacts": str(artifacts),
        "dataset": str(dataset),
        "dataset_images": len(paths),
        "file_cache_warmup": cache,
        "backends": {},
    }
    for name in args.backends:
        report["backends"][name] = _run_backend(
            specs[name],
            artifacts=artifacts,
            paths=paths,
            warmup=max(0, int(args.warmup)),
            progress_every=max(0, int(args.progress_every)),
        )
        _write_report(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

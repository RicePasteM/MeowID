#!/usr/bin/env python3
"""Evaluate MeowID hard-route retrieval on a register/query benchmark."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cat_recognition import MeowID


@dataclass(frozen=True)
class Record:
    cat_id: str
    path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path(
            "/data1/hebei/huzhangchi/catdata/"
            "dataset_v240612_final_meowid_vs_detection"
        ),
    )
    parser.add_argument(
        "--model", type=Path, default=Path("artifacts/MeowID-Base")
    )
    parser.add_argument("--backend", choices=("torch", "onnx", "tensorrt"), default="torch")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--aggregation", choices=("max",), default="max")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def load_records(dataset: Path) -> tuple[list[Record], list[Record]]:
    register: list[Record] = []
    query: list[Record] = []
    with (dataset / "manifest.csv").open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            path = (dataset / row["meowid_path"]).resolve()
            record = Record(str(row["identity"]), path)
            if row["role"] == "train_register":
                register.append(record)
            elif row["role"] == "query":
                query.append(record)
    register.sort(key=lambda item: (item.cat_id, str(item.path)))
    query.sort(key=lambda item: (item.cat_id, str(item.path)))
    if not register or not query:
        raise ValueError("Register/query records are empty")
    if set(item.path for item in register) & set(item.path for item in query):
        raise ValueError("Register and query source paths overlap")
    missing = set(item.cat_id for item in query) - set(item.cat_id for item in register)
    if missing:
        raise ValueError(f"Query identities missing from register set: {sorted(missing)[:5]}")
    return register, query


def cache_metadata(args: argparse.Namespace, manifest_hash: str) -> dict:
    return {
        "schema_version": 1,
        "manifest_sha256": manifest_hash,
        "model": str(args.model.resolve()),
        "backend": args.backend,
        "device": args.device,
        "batch_size": args.batch_size,
        "shard_size": args.shard_size,
        "git_commit": git_commit(),
    }


def prepare_cache(path: Path, metadata: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    metadata_path = path / "metadata.json"
    if metadata_path.exists():
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        ignored = {"device", "git_commit"}
        comparable = {key: value for key, value in metadata.items() if key not in ignored}
        existing_comparable = {
            key: value for key, value in existing.items() if key not in ignored
        }
        if existing_comparable != comparable:
            raise ValueError(
                "Embedding cache metadata differs from this run; choose another cache directory"
            )
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def shard_path(cache_dir: Path, name: str, start: int, stop: int) -> Path:
    return cache_dir / name / f"{start:06d}_{stop:06d}.npz"


def valid_cached_shard(path: Path, records: list[Record]) -> bool:
    if not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            ids = data["cat_ids"].astype(str).tolist()
            paths = data["paths"].astype(str).tolist()
            return ids == [item.cat_id for item in records] and paths == [str(item.path) for item in records]
    except Exception:
        return False


def save_shard(path: Path, records: list[Record], results) -> None:
    body = np.stack([item.body_embedding for item in results]).astype(np.float32)
    face_present = np.asarray([item.face_embedding is not None for item in results], dtype=bool)
    face = np.zeros_like(body)
    for index, item in enumerate(results):
        if item.face_embedding is not None:
            face[index] = item.face_embedding
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            cat_ids=np.asarray([item.cat_id for item in records]),
            paths=np.asarray([str(item.path) for item in records]),
            body=body,
            face=face,
            face_present=face_present,
        )
    temporary.replace(path)


def extract_shards(
    model: MeowID,
    records: list[Record],
    name: str,
    cache_dir: Path,
    shard_size: int,
) -> None:
    ranges = [
        (start, min(start + shard_size, len(records)))
        for start in range(0, len(records), shard_size)
    ]
    for start, stop in tqdm(ranges, desc=f"extract {name}"):
        current = records[start:stop]
        destination = shard_path(cache_dir, name, start, stop)
        if valid_cached_shard(destination, current):
            continue
        results = model.embed([item.path for item in current])
        if len(results) != len(current):
            raise RuntimeError("Embedding result count mismatch")
        save_shard(destination, current, results)


def load_shards(
    records: list[Record], name: str, cache_dir: Path, shard_size: int
) -> dict[str, np.ndarray]:
    payload: dict[str, list[np.ndarray]] = {
        "cat_ids": [], "paths": [], "body": [], "face": [], "face_present": []
    }
    for start in range(0, len(records), shard_size):
        stop = min(start + shard_size, len(records))
        path = shard_path(cache_dir, name, start, stop)
        if not valid_cached_shard(path, records[start:stop]):
            raise RuntimeError(f"Missing or invalid cache shard: {path}")
        with np.load(path, allow_pickle=False) as data:
            for key in payload:
                payload[key].append(np.asarray(data[key]))
    return {key: np.concatenate(values, axis=0) for key, values in payload.items()}


def normalize(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1.0e-12)


def resolve_topk(gallery_size: int) -> dict[str, int]:
    return {
        "top1": 1,
        "top5%": max(1, int(math.ceil(gallery_size * 0.05))),
        "top50%": max(1, int(math.ceil(gallery_size * 0.50))),
    }


def route_metrics(
    query_embeddings: np.ndarray,
    query_ids: np.ndarray,
    gallery_embeddings: np.ndarray,
    gallery_ids: np.ndarray,
) -> dict[str, float]:
    unique_ids = np.asarray(sorted(set(gallery_ids.astype(str).tolist())))
    metrics = {
        "num_queries": float(len(query_ids)),
        "num_gallery": float(len(gallery_ids)),
        "num_gallery_cats": float(len(unique_ids)),
    }
    if len(query_ids) == 0 or len(unique_ids) == 0:
        metrics.update({"top1": 0.0, "top5%": 0.0, "top50%": 0.0, "mAP": 0.0})
        return metrics

    similarity = normalize(query_embeddings) @ normalize(gallery_embeddings).T
    grouped = np.empty((len(query_ids), len(unique_ids)), dtype=np.float32)
    for index, cat_id in enumerate(unique_ids):
        grouped[:, index] = similarity[:, gallery_ids.astype(str) == cat_id].max(axis=1)
    order = np.argsort(-grouped, axis=1)
    ranked_ids = unique_ids[order]
    correct = ranked_ids == query_ids.astype(str)[:, None]

    for name, k in resolve_topk(len(unique_ids)).items():
        metrics[name] = float(correct[:, :k].any(axis=1).mean())
    reciprocal_ranks = []
    for row in correct:
        positions = np.flatnonzero(row)
        reciprocal_ranks.append(0.0 if len(positions) == 0 else 1.0 / (int(positions[0]) + 1))
    metrics["mAP"] = float(np.mean(reciprocal_ranks))
    return metrics


def combine_hard_route(face: dict[str, float], body: dict[str, float]) -> dict[str, float]:
    face_count = int(face["num_queries"])
    body_count = int(body["num_queries"])
    total = face_count + body_count
    output = {
        "num_queries": float(total),
        "num_face_queries": float(face_count),
        "num_no_face_queries": float(body_count),
        "num_gallery_cats": float(max(face["num_gallery_cats"], body["num_gallery_cats"])),
        "num_face_gallery_cats": float(face["num_gallery_cats"]),
    }
    for name in ("top1", "top5%", "top50%", "mAP"):
        output[name] = (
            face[name] * face_count + body[name] * body_count
        ) / max(total, 1)
    return output


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else dataset / "results" / f"meowid_base_{args.backend}_embedding_metrics.json"
    )
    cache_dir = (
        args.cache_dir.resolve()
        if args.cache_dir is not None
        else dataset / "results" / f"embedding_cache_{args.backend}"
    )
    register_records, query_records = load_records(dataset)
    manifest_hash = sha256(dataset / "manifest.csv")
    metadata = cache_metadata(args, manifest_hash)
    prepare_cache(cache_dir, metadata)

    missing = False
    for name, records in (("register", register_records), ("query", query_records)):
        for start in range(0, len(records), args.shard_size):
            stop = min(start + args.shard_size, len(records))
            if not valid_cached_shard(
                shard_path(cache_dir, name, start, stop), records[start:stop]
            ):
                missing = True
                break
    extraction_seconds = 0.0
    if missing:
        model = MeowID(
            model=args.model,
            backend=args.backend,
            device=args.device,
            batch_size=args.batch_size,
            half=False,
        )
        started = time.time()
        extract_shards(model, register_records, "register", cache_dir, args.shard_size)
        extract_shards(model, query_records, "query", cache_dir, args.shard_size)
        extraction_seconds = time.time() - started

    register = load_shards(register_records, "register", cache_dir, args.shard_size)
    query = load_shards(query_records, "query", cache_dir, args.shard_size)
    face_query = query["face_present"].astype(bool)
    face_gallery = register["face_present"].astype(bool)
    face = route_metrics(
        query["face"][face_query],
        query["cat_ids"][face_query],
        register["face"][face_gallery],
        register["cat_ids"][face_gallery],
    )
    body = route_metrics(
        query["body"][~face_query],
        query["cat_ids"][~face_query],
        register["body"],
        register["cat_ids"],
    )
    hard_route = combine_hard_route(face, body)
    report = {
        "protocol": {
            "manifest_sha256": manifest_hash,
            "register_images": len(register_records),
            "query_images": len(query_records),
            "identities": len(set(item.cat_id for item in register_records)),
            "group_by": "cat",
            "aggregation": args.aggregation,
            "routing": "face_if_detected_else_body",
            "backend": args.backend,
            "device": args.device,
            "model": str(args.model.resolve()),
            "git_commit": git_commit(),
            "extraction_seconds_this_run": extraction_seconds,
        },
        "face_route": face,
        "body_fallback_route": body,
        "hard_route": hard_route,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temporary.replace(output)
    print(json.dumps(report, indent=2))
    print(f"output={output}")


if __name__ == "__main__":
    main()

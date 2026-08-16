from __future__ import annotations

import json
import math
import os
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .types import EmbeddingResult, SearchMatch


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"Expected a matrix, got shape {values.shape}")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1.0e-12)


def _aggregate(scores: list[float], method: str) -> float:
    if method == "max":
        return max(scores)
    if method == "mean":
        return float(np.mean(scores))
    if method == "logsumexp":
        maximum = max(scores)
        return maximum + math.log(sum(math.exp(value - maximum) for value in scores) / len(scores))
    raise ValueError(f"Unsupported aggregation: {method}")


class EmbeddingRegistry:
    """Persistent two-space gallery for body and face hard routing.

    Exact inner-product search is intentional here: it has no native library
    dependency, is deterministic across backends, and is fast for normal
    per-device registration galleries.  A future approximate index can wrap
    the same public API without changing ``MeowID``.
    """

    VERSION = 1

    def __init__(self, dim: int = 512, path: str | Path | None = None) -> None:
        self.dim = int(dim)
        self.path = None if path is None else Path(path)
        self.embeddings = np.empty((0, self.dim), dtype=np.float32)
        self.cat_ids: list[str] = []
        self.routes: list[str] = []
        self.metadata: list[dict[str, Any]] = []
        self._lock = threading.RLock()
        if self.path is not None and (self.path / "registry.json").exists():
            self.load(self.path)

    def __len__(self) -> int:
        return len(self.cat_ids)

    @property
    def cats(self) -> list[str]:
        return sorted(set(self.cat_ids))

    def counts(self) -> dict[str, int]:
        return {
            "cats": len(self.cats),
            "vectors": len(self),
            "body_vectors": self.routes.count("body"),
            "face_vectors": self.routes.count("face"),
        }

    def add(
        self,
        cat_id: str,
        results: EmbeddingResult | Iterable[EmbeddingResult],
        metadata: dict[str, Any] | None = None,
    ) -> int:
        if isinstance(results, EmbeddingResult):
            items = [results]
        else:
            items = list(results)
        vectors = []
        ids = []
        routes = []
        entries = []
        for result in items:
            base_metadata = {
                "source": result.source,
                "face_detected": result.face_detected,
                **dict(metadata or {}),
            }
            vectors.append(np.asarray(result.body_embedding, dtype=np.float32))
            ids.append(str(cat_id))
            routes.append("body")
            entries.append(dict(base_metadata))
            if result.face_embedding is not None:
                vectors.append(np.asarray(result.face_embedding, dtype=np.float32))
                ids.append(str(cat_id))
                routes.append("face")
                entries.append(dict(base_metadata))
        if not vectors:
            return 0
        matrix = np.stack(vectors)
        if matrix.shape[1] != self.dim:
            raise ValueError(f"Expected embedding dimension {self.dim}, got {matrix.shape[1]}")
        matrix = _normalize_rows(matrix)
        with self._lock:
            self.embeddings = np.concatenate([self.embeddings, matrix], axis=0)
            self.cat_ids.extend(ids)
            self.routes.extend(routes)
            self.metadata.extend(entries)
        return len(vectors)

    def remove(self, cat_id: str) -> int:
        with self._lock:
            keep = np.asarray([value != str(cat_id) for value in self.cat_ids], dtype=bool)
            removed = int((~keep).sum())
            self.embeddings = self.embeddings[keep]
            self.cat_ids = [value for value, selected in zip(self.cat_ids, keep) if selected]
            self.routes = [value for value, selected in zip(self.routes, keep) if selected]
            self.metadata = [value for value, selected in zip(self.metadata, keep) if selected]
            return removed

    def clear(self) -> None:
        with self._lock:
            self.embeddings = np.empty((0, self.dim), dtype=np.float32)
            self.cat_ids.clear()
            self.routes.clear()
            self.metadata.clear()

    def search(
        self,
        embedding: np.ndarray,
        route: str,
        top_k: int = 5,
        aggregation: str = "max",
    ) -> list[SearchMatch]:
        route = str(route).lower()
        if route not in {"body", "face"}:
            raise ValueError(f"Unknown route: {route}")
        with self._lock:
            selected = np.asarray([value == route for value in self.routes], dtype=bool)
            if not selected.any():
                return []
            gallery = self.embeddings[selected]
            ids = [value for value, keep in zip(self.cat_ids, selected) if keep]
            metadata = [value for value, keep in zip(self.metadata, selected) if keep]
        query = _normalize_rows(np.asarray(embedding, dtype=np.float32).reshape(1, -1))[0]
        scores = np.clip(gallery @ query, -1.0, 1.0)
        grouped: dict[str, list[tuple[float, dict[str, Any]]]] = {}
        for score, cat_id, entry in zip(scores.tolist(), ids, metadata):
            grouped.setdefault(cat_id, []).append((float(score), entry))
        matches = []
        for cat_id, items in grouped.items():
            best_score, best_metadata = max(items, key=lambda value: value[0])
            matches.append(
                SearchMatch(
                    cat_id=cat_id,
                    score=float(_aggregate([value[0] for value in items], aggregation)),
                    support=len(items),
                    route=route,
                    metadata={**best_metadata, "best_vector_score": best_score},
                )
            )
        matches.sort(key=lambda value: value.score, reverse=True)
        return matches[: int(top_k)]

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("A registry path is required")
        target.mkdir(parents=True, exist_ok=True)
        arrays_path = target / "embeddings.npz"
        json_path = target / "registry.json"
        arrays_tmp = target / ".embeddings.npz.tmp"
        json_tmp = target / ".registry.json.tmp"
        with self._lock:
            with arrays_tmp.open("wb") as handle:
                np.savez_compressed(handle, embeddings=self.embeddings)
            payload = {
                "version": self.VERSION,
                "dim": self.dim,
                "cat_ids": self.cat_ids,
                "routes": self.routes,
                "metadata": self.metadata,
            }
            with json_tmp.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
            os.replace(arrays_tmp, arrays_path)
            os.replace(json_tmp, json_path)
            self.path = target
        return target

    def load(self, path: str | Path | None = None) -> EmbeddingRegistry:
        target = Path(path) if path is not None else self.path
        if target is None:
            raise ValueError("A registry path is required")
        with (target / "registry.json").open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if int(payload.get("version", -1)) != self.VERSION:
            raise ValueError(f"Unsupported registry version: {payload.get('version')}")
        dim = int(payload["dim"])
        with np.load(target / "embeddings.npz") as arrays:
            embeddings = np.asarray(arrays["embeddings"], dtype=np.float32)
        if embeddings.ndim != 2 or embeddings.shape[1] != dim:
            raise ValueError("Registry embedding matrix is invalid")
        cat_ids = [str(value) for value in payload["cat_ids"]]
        routes = [str(value) for value in payload["routes"]]
        metadata = [dict(value) for value in payload["metadata"]]
        if not (len(embeddings) == len(cat_ids) == len(routes) == len(metadata)):
            raise ValueError("Registry arrays and metadata have different lengths")
        with self._lock:
            self.dim = dim
            self.embeddings = embeddings
            self.cat_ids = cat_ids
            self.routes = routes
            self.metadata = metadata
            self.path = target
        return self

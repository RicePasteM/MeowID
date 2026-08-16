from __future__ import annotations

from contextlib import nullcontext

import numpy as np
import torch

from cat_recognition.utils import all_gather_object, is_main_process, save_retrieval_case_visualizations


def _autocast_context(use_amp: bool):
    if use_amp:
        return torch.cuda.amp.autocast()
    return nullcontext()


def _normalize_np(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1.0e-12, a_max=None)
    return x / norms


def _resolve_topk(topk_raw: list[float], gallery_size: int) -> tuple[list[int], list[str]]:
    effective: list[int] = []
    names: list[str] = []
    for k in topk_raw:
        if k < 1.0:
            eff = max(1, int(np.ceil(gallery_size * k)))
            names.append(f"top{int(round(k * 100))}%")
        else:
            eff = int(k)
            names.append(f"top{int(k)}")
        effective.append(eff)
    return effective, names


def _cfg_to_dict(cfg) -> dict:
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    return dict(cfg)


def _resolve_aggregation_methods(eval_cfg, aggregations: list[str] | None = None) -> list[str]:
    base_cfg = _cfg_to_dict(eval_cfg)
    ordered = [str(base_cfg.get("aggregation", "max"))]
    if aggregations is None:
        aggregations = [str(item) for item in base_cfg.get("compare_aggregations", [])]
    ordered.extend(str(item) for item in aggregations)

    seen: set[str] = set()
    resolved = []
    for item in ordered:
        method = item.lower()
        if method not in seen:
            seen.add(method)
            resolved.append(method)
    return resolved


def _aggregate_group_scores(scores: np.ndarray, method: str) -> np.ndarray:
    if scores.ndim != 2:
        raise ValueError(f"Expected 2D score array, got shape={scores.shape}")

    method = method.lower()
    if method == "max":
        return np.max(scores, axis=1)
    if method == "mean":
        return np.mean(scores, axis=1)
    if method == "logsumexp":
        max_scores = np.max(scores, axis=1)
        aggregated = np.full_like(max_scores, fill_value=-np.inf, dtype=np.float32)
        finite_mask = np.isfinite(max_scores)
        if finite_mask.any():
            stable = np.exp(scores[finite_mask] - max_scores[finite_mask, None])
            aggregated[finite_mask] = max_scores[finite_mask] + np.log(np.mean(stable, axis=1))
        return aggregated
    raise ValueError(f"Unsupported aggregation method: {method}")


def _aggregate_similarity_by_cat(
    similarity: np.ndarray,
    gallery_cat_ids: list[str],
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    grouped_indices: dict[str, list[int]] = {}
    for index, cat_id in enumerate(gallery_cat_ids):
        grouped_indices.setdefault(str(cat_id), []).append(index)

    aggregated_scores = []
    aggregated_cat_ids = []
    for cat_id, indices in grouped_indices.items():
        aggregated_cat_ids.append(cat_id)
        aggregated_scores.append(_aggregate_group_scores(similarity[:, indices], method))

    if not aggregated_scores:
        return np.empty((0,), dtype=object), np.empty((similarity.shape[0], 0), dtype=np.float32)

    stacked = np.stack(aggregated_scores, axis=1).astype(np.float32, copy=False)
    return np.array(aggregated_cat_ids, dtype=object), stacked


def _merge_payloads(payloads: list[dict]) -> dict[str, list | np.ndarray]:
    embeddings = [payload["embeddings"] for payload in payloads if payload["embeddings"].size > 0]
    merged_embeddings = np.concatenate(embeddings, axis=0) if embeddings else np.empty((0, 0), dtype=np.float32)

    merged = {
        "embeddings": merged_embeddings,
        "labels": [],
        "cat_ids": [],
        "paths": [],
    }
    include_face_exists = any("face_exists" in payload for payload in payloads)
    if include_face_exists:
        merged["face_exists"] = []
    for payload in payloads:
        merged["labels"].extend(payload["labels"])
        merged["cat_ids"].extend(payload["cat_ids"])
        merged["paths"].extend(payload["paths"])
        if include_face_exists:
            values = payload.get("face_exists")
            if values is None:
                values = [False] * len(payload["paths"])
            merged["face_exists"].extend(bool(value) for value in values)
    return merged


def extract_embeddings(model, loader, device: torch.device, use_amp: bool = False) -> dict[str, list | np.ndarray]:
    encoder = model.module if hasattr(model, "module") else model
    encoder.eval()

    local_embeddings = []
    local_labels = []
    local_cat_ids = []
    local_paths = []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            with _autocast_context(use_amp and device.type == "cuda"):
                if "face_image" in batch:
                    face_images = batch["face_image"].to(device, non_blocking=True)
                    face_mask = batch["face_exists"].to(device, non_blocking=True).bool()
                    embeddings = encoder.encode(
                        images,
                        face_images=face_images,
                        face_mask=face_mask,
                    )
                else:
                    embeddings = encoder.encode(images)

            local_embeddings.append(embeddings.detach().cpu().float().numpy())
            local_labels.extend([int(item) for item in batch["label"]])
            local_cat_ids.extend(list(batch["cat_id"]))
            local_paths.extend(list(batch["path"]))

    if local_embeddings:
        stacked = np.concatenate(local_embeddings, axis=0)
    else:
        stacked = np.empty((0, encoder.embedding_dim), dtype=np.float32)

    payload = {
        "embeddings": stacked,
        "labels": local_labels,
        "cat_ids": local_cat_ids,
        "paths": local_paths,
    }
    return _merge_payloads(all_gather_object(payload))


def extract_expert_embeddings(
    model,
    loader,
    device: torch.device,
    use_amp: bool = False,
    routes: tuple[str, ...] = ("body", "face"),
) -> dict[str, dict[str, list | np.ndarray]]:
    """Extract independent body and face-route embeddings in one pass.

    Body data contains every whole image plus its face-availability flag. Face
    data contains only samples with an aligned face. The two payloads are kept
    separate because their cosine spaces are not assumed to be compatible.
    """

    requested = {str(route).lower() for route in routes}
    unsupported = requested.difference({"body", "face"})
    if unsupported:
        raise ValueError(f"Unsupported expert routes: {sorted(unsupported)}")

    encoder = model.module if hasattr(model, "module") else model
    if not bool(getattr(encoder, "supports_expert_routes", False)):
        raise TypeError("Model does not expose independent expert routes")
    encoder.eval()

    local: dict[str, dict[str, list]] = {}
    for route in requested:
        local[route] = {
            "embeddings": [],
            "labels": [],
            "cat_ids": [],
            "paths": [],
            "face_exists": [],
        }

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            face_mask = batch["face_exists"].to(device, non_blocking=True).bool()
            with _autocast_context(use_amp and device.type == "cuda"):
                if requested == {"body"}:
                    # Final testing loads the best body and face checkpoints
                    # independently. Avoid running the stale/unneeded face
                    # branch when only the body checkpoint is being evaluated.
                    outputs = {"body_embeddings": encoder.encode_body(images)}
                else:
                    face_images = batch["face_image"].to(device, non_blocking=True)
                    outputs = encoder.encode_routes(
                        images,
                        face_images=face_images,
                        face_mask=face_mask,
                    )

            labels = [int(item) for item in batch["label"]]
            cat_ids = list(batch["cat_id"])
            paths = list(batch["path"])
            face_exists = [bool(item) for item in face_mask.detach().cpu().tolist()]

            if "body" in requested:
                local["body"]["embeddings"].append(
                    outputs["body_embeddings"].detach().cpu().float().numpy()
                )
                local["body"]["labels"].extend(labels)
                local["body"]["cat_ids"].extend(cat_ids)
                local["body"]["paths"].extend(paths)
                local["body"]["face_exists"].extend(face_exists)

            if "face" in requested and any(face_exists):
                selected = face_mask
                local["face"]["embeddings"].append(
                    outputs["face_route_embeddings"][selected]
                    .detach()
                    .cpu()
                    .float()
                    .numpy()
                )
                selected_indices = [index for index, exists in enumerate(face_exists) if exists]
                local["face"]["labels"].extend(labels[index] for index in selected_indices)
                local["face"]["cat_ids"].extend(cat_ids[index] for index in selected_indices)
                local["face"]["paths"].extend(paths[index] for index in selected_indices)
                local["face"]["face_exists"].extend([True] * len(selected_indices))

    payload: dict[str, dict[str, list | np.ndarray]] = {}
    for route in requested:
        chunks = local[route]["embeddings"]
        payload[route] = {
            "embeddings": (
                np.concatenate(chunks, axis=0)
                if chunks
                else np.empty((0, encoder.embedding_dim), dtype=np.float32)
            ),
            "labels": local[route]["labels"],
            "cat_ids": local[route]["cat_ids"],
            "paths": local[route]["paths"],
            "face_exists": local[route]["face_exists"],
        }

    gathered = all_gather_object(payload)
    return {
        route: _merge_payloads([rank_payload[route] for rank_payload in gathered])
        for route in requested
    }


def subset_embedding_data(
    data: dict[str, list | np.ndarray],
    mask: list[bool] | np.ndarray,
) -> dict[str, list | np.ndarray]:
    resolved_mask = np.asarray(mask, dtype=bool)
    indices = np.flatnonzero(resolved_mask)
    embeddings = np.asarray(data["embeddings"])
    subset: dict[str, list | np.ndarray] = {
        "embeddings": embeddings[indices],
        "labels": [data["labels"][index] for index in indices],
        "cat_ids": [data["cat_ids"][index] for index in indices],
        "paths": [data["paths"][index] for index in indices],
    }
    if "face_exists" in data:
        subset["face_exists"] = [data["face_exists"][index] for index in indices]
    return subset


def _mask_same_path(similarity: np.ndarray, query_paths: list[str], gallery_paths: list[str]) -> None:
    gallery_map: dict[str, list[int]] = {}
    for index, path in enumerate(gallery_paths):
        gallery_map.setdefault(path, []).append(index)

    for query_index, path in enumerate(query_paths):
        for gallery_index in gallery_map.get(path, []):
            similarity[query_index, gallery_index] = -np.inf


def _topk_indices(similarity: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    top_k = min(top_k, similarity.shape[1])
    if top_k <= 0:
        raise ValueError("Gallery set is empty.")

    partition = np.argpartition(-similarity, kth=top_k - 1, axis=1)[:, :top_k]
    partition_scores = np.take_along_axis(similarity, partition, axis=1)
    order = np.argsort(-partition_scores, axis=1)
    indices = np.take_along_axis(partition, order, axis=1)
    scores = np.take_along_axis(partition_scores, order, axis=1)
    return indices, scores


def _compute_map(
    metrics: dict[str, float],
    query_cat_ids: np.ndarray,
    top_cat_ids: np.ndarray,
    ranked_similarity: np.ndarray,
) -> None:
    n_queries = len(query_cat_ids)
    if n_queries == 0 or top_cat_ids.shape[1] == 0:
        metrics["mAP"] = 0.0
        return

    aps = []
    gallery_cat_count = top_cat_ids.shape[1]

    for i in range(n_queries):
        query_id = query_cat_ids[i]
        correct = (top_cat_ids[i, :] == query_id).astype(np.float32)
        num_relevant = correct.sum()
        if num_relevant == 0:
            aps.append(0.0)
            continue

        cumsum = np.cumsum(correct)
        positions = np.arange(1, gallery_cat_count + 1, dtype=np.float32)
        precision_at_k = cumsum / positions
        ap = float(np.sum(precision_at_k * correct) / num_relevant)
        aps.append(ap)

    aps = np.array(aps, dtype=np.float32)
    metrics["mAP"] = float(np.mean(aps))


def _build_retrieval_outputs(
    query_data: dict[str, list | np.ndarray],
    gallery_data: dict[str, list | np.ndarray],
    eval_cfg,
) -> dict[str, object]:
    query_embeddings = _normalize_np(query_data["embeddings"])
    gallery_embeddings = _normalize_np(gallery_data["embeddings"])

    query_paths = list(query_data["paths"])
    gallery_paths = list(gallery_data["paths"])
    query_cat_ids = np.array(query_data["cat_ids"], dtype=object)
    gallery_ids = np.array(gallery_data["cat_ids"], dtype=object)
    retrieval_level = str(eval_cfg.get("group_by", "cat")).lower()
    aggregation = str(eval_cfg.get("aggregation", "max")).lower()

    if query_embeddings.size == 0 or gallery_embeddings.size == 0:
        return {
            "metrics": {
                "num_queries": float(len(query_cat_ids)),
                "num_gallery": float(len(gallery_paths)),
                "num_gallery_cats": float(len(set(gallery_data["cat_ids"]))),
            },
            "query_paths": query_paths,
            "query_cat_ids": query_cat_ids,
            "top_cat_ids": np.empty((len(query_paths), 0), dtype=object),
            "top_scores": np.empty((len(query_paths), 0), dtype=np.float32),
            "gallery_paths": gallery_paths,
            "gallery_cat_ids": list(gallery_data["cat_ids"]),
            "similarity": np.empty((len(query_paths), len(gallery_paths)), dtype=np.float32),
            "aggregation": aggregation,
            "group_by": retrieval_level,
        }

    similarity = np.matmul(query_embeddings, gallery_embeddings.T)
    _mask_same_path(similarity, query_paths, gallery_paths)

    topk_raw = [float(item) for item in eval_cfg.get("topk", [1, 5])]

    if retrieval_level == "cat":
        ranked_gallery_ids, ranked_similarity = _aggregate_similarity_by_cat(
            similarity,
            list(gallery_ids),
            method=aggregation,
        )
        topk_effective, topk_names = _resolve_topk(topk_raw, ranked_similarity.shape[1])
        max_k = max(topk_effective)
        top_indices, top_scores = _topk_indices(ranked_similarity, max_k)
        top_cat_ids = ranked_gallery_ids[top_indices]
        num_gallery = float(len(gallery_paths))
        num_gallery_cats = float(len(ranked_gallery_ids))
        full_sorted_indices = np.argsort(-ranked_similarity, axis=1)
        full_sorted_cat_ids = ranked_gallery_ids[full_sorted_indices]
    elif retrieval_level == "image":
        topk_effective, topk_names = _resolve_topk(topk_raw, similarity.shape[1])
        max_k = max(topk_effective)
        top_indices, top_scores = _topk_indices(similarity, max_k)
        top_cat_ids = gallery_ids[top_indices]
        num_gallery = float(len(gallery_ids))
        num_gallery_cats = float(len(set(gallery_data["cat_ids"])))
        full_sorted_indices = np.argsort(-similarity, axis=1)
        full_sorted_cat_ids = gallery_ids[full_sorted_indices]
    else:
        raise ValueError(f"Unsupported evaluation.group_by: {retrieval_level}")

    metrics: dict[str, float] = {
        "num_queries": float(len(query_cat_ids)),
        "num_gallery": num_gallery,
        "num_gallery_cats": num_gallery_cats,
    }

    if len(query_cat_ids) > 0:
        for k, name in zip(topk_effective, topk_names):
            match = (top_cat_ids[:, :k] == query_cat_ids[:, None]).any(axis=1)
            metrics[name] = float(match.mean())

    _compute_map(metrics, query_cat_ids, full_sorted_cat_ids, ranked_similarity)

    return {
        "metrics": metrics,
        "query_paths": query_paths,
        "query_cat_ids": query_cat_ids,
        "top_cat_ids": top_cat_ids,
        "top_scores": top_scores,
        "gallery_paths": gallery_paths,
        "gallery_cat_ids": list(gallery_data["cat_ids"]),
        "similarity": similarity,
        "aggregation": aggregation,
        "group_by": retrieval_level,
    }


def _save_retrieval_visualizations(
    outputs: dict[str, object],
    visualization_dir: str | None = None,
    visualization_cfg=None,
    visualization_metadata: dict | None = None,
) -> None:
    if not visualization_dir or not is_main_process():
        return

    cfg = _cfg_to_dict(visualization_cfg or {})
    if cfg and not bool(cfg.get("enabled", True)):
        return

    top_cat_ids = outputs["top_cat_ids"]
    if getattr(top_cat_ids, "size", 0) == 0:
        return

    metadata = dict(visualization_metadata or {})
    metadata.setdefault("aggregation", outputs["aggregation"])
    metadata.setdefault("group_by", outputs["group_by"])

    save_retrieval_case_visualizations(
        query_paths=outputs["query_paths"],
        query_cat_ids=outputs["query_cat_ids"],
        top_cat_ids=outputs["top_cat_ids"],
        top_scores=outputs["top_scores"],
        gallery_paths=outputs["gallery_paths"],
        gallery_cat_ids=outputs["gallery_cat_ids"],
        similarity=outputs["similarity"],
        output_dir=visualization_dir,
        num_cases=int(cfg.get("num_cases", 5)),
        top_k=int(cfg.get("top_k", 5)),
        image_size=int(cfg.get("image_size", 220)),
        seed=int(cfg.get("seed", 42)),
        metadata=metadata,
    )


def evaluate_retrieval(
    model,
    query_loader,
    gallery_loader,
    device: torch.device,
    eval_cfg,
    use_amp: bool = False,
    visualization_dir: str | None = None,
    visualization_cfg=None,
    visualization_metadata: dict | None = None,
) -> dict[str, float]:
    query_data = extract_embeddings(model, query_loader, device, use_amp=use_amp)
    if gallery_loader is query_loader:
        gallery_data = query_data
    else:
        gallery_data = extract_embeddings(model, gallery_loader, device, use_amp=use_amp)
    outputs = _build_retrieval_outputs(query_data, gallery_data, eval_cfg)
    _save_retrieval_visualizations(
        outputs,
        visualization_dir=visualization_dir,
        visualization_cfg=visualization_cfg,
        visualization_metadata=visualization_metadata,
    )
    return dict(outputs["metrics"])


def compute_retrieval_metrics(
    query_data: dict[str, list | np.ndarray],
    gallery_data: dict[str, list | np.ndarray],
    eval_cfg,
) -> dict[str, float]:
    return dict(_build_retrieval_outputs(query_data, gallery_data, eval_cfg)["metrics"])


def compute_retrieval_outputs_by_aggregation(
    query_data: dict[str, list | np.ndarray],
    gallery_data: dict[str, list | np.ndarray],
    eval_cfg,
    aggregations: list[str] | None = None,
) -> dict[str, dict[str, object]]:
    base_cfg = _cfg_to_dict(eval_cfg)
    results: dict[str, dict[str, object]] = {}
    for method in _resolve_aggregation_methods(base_cfg, aggregations):
        current_cfg = dict(base_cfg)
        current_cfg["aggregation"] = method
        current_cfg["group_by"] = str(base_cfg.get("group_by", "cat"))
        results[method] = _build_retrieval_outputs(query_data, gallery_data, current_cfg)
    return results


def compute_retrieval_metrics_by_aggregation(
    query_data: dict[str, list | np.ndarray],
    gallery_data: dict[str, list | np.ndarray],
    eval_cfg,
    aggregations: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    outputs = compute_retrieval_outputs_by_aggregation(query_data, gallery_data, eval_cfg, aggregations=aggregations)
    return {method: dict(payload["metrics"]) for method, payload in outputs.items()}


def _combine_hard_route_metrics(
    face_metrics: dict[str, float],
    no_face_body_metrics: dict[str, float],
) -> dict[str, float]:
    face_count = int(face_metrics.get("num_queries", 0.0))
    no_face_count = int(no_face_body_metrics.get("num_queries", 0.0))
    total = face_count + no_face_count
    combined: dict[str, float] = {
        "num_queries": float(total),
        "num_face_queries": float(face_count),
        "num_no_face_queries": float(no_face_count),
        # Every test image has a body embedding; report the whole gallery size
        # as the primary hard-route gallery size and the optional face gallery
        # separately.
        "num_gallery": float(no_face_body_metrics.get("num_gallery", 0.0)),
        "num_face_gallery": float(face_metrics.get("num_gallery", 0.0)),
        "num_gallery_cats": float(
            max(
                face_metrics.get("num_gallery_cats", 0.0),
                no_face_body_metrics.get("num_gallery_cats", 0.0),
            )
        ),
    }
    if total <= 0:
        return combined

    metric_names = {
        key
        for key in set(face_metrics).union(no_face_body_metrics)
        if key == "mAP" or (key.startswith("top") and not key.startswith("top_"))
    }
    for key in sorted(metric_names):
        face_value = float(face_metrics.get(key, 0.0))
        no_face_value = float(no_face_body_metrics.get(key, 0.0))
        combined[key] = (face_value * face_count + no_face_value * no_face_count) / total
    return combined


def compute_expert_route_metrics_by_aggregation(
    body_query_data: dict[str, list | np.ndarray],
    body_gallery_data: dict[str, list | np.ndarray],
    face_query_data: dict[str, list | np.ndarray],
    face_gallery_data: dict[str, list | np.ndarray],
    eval_cfg,
    aggregations: list[str] | None = None,
) -> dict[str, dict[str, dict[str, float]]]:
    """Evaluate body, face, and route-aware hard routing independently."""

    face_exists = body_query_data.get("face_exists")
    if face_exists is None:
        raise ValueError("Body query data must include face_exists")
    no_face_query_data = subset_embedding_data(
        body_query_data,
        ~np.asarray(face_exists, dtype=bool),
    )

    result: dict[str, dict[str, dict[str, float]]] = {
        "body": {},
        "face": {},
        "body_no_face": {},
        "hard_route": {},
    }
    base_cfg = _cfg_to_dict(eval_cfg)
    for method in _resolve_aggregation_methods(base_cfg, aggregations):
        current_cfg = dict(base_cfg)
        current_cfg["aggregation"] = method
        body_metrics = dict(
            _build_retrieval_outputs(body_query_data, body_gallery_data, current_cfg)["metrics"]
        )
        face_metrics = dict(
            _build_retrieval_outputs(face_query_data, face_gallery_data, current_cfg)["metrics"]
        )
        no_face_body_metrics = dict(
            _build_retrieval_outputs(no_face_query_data, body_gallery_data, current_cfg)["metrics"]
        )
        result["body"][method] = body_metrics
        result["face"][method] = face_metrics
        result["body_no_face"][method] = no_face_body_metrics
        result["hard_route"][method] = _combine_hard_route_metrics(
            face_metrics,
            no_face_body_metrics,
        )
    return result


def evaluate_expert_route_aggregations(
    model,
    query_loader,
    gallery_loader,
    device: torch.device,
    eval_cfg,
    aggregations: list[str] | None = None,
    use_amp: bool = False,
) -> dict[str, dict[str, dict[str, float]]]:
    query_routes = extract_expert_embeddings(
        model,
        query_loader,
        device,
        use_amp=use_amp,
        routes=("body", "face"),
    )
    if gallery_loader is query_loader:
        gallery_routes = query_routes
    else:
        gallery_routes = extract_expert_embeddings(
            model,
            gallery_loader,
            device,
            use_amp=use_amp,
            routes=("body", "face"),
        )
    return compute_expert_route_metrics_by_aggregation(
        body_query_data=query_routes["body"],
        body_gallery_data=gallery_routes["body"],
        face_query_data=query_routes["face"],
        face_gallery_data=gallery_routes["face"],
        eval_cfg=eval_cfg,
        aggregations=aggregations,
    )


def evaluate_retrieval_aggregations(
    model,
    query_loader,
    gallery_loader,
    device: torch.device,
    eval_cfg,
    aggregations: list[str] | None = None,
    use_amp: bool = False,
    visualization_dir: str | None = None,
    visualization_cfg=None,
    visualization_metadata: dict | None = None,
    visualization_aggregation: str | None = None,
) -> dict[str, dict[str, float]]:
    query_data = extract_embeddings(model, query_loader, device, use_amp=use_amp)
    if gallery_loader is query_loader:
        gallery_data = query_data
    else:
        gallery_data = extract_embeddings(model, gallery_loader, device, use_amp=use_amp)
    outputs = compute_retrieval_outputs_by_aggregation(query_data, gallery_data, eval_cfg, aggregations=aggregations)

    if outputs:
        selected_aggregation = str(
            visualization_aggregation
            or _cfg_to_dict(eval_cfg).get("aggregation", "max")
        ).lower()
        selected_outputs = outputs.get(selected_aggregation)
        if selected_outputs is None:
            first_method = next(iter(outputs))
            selected_outputs = outputs[first_method]
        _save_retrieval_visualizations(
            selected_outputs,
            visualization_dir=visualization_dir,
            visualization_cfg=visualization_cfg,
            visualization_metadata=visualization_metadata,
        )

    return {method: dict(payload["metrics"]) for method, payload in outputs.items()}

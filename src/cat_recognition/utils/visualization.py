from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Sequence

import torch
from PIL import Image, ImageDraw, ImageOps
from torchvision.utils import save_image


def _to_list(value) -> list:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def denormalize_images(
    images: torch.Tensor,
    mean: Sequence[float],
    std: Sequence[float],
) -> torch.Tensor:
    mean_tensor = torch.tensor(mean, dtype=images.dtype).view(1, -1, 1, 1)
    std_tensor = torch.tensor(std, dtype=images.dtype).view(1, -1, 1, 1)
    return (images.detach().cpu().float() * std_tensor + mean_tensor).clamp(0.0, 1.0)


def save_training_batch_preview(
    images: torch.Tensor,
    labels,
    cat_ids,
    paths,
    output_path: str | Path,
    mean: Sequence[float],
    std: Sequence[float],
    max_images: int = 16,
    nrow: int = 4,
    metadata: dict | None = None,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    limit = min(int(max_images), int(images.shape[0]))
    if limit <= 0:
        return

    vis_images = denormalize_images(images[:limit], mean=mean, std=std)
    save_image(vis_images, output_path, nrow=max(1, min(int(nrow), limit)))

    payload = {
        "image_file": output_path.name,
        "num_images": limit,
        "labels": _to_list(labels)[:limit],
        "cat_ids": _to_list(cat_ids)[:limit],
        "paths": _to_list(paths)[:limit],
    }
    if metadata:
        payload.update(metadata)

    with output_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _fit_panel_image(path: str, image_size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return ImageOps.fit(image, (image_size, image_size), method=Image.Resampling.BICUBIC)


def _draw_caption(draw: ImageDraw.ImageDraw, origin_x: int, origin_y: int, lines: list[str]) -> None:
    for index, line in enumerate(lines):
        draw.text((origin_x, origin_y + index * 16), line, fill=(20, 20, 20))


def save_retrieval_case_visualizations(
    query_paths: Sequence[str],
    query_cat_ids: Sequence[str],
    top_cat_ids,
    top_scores,
    gallery_paths: Sequence[str],
    gallery_cat_ids: Sequence[str],
    similarity,
    output_dir: str | Path,
    num_cases: int = 5,
    top_k: int = 3,
    image_size: int = 220,
    seed: int = 42,
    metadata: dict | None = None,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_queries = len(query_paths)
    if total_queries <= 0:
        return

    limit = min(int(num_cases), total_queries)
    rng = random.Random(int(seed))
    indices = list(range(total_queries))
    rng.shuffle(indices)
    selected_indices = sorted(indices[:limit])

    gallery_groups: dict[str, list[int]] = {}
    for gallery_index, cat_id in enumerate(gallery_cat_ids):
        gallery_groups.setdefault(str(cat_id), []).append(gallery_index)

    index_payload = {
        "num_cases": limit,
        "metadata": metadata or {},
        "cases": [],
    }

    panel_width = (top_k + 1) * image_size + (top_k + 2) * 16
    panel_height = image_size + 120

    for case_number, query_index in enumerate(selected_indices, start=1):
        canvas = Image.new("RGB", (panel_width, panel_height), color=(248, 248, 245))
        draw = ImageDraw.Draw(canvas)

        x = 16
        y = 16

        query_image = _fit_panel_image(str(query_paths[query_index]), image_size=image_size)
        canvas.paste(query_image, (x, y))
        _draw_caption(
            draw,
            x,
            y + image_size + 8,
            [
                "QUERY",
                f"gt={query_cat_ids[query_index]}",
            ],
        )

        case_payload = {
            "query_path": str(query_paths[query_index]),
            "query_cat_id": str(query_cat_ids[query_index]),
            "predictions": [],
        }

        for rank in range(min(int(top_k), int(top_cat_ids.shape[1]))):
            cat_id = str(top_cat_ids[query_index, rank])
            aggregated_score = float(top_scores[query_index, rank])
            gallery_indices = gallery_groups.get(cat_id, [])
            representative_index = None
            representative_score = float("-inf")
            representative_path = ""
            if gallery_indices:
                best_local_index = max(gallery_indices, key=lambda index: float(similarity[query_index, index]))
                representative_index = int(best_local_index)
                representative_score = float(similarity[query_index, best_local_index])
                representative_path = str(gallery_paths[best_local_index])

            panel_x = x + (rank + 1) * (image_size + 16)
            if representative_path:
                gallery_image = _fit_panel_image(representative_path, image_size=image_size)
                canvas.paste(gallery_image, (panel_x, y))
            _draw_caption(
                draw,
                panel_x,
                y + image_size + 8,
                [
                    f"TOP{rank + 1}",
                    f"pred={cat_id}",
                    f"agg={aggregated_score:.4f}",
                    f"rep={representative_score:.4f}",
                ],
            )

            case_payload["predictions"].append(
                {
                    "rank": rank + 1,
                    "cat_id": cat_id,
                    "aggregated_score": aggregated_score,
                    "representative_score": representative_score,
                    "representative_path": representative_path,
                    "representative_index": representative_index,
                }
            )

        file_stem = f"case_{case_number:02d}"
        image_path = output_dir / f"{file_stem}.jpg"
        canvas.save(image_path, quality=95)
        with image_path.with_suffix(".json").open("w", encoding="utf-8") as handle:
            json.dump(case_payload, handle, ensure_ascii=False, indent=2)
        index_payload["cases"].append({"image_file": image_path.name, **case_payload})

    with (output_dir / "index.json").open("w", encoding="utf-8") as handle:
        json.dump(index_payload, handle, ensure_ascii=False, indent=2)

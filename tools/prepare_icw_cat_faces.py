"""Build an aligned ICW cat-face dataset from ``catface.csv``.

Every row marked ``detected=1`` is used. Detection confidence is retained as
metadata but never filters, weights, or orders samples. The primary alignment
is a similarity transform fitted to the two eyes and mouth. If those three
points cannot define a stable transform, a square crop around all finite
keypoints is used as a fallback. Warp borders are always black. Alignment
profiles make the target crop reproducible while keeping all other image
processing identical for controlled experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


KEYPOINT_NAMES = [
    "left_eye",
    "right_eye",
    "mouth",
    "left_ear1",
    "left_ear2",
    "left_ear3",
    "right_ear1",
    "right_ear2",
    "right_ear3",
]

ALIGNMENT_PROFILES = {
    "balanced_3pt": {
        "core_keypoints": ["left_eye", "right_eye", "mouth"],
        "canonical_core": [
            [0.36, 0.43],
            [0.64, 0.43],
            [0.50, 0.64],
        ],
        "description": "Original ICW three-point alignment with wider context.",
    },
    "petface_tight_3pt": {
        "core_keypoints": ["left_eye", "right_eye", "mouth"],
        "canonical_core": [
            [56.0 / 224.0, 114.75322978 / 224.0],
            [168.0 / 224.0, 114.58009847 / 224.0],
            [
                ((76.15839386 + 147.32220459) / 2.0) / 224.0,
                ((183.4698995 + 183.47365316) / 2.0) / 224.0,
            ],
        ],
        "description": (
            "Tight three-point approximation of the PetFace cat template. "
            "The mouth-center target is the midpoint of PetFace's two mouth corners."
        ),
    },
}

MANIFEST_FIELDS = [
    "row_index",
    "split",
    "cat_id",
    "image_filename",
    "image_path",
    "source_path",
    "score",
    "alignment_profile",
    "source_width",
    "source_height",
    "alignment_method",
    "fallback_reason",
    "alignment_rmse",
    "valid_fraction",
    "keypoints_inside_fraction",
    "transform_00",
    "transform_01",
    "transform_02",
    "transform_10",
    "transform_11",
    "transform_12",
]

FAILED_FIELDS = [
    "row_index",
    "split",
    "cat_id",
    "image_filename",
    "source_path",
    "score",
    "reason",
]


def estimate_similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("source and target must both have shape (N, 2)")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("core landmarks contain non-finite coordinates")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_variance = float(np.mean(np.sum(source_centered**2, axis=1)))
    if source_variance < 1e-8:
        raise ValueError("degenerate core landmarks")

    covariance = target_centered.T @ source_centered / source.shape[0]
    left, singular_values, right_t = np.linalg.svd(covariance)
    signs = np.ones(2, dtype=np.float64)
    if np.linalg.det(left) * np.linalg.det(right_t) < 0:
        signs[-1] = -1.0
    rotation = left @ np.diag(signs) @ right_t
    scale = float(np.sum(singular_values * signs) / source_variance)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("invalid similarity scale")

    linear = scale * rotation
    translation = target_mean - linear @ source_mean
    return np.column_stack([linear, translation]).astype(np.float64)


def estimate_keypoint_bbox_transform(keypoints: np.ndarray, output_size: int) -> np.ndarray:
    finite = keypoints[np.isfinite(keypoints).all(axis=1)]
    if len(finite) < 2:
        raise ValueError("fewer than two finite keypoints")
    minimum = finite.min(axis=0)
    maximum = finite.max(axis=0)
    center = (minimum + maximum) / 2.0
    side = float(max(maximum - minimum))
    if not math.isfinite(side) or side < 1.0:
        raise ValueError("degenerate keypoint bounding box")
    side *= 1.25
    scale = output_size / side
    top_left = center - side / 2.0
    return np.asarray(
        [[scale, 0.0, -top_left[0] * scale], [0.0, scale, -top_left[1] * scale]],
        dtype=np.float64,
    )


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=points.dtype)])
    return homogeneous @ matrix.T


def parse_keypoints(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [[float(row[f"{name}_x"]), float(row[f"{name}_y"])] for name in KEYPOINT_NAMES],
        dtype=np.float64,
    )


def destination_relative_path(row: dict[str, str]) -> Path:
    return Path(row["split"]) / row["cat_folder"] / f"{Path(row['image_filename']).stem}.jpg"


def align_one(task):
    row, settings = task
    data_root = Path(settings["data_root"])
    output_root = Path(settings["output_root"])
    output_size = int(settings["output_size"])
    alignment_profile = settings["alignment_profile"]
    profile = ALIGNMENT_PROFILES[alignment_profile]
    canonical_core = np.asarray(profile["canonical_core"], dtype=np.float64)
    source_path = data_root / row["relative_path"]
    image_relative = destination_relative_path(row)
    output_path = output_root / image_relative
    result = {
        "row_index": row["row_index"],
        "split": row["split"],
        "cat_id": row["cat_folder"],
        "image_filename": row["image_filename"],
        "image_path": image_relative.as_posix(),
        "source_path": row["relative_path"],
        "score": row["score"],
        "alignment_profile": alignment_profile,
        "accepted": False,
        "reason": "",
    }

    try:
        with Image.open(source_path) as source_image:
            image = source_image.convert("RGB")
        width, height = image.size
        keypoints = parse_keypoints(row)
        fallback_reason = ""
        alignment_method = "similarity"
        alignment_rmse = float("nan")
        try:
            core = keypoints[:3]
            eye_distance = float(np.linalg.norm(core[1] - core[0]))
            if not math.isfinite(eye_distance) or eye_distance < 1.0:
                raise ValueError(f"invalid eye distance: {eye_distance}")
            target_core = canonical_core * output_size
            matrix = estimate_similarity(core, target_core)
            transformed_core = transform_points(core, matrix)
            alignment_rmse = float(
                np.sqrt(np.mean(np.sum((transformed_core - target_core) ** 2, axis=1)))
                / output_size
            )
        except Exception as exc:  # noqa: BLE001
            alignment_method = "keypoint_bbox_fallback"
            fallback_reason = f"{type(exc).__name__}: {exc}"
            matrix = estimate_keypoint_bbox_transform(keypoints, output_size)

        forward = np.vstack([matrix, [0.0, 0.0, 1.0]])
        inverse = np.linalg.inv(forward)[:2]
        inverse_coefficients = tuple(float(value) for value in inverse.reshape(-1))
        aligned = image.transform(
            (output_size, output_size),
            Image.Transform.AFFINE,
            inverse_coefficients,
            resample=Image.Resampling.BICUBIC,
            fillcolor=(0, 0, 0),
        )
        source_mask = Image.new("L", (width, height), color=255)
        aligned_mask = source_mask.transform(
            (output_size, output_size),
            Image.Transform.AFFINE,
            inverse_coefficients,
            resample=Image.Resampling.NEAREST,
            fillcolor=0,
        )
        aligned_mask_array = np.asarray(aligned_mask)
        valid_fraction = float(
            np.count_nonzero(aligned_mask_array) / aligned_mask_array.size
        )
        transformed_keypoints = transform_points(keypoints, matrix)
        inside = (
            np.isfinite(transformed_keypoints).all(axis=1)
            & (transformed_keypoints[:, 0] >= 0)
            & (transformed_keypoints[:, 0] <= output_size)
            & (transformed_keypoints[:, 1] >= 0)
            & (transformed_keypoints[:, 1] <= output_size)
        )
        keypoints_inside_fraction = float(inside.mean())

        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = output_path.with_name(f".{output_path.stem}.tmp.jpg")
        aligned.save(
            temporary_path,
            format="JPEG",
            quality=int(settings["jpeg_quality"]),
        )
        os.replace(temporary_path, output_path)

        result.update(
            {
                "accepted": True,
                "source_width": width,
                "source_height": height,
                "alignment_method": alignment_method,
                "fallback_reason": fallback_reason,
                "alignment_rmse": "" if not math.isfinite(alignment_rmse) else f"{alignment_rmse:.6f}",
                "valid_fraction": f"{valid_fraction:.6f}",
                "keypoints_inside_fraction": f"{keypoints_inside_fraction:.6f}",
                "transform_00": f"{matrix[0, 0]:.9f}",
                "transform_01": f"{matrix[0, 1]:.9f}",
                "transform_02": f"{matrix[0, 2]:.9f}",
                "transform_10": f"{matrix[1, 0]:.9f}",
                "transform_11": f"{matrix[1, 1]:.9f}",
                "transform_12": f"{matrix[1, 2]:.9f}",
            }
        )
    except Exception as exc:  # noqa: BLE001
        result["reason"] = f"{type(exc).__name__}: {exc}"
    return result


def write_csv_atomic(path: Path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, path)


def write_json_atomic(path: Path, payload):
    temporary_path = path.with_name(f".{path.name}.tmp")
    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(temporary_path, path)


def build_summary(
    all_rows,
    detected_rows,
    accepted,
    failed,
    output_size,
    elapsed,
    alignment_profile,
):
    split_summary = {}
    for split in ("train", "val", "test"):
        split_rows = [row for row in accepted if row["split"] == split]
        split_summary[split] = {
            "images": len(split_rows),
            "identities": len({row["cat_id"] for row in split_rows}),
            "similarity": sum(row["alignment_method"] == "similarity" for row in split_rows),
            "fallback": sum(row["alignment_method"] == "keypoint_bbox_fallback" for row in split_rows),
        }
    profile = ALIGNMENT_PROFILES[alignment_profile]
    return {
        "source_annotations": len(all_rows),
        "detected_faces": len(detected_rows),
        "accepted_faces": len(accepted),
        "failed_faces": len(failed),
        "confidence_filtering": False,
        "output_size": [output_size, output_size],
        "border": "black",
        "alignment_profile": alignment_profile,
        "core_keypoints": profile["core_keypoints"],
        "canonical_core_normalized": profile["canonical_core"],
        "profile_description": profile["description"],
        "elapsed_seconds": round(elapsed, 3),
        "alignment_methods": dict(Counter(row["alignment_method"] for row in accepted)),
        "splits": split_summary,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare all detected ICW cat faces")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output-size", type=int, default=256)
    parser.add_argument(
        "--alignment-profile",
        choices=sorted(ALIGNMENT_PROFILES),
        default="balanced_3pt",
        help="Target landmark layout; balanced_3pt reproduces the original behavior",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.output_size <= 0 or args.workers <= 0:
        raise ValueError("output size and workers must be positive")
    if args.data_root.resolve() == args.output_root.resolve():
        raise ValueError("output root must differ from source data root")
    annotations = args.annotations or args.data_root / "catface.csv"
    with annotations.open(newline="", encoding="utf-8-sig") as file:
        all_rows = list(csv.DictReader(file))
    detected_rows = [
        row for row in all_rows if row["detected"] == "1" and not row["error"]
    ]
    if args.limit is not None:
        detected_rows = detected_rows[: args.limit]

    settings = {
        "data_root": str(args.data_root),
        "output_root": str(args.output_root),
        "output_size": args.output_size,
        "alignment_profile": args.alignment_profile,
        "jpeg_quality": args.jpeg_quality,
    }
    started = time.monotonic()
    results = []
    tasks = ((row, settings) for row in detected_rows)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, result in enumerate(executor.map(align_one, tasks, chunksize=32), start=1):
            results.append(result)
            if index % 2000 == 0 or index == len(detected_rows):
                elapsed = time.monotonic() - started
                print(
                    f"processed={index}/{len(detected_rows)} ({index / len(detected_rows):.1%}) "
                    f"rate={index / max(elapsed, 1e-6):.1f} images/s",
                    flush=True,
                )

    accepted = sorted(
        (row for row in results if row["accepted"]),
        key=lambda row: int(row["row_index"]),
    )
    failed = sorted(
        (row for row in results if not row["accepted"]),
        key=lambda row: int(row["row_index"]),
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    write_csv_atomic(args.output_root / "manifest.csv", accepted, MANIFEST_FIELDS)
    for split in ("train", "val", "test"):
        split_rows = [row for row in accepted if row["split"] == split]
        write_csv_atomic(args.output_root / f"{split}.csv", split_rows, MANIFEST_FIELDS)
    write_csv_atomic(args.output_root / "failed.csv", failed, FAILED_FIELDS)
    summary = build_summary(
        all_rows,
        detected_rows,
        accepted,
        failed,
        args.output_size,
        time.monotonic() - started,
        args.alignment_profile,
    )
    write_json_atomic(args.output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

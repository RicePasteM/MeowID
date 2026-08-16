"""Align ICW cat faces for identity-recognition training.

The alignment uses a least-squares similarity transform fitted to the two eyes
and mouth. Similarity alignment removes translation, in-plane rotation and
scale without shearing the face. The remaining ear landmarks are used for
coverage diagnostics and are recorded in the output manifest.

Example (preview 20 randomly selected faces)::

    python tools/cat_face/align_icw_faces.py \
        --data-root /data1/hebei/huzhangchi/catdata/icw_split \
        --output-root /data1/hebei/huzhangchi/catdata/icw_catface_aligned_preview \
        --min-score 0.8 --min-valid-fraction 0.9 --sample-count 20 \
        --gallery /data1/hebei/huzhangchi/catdata/icw_catface_alignment_20.jpg

Omit ``--sample-count`` to build the complete aligned dataset.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np


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

# A deliberately loose crop that normally retains both ears and the lower jaw.
# Coordinates are normalized and multiplied by the requested output size.
CANONICAL_CORE = np.asarray(
    [
        [0.36, 0.43],  # left eye
        [0.64, 0.43],  # right eye
        [0.50, 0.64],  # mouth
    ],
    dtype=np.float64,
)

MANIFEST_FIELDS = [
    "row_index",
    "split",
    "cat_id",
    "image_filename",
    "source_path",
    "aligned_path",
    "score",
    "source_width",
    "source_height",
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


def estimate_similarity(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return a 2x3 similarity transform mapping source points to target."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("source and target must both have shape (N, 2)")

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    source_variance = np.mean(np.sum(source_centered**2, axis=1))
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


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=points.dtype)])
    return homogeneous @ matrix.T


def parse_keypoints(row: dict[str, str]) -> np.ndarray:
    return np.asarray(
        [[float(row[f"{name}_x"]), float(row[f"{name}_y"])] for name in KEYPOINT_NAMES],
        dtype=np.float64,
    )


def destination_relative_path(row: dict[str, str]) -> Path:
    # ICW currently uses unique numeric JPEG names within each identity folder.
    # Normalizing the output extension makes the recognition dataset uniform.
    return Path(row["split"]) / row["cat_folder"] / f"{Path(row['image_filename']).stem}.jpg"


def align_one(task):
    row, settings = task
    data_root = Path(settings["data_root"])
    output_root = Path(settings["output_root"])
    output_size = settings["output_size"]
    source_path = data_root / row["relative_path"]
    aligned_relative = destination_relative_path(row)
    aligned_path = output_root / aligned_relative

    result = {
        "row_index": row["row_index"],
        "split": row["split"],
        "cat_id": row["cat_folder"],
        "image_filename": row["image_filename"],
        "source_path": row["relative_path"],
        "aligned_path": aligned_relative.as_posix(),
        "score": row["score"],
        "accepted": False,
        "reason": "",
    }
    try:
        image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("cv2.imread returned None")
        height, width = image.shape[:2]
        keypoints = parse_keypoints(row)
        core = keypoints[:3]
        eye_distance = float(np.linalg.norm(core[1] - core[0]))
        if eye_distance < settings["min_eye_distance"]:
            raise ValueError(f"eye distance too small: {eye_distance:.3f}")

        target_core = CANONICAL_CORE * output_size
        matrix = estimate_similarity(core, target_core)
        transformed_core = transform_points(core, matrix)
        alignment_rmse = float(
            np.sqrt(np.mean(np.sum((transformed_core - target_core) ** 2, axis=1)))
            / output_size
        )
        if alignment_rmse > settings["max_alignment_rmse"]:
            raise ValueError(f"alignment RMSE too high: {alignment_rmse:.5f}")

        aligned = cv2.warpAffine(
            image,
            matrix,
            (output_size, output_size),
            flags=cv2.INTER_CUBIC,
            # Black is preferable to reflection here: many ICW images already
            # use a black cutout background, while reflection can duplicate an
            # ear or eye when a rotated crop crosses the source boundary.
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        source_mask = np.full((height, width), 255, dtype=np.uint8)
        aligned_mask = cv2.warpAffine(
            source_mask,
            matrix,
            (output_size, output_size),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid_fraction = float(np.count_nonzero(aligned_mask) / aligned_mask.size)
        if valid_fraction < settings["min_valid_fraction"]:
            raise ValueError(f"valid source coverage too low: {valid_fraction:.5f}")

        transformed_keypoints = transform_points(keypoints, matrix)
        inside = (
            (transformed_keypoints[:, 0] >= 0)
            & (transformed_keypoints[:, 0] <= output_size)
            & (transformed_keypoints[:, 1] >= 0)
            & (transformed_keypoints[:, 1] <= output_size)
        )
        keypoints_inside_fraction = float(inside.mean())

        aligned_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = aligned_path.with_name(f".{aligned_path.stem}.tmp.jpg")
        if not cv2.imwrite(
            str(temporary_path),
            aligned,
            [cv2.IMWRITE_JPEG_QUALITY, settings["jpeg_quality"]],
        ):
            raise OSError(f"failed to write {temporary_path}")
        os.replace(temporary_path, aligned_path)

        result.update(
            {
                "accepted": True,
                "source_width": width,
                "source_height": height,
                "alignment_rmse": f"{alignment_rmse:.6f}",
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


def create_gallery(results, output_root: Path, gallery_path: Path, output_size: int):
    if not results:
        raise ValueError("cannot create a gallery without accepted faces")
    columns = 5
    label_height = 28
    rows = math.ceil(len(results) / columns)
    canvas = np.full(
        (rows * (output_size + label_height), columns * output_size, 3),
        245,
        dtype=np.uint8,
    )
    for index, result in enumerate(results):
        image = cv2.imread(str(output_root / result["aligned_path"]), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(output_root / result["aligned_path"])
        if image.shape[:2] != (output_size, output_size):
            image = cv2.resize(image, (output_size, output_size), interpolation=cv2.INTER_AREA)
        row, column = divmod(index, columns)
        top = row * (output_size + label_height)
        left = column * output_size
        canvas[top : top + output_size, left : left + output_size] = image
        label = f"ID {result['cat_id']}  score {float(result['score']):.3f}"
        cv2.putText(
            canvas,
            label,
            (left + 5, top + output_size + 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )
    gallery_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = gallery_path.with_name(f".{gallery_path.stem}.tmp.jpg")
    if not cv2.imwrite(str(temporary_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 94]):
        raise OSError(f"failed to write {temporary_path}")
    os.replace(temporary_path, gallery_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Align ICW cat faces for recognition")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--rejected", type=Path, default=None)
    parser.add_argument("--gallery", type=Path, default=None)
    parser.add_argument("--output-size", type=int, default=224)
    parser.add_argument("--min-score", type=float, default=0.8)
    parser.add_argument("--min-eye-distance", type=float, default=8.0)
    parser.add_argument("--max-alignment-rmse", type=float, default=0.12)
    parser.add_argument("--min-valid-fraction", type=float, default=0.65)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--sample-count",
        type=int,
        default=None,
        help="Randomly produce this many accepted faces; omit for the full dataset",
    )
    parser.add_argument("--seed", type=int, default=20260805)
    return parser.parse_args()


def main():
    args = parse_args()
    annotations = args.annotations or args.data_root / "catface.csv"
    manifest = args.manifest or args.output_root / "manifest.csv"
    rejected_path = args.rejected or args.output_root / "rejected.csv"

    with annotations.open(newline="", encoding="utf-8-sig") as file:
        all_rows = list(csv.DictReader(file))
    eligible = [
        row
        for row in all_rows
        if row["detected"] == "1"
        and not row["error"]
        and float(row["score"]) >= args.min_score
    ]
    eligible_total = len(eligible)
    if args.sample_count is not None:
        if args.sample_count <= 0:
            raise ValueError("--sample-count must be positive")
        rng = random.Random(args.seed)
        rng.shuffle(eligible)

    settings = {
        "data_root": str(args.data_root),
        "output_root": str(args.output_root),
        "output_size": args.output_size,
        "min_eye_distance": args.min_eye_distance,
        "max_alignment_rmse": args.max_alignment_rmse,
        "min_valid_fraction": args.min_valid_fraction,
        "jpeg_quality": args.jpeg_quality,
    }
    if args.sample_count is not None:
        # Quality filters can reject a sampled image. Continue drawing from the
        # shuffled pool so --sample-count denotes accepted aligned faces.
        results = []
        accepted_count = 0
        for row in eligible:
            result = align_one((row, settings))
            results.append(result)
            accepted_count += int(result["accepted"])
            if accepted_count >= args.sample_count:
                break
    elif args.workers > 1:
        tasks = ((row, settings) for row in eligible)
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            results = list(executor.map(align_one, tasks, chunksize=32))
    else:
        results = [align_one((row, settings)) for row in eligible]

    accepted = [result for result in results if result["accepted"]]
    rejected = [result for result in results if not result["accepted"]]
    accepted.sort(key=lambda row: int(row["row_index"]))
    rejected.sort(key=lambda row: int(row["row_index"]))
    write_csv_atomic(manifest, accepted, MANIFEST_FIELDS)
    write_csv_atomic(
        rejected_path,
        rejected,
        [
            "row_index",
            "split",
            "cat_id",
            "image_filename",
            "source_path",
            "score",
            "reason",
        ],
    )
    if args.gallery is not None:
        create_gallery(accepted, args.output_root, args.gallery, args.output_size)

    print(
        f"annotations={len(all_rows)}, eligible={eligible_total}, attempted={len(results)}, "
        f"accepted={len(accepted)}, rejected={len(rejected)}"
    )
    print(f"manifest={manifest}")
    if args.gallery is not None:
        print(f"gallery={args.gallery}")


if __name__ == "__main__":
    main()

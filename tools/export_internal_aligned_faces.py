#!/usr/bin/env python3
"""Export MeowID PetFace-tight aligned faces for the internal benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from tqdm import tqdm

from cat_recognition import MeowID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--benchmark",
        type=Path,
        default=Path(
            "/data1/hebei/huzhangchi/catdata/"
            "dataset_v240612_final_meowid_vs_detection"
        ),
    )
    parser.add_argument(
        "--model", type=Path, default=Path("artifacts/MeowID-Base")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shard-size", type=int, default=64)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def write_csv_atomic(path: Path, rows: list[dict]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "role",
                "identity",
                "source_path",
                "aligned_path",
                "face_present",
            ),
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def main() -> None:
    args = parse_args()
    benchmark = args.benchmark.resolve()
    output = (
        args.output.resolve()
        if args.output is not None
        else benchmark / "results/aligned_faces_meowid_tight"
    )
    output.mkdir(parents=True, exist_ok=True)

    with (benchmark / "manifest.csv").open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    source_rows.sort(
        key=lambda row: (
            0 if row["role"] == "train_register" else 1,
            row["identity"],
            row["meowid_path"],
        )
    )
    model = MeowID(
        model=args.model.resolve(),
        backend="torch",
        device=args.device,
        batch_size=args.batch_size,
        half=False,
    )

    records: list[dict] = []
    for start in tqdm(
        range(0, len(source_rows), args.shard_size), desc="align internal faces"
    ):
        chunk = source_rows[start : start + args.shard_size]
        paths = [(benchmark / row["meowid_path"]).resolve() for row in chunk]
        aligned = model.align(paths)
        if len(aligned) != len(chunk):
            raise RuntimeError("Alignment result count mismatch")
        for row, source_path, face in zip(chunk, paths, aligned):
            relative = ""
            if face is not None:
                role_dir = "register" if row["role"] == "train_register" else "query"
                suffix = f"{source_path.stem}.jpg"
                destination = output / role_dir / row["identity"] / suffix
                destination.parent.mkdir(parents=True, exist_ok=True)
                temporary = destination.with_name(f".{destination.stem}.tmp.jpg")
                face.save(
                    temporary,
                    format="JPEG",
                    quality=args.jpeg_quality,
                    subsampling=0,
                )
                os.replace(temporary, destination)
                relative = destination.relative_to(output).as_posix()
            records.append(
                {
                    "role": row["role"],
                    "identity": row["identity"],
                    "source_path": str(source_path),
                    "aligned_path": relative,
                    "face_present": int(face is not None),
                }
            )

    write_csv_atomic(output / "manifest.csv", records)
    summary = {
        "benchmark": str(benchmark),
        "model": str(args.model.resolve()),
        "alignment": model.alignment,
        "images": len(records),
        "register_images": sum(row["role"] == "train_register" for row in records),
        "query_images": sum(row["role"] == "query" for row in records),
        "register_faces": sum(
            row["role"] == "train_register" and row["face_present"] == 1
            for row in records
        ),
        "query_faces": sum(
            row["role"] == "query" and row["face_present"] == 1
            for row in records
        ),
        "border": "black",
    }
    temporary = output / ".summary.json.tmp"
    temporary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    os.replace(temporary, output / "summary.json")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

"""Run batched ECPose cat-face inference over an ICW split dataset.

Launch with torchrun to shard metadata rows across GPUs. Every source image is
represented by exactly one output row. The highest-scoring face query is kept;
keypoint fields are blank when its score is below the configured threshold.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from engine.core import YAMLConfig  # noqa: E402


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


class ICWDataset(Dataset):
    def __init__(self, records, indices, image_size):
        self.records = records
        self.indices = indices
        self.image_size = tuple(image_size)
        self.transform = T.Compose(
            [
                T.Resize(self.image_size),
                T.ToTensor(),
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        row_index = self.indices[item]
        record = self.records[row_index]
        error = ""
        width = height = 0
        try:
            with Image.open(record["absolute_path"]) as image:
                image = image.convert("RGB")
                width, height = image.size
                tensor = self.transform(image)
        except Exception as exc:  # noqa: BLE001
            tensor = torch.zeros((3, *self.image_size), dtype=torch.float32)
            error = f"{type(exc).__name__}: {exc}"
        return tensor, row_index, torch.tensor([width, height]), error


def build_records(data_root: Path):
    folder_to_split = {}
    for split in ("train", "val", "test"):
        split_root = data_root / split
        for folder in split_root.iterdir():
            if not folder.is_dir():
                continue
            if folder.name in folder_to_split:
                raise RuntimeError(f"Cat folder occurs in multiple splits: {folder.name}")
            folder_to_split[folder.name] = split

    metadata_path = data_root / "metadata.csv"
    records = []
    with metadata_path.open(newline="", encoding="utf-8-sig") as file:
        for row_index, row in enumerate(csv.DictReader(file)):
            cat_folder = row["cat_folder"].strip('"\ufeff')
            image_filename = row["image_filename"]
            split = folder_to_split.get(cat_folder)
            if split is None:
                raise FileNotFoundError(f"No split found for cat folder {cat_folder}")
            relative_path = Path(split) / cat_folder / image_filename
            absolute_path = data_root / relative_path
            if not absolute_path.is_file():
                raise FileNotFoundError(absolute_path)
            records.append(
                {
                    "row_index": row_index,
                    "split": split,
                    "cat_folder": cat_folder,
                    "image_filename": image_filename,
                    "relative_path": relative_path.as_posix(),
                    "absolute_path": str(absolute_path),
                }
            )
    return records


def build_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    cfg = YAMLConfig(str(config_path))
    if "ViTAdapter" in cfg.yaml_cfg:
        cfg.yaml_cfg["ViTAdapter"]["skip_load_backbone"] = True

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["ema"]["module"] if "ema" in checkpoint else checkpoint["model"]
    cfg.model.load_state_dict(state)

    model = cfg.model.deploy().to(device).eval()
    postprocessor = cfg.postprocessor.deploy().to(device).eval()
    return model, postprocessor, tuple(cfg.yaml_cfg["eval_spatial_size"])


def output_fields():
    fields = [
        "row_index",
        "split",
        "cat_folder",
        "image_filename",
        "relative_path",
        "image_width",
        "image_height",
        "detected",
        "score",
        "label",
    ]
    for name in KEYPOINT_NAMES:
        fields.extend([f"{name}_x", f"{name}_y"])
    fields.append("error")
    return fields


def write_prediction(writer, record, width, height, score, label, keypoints, threshold, error):
    detected = not error and score >= threshold
    output = {
        "row_index": record["row_index"],
        "split": record["split"],
        "cat_folder": record["cat_folder"],
        "image_filename": record["image_filename"],
        "relative_path": record["relative_path"],
        "image_width": width,
        "image_height": height,
        "detected": int(detected),
        "score": f"{score:.6f}",
        "label": label,
        "error": error,
    }
    for keypoint_index, name in enumerate(KEYPOINT_NAMES):
        if detected:
            # The regression head can extrapolate slightly beyond an image,
            # especially for ear contours near a crop boundary. Keep exported
            # coordinates in the valid continuous image-coordinate range.
            x = min(max(float(keypoints[keypoint_index, 0]), 0.0), float(width))
            y = min(max(float(keypoints[keypoint_index, 1]), 0.0), float(height))
            output[f"{name}_x"] = f"{x:.3f}"
            output[f"{name}_y"] = f"{y:.3f}"
        else:
            output[f"{name}_x"] = ""
            output[f"{name}_y"] = ""
    writer.writerow(output)
    return detected


def merge_parts(output_path: Path, world_size: int, expected_rows: int):
    rows = []
    for rank in range(world_size):
        part_path = output_path.with_name(f"{output_path.name}.part{rank}")
        with part_path.open(newline="", encoding="utf-8") as file:
            rows.extend(csv.DictReader(file))

    rows.sort(key=lambda row: int(row["row_index"]))
    if len(rows) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} rows, got {len(rows)}")
    if any(int(row["row_index"]) != index for index, row in enumerate(rows)):
        raise RuntimeError("Merged output has missing or duplicate row indices")

    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    with temporary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=output_fields())
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary_path, output_path)

    for rank in range(world_size):
        output_path.with_name(f"{output_path.name}.part{rank}").unlink()


def parse_args():
    parser = argparse.ArgumentParser(description="Predict ICW cat-face keypoints")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size per GPU")
    parser.add_argument("--num-workers", type=int, default=4, help="Workers per GPU")
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test row limit")
    return parser.parse_args()


def main():
    args = parse_args()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        dist.init_process_group(backend="nccl")

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    records = build_records(args.data_root)
    if args.limit is not None:
        records = records[: args.limit]
    indices = list(range(rank, len(records), world_size))
    model, postprocessor, image_size = build_model(args.config, args.checkpoint, device)
    dataset = ICWDataset(records, indices, image_size)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    part_path = args.output.with_name(f"{args.output.name}.part{rank}")
    detected_count = error_count = processed_count = 0
    started = time.monotonic()
    with part_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=output_fields())
        writer.writeheader()
        for batch_index, (images, row_indices, original_sizes, errors) in enumerate(loader):
            images = images.to(device, non_blocking=True)
            original_sizes_gpu = original_sizes.to(device, non_blocking=True)
            with torch.inference_mode(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(images)
                scores, labels, keypoints = postprocessor(outputs, original_sizes_gpu)

            scores = scores[:, 0].float().cpu()
            labels = labels[:, 0].cpu()
            keypoints = keypoints[:, 0].float().cpu()
            for item_index, row_index_tensor in enumerate(row_indices):
                row_index = int(row_index_tensor)
                record = records[row_index]
                width, height = (int(value) for value in original_sizes[item_index])
                error = errors[item_index]
                detected_count += write_prediction(
                    writer,
                    record,
                    width,
                    height,
                    float(scores[item_index]),
                    int(labels[item_index]),
                    keypoints[item_index],
                    args.threshold,
                    error,
                )
                error_count += int(bool(error))
                processed_count += 1

            if rank == 0 and (batch_index + 1) % 100 == 0:
                elapsed = time.monotonic() - started
                rate = processed_count / elapsed
                print(
                    f"rank0 {processed_count}/{len(indices)} "
                    f"({processed_count / len(indices):.1%}), {rate:.1f} images/s",
                    flush=True,
                )

    stats = torch.tensor([processed_count, detected_count, error_count], device=device)
    if distributed:
        dist.all_reduce(stats, op=dist.ReduceOp.SUM)
        dist.barrier()
    if rank == 0:
        merge_parts(args.output, world_size, len(records))
        total, detected, errors = (int(value) for value in stats.cpu())
        print(
            f"Wrote {args.output}: rows={total}, detected={detected}, "
            f"not_detected={total - detected}, errors={errors}",
            flush=True,
        )
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

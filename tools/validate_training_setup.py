from __future__ import annotations

import argparse
import hashlib
import logging
import os
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from common import PROJECT_ROOT

from cat_recognition.config import apply_overrides, load_config
from cat_recognition.data import build_paired_face_hint_split_dataset
from cat_recognition.engine import MeowIDBaseTrainer
from cat_recognition.models import build_model
from cat_recognition.optim import build_optimizer, build_scheduler


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_training_weights(root: Path) -> None:
    checksum_path = root / "SHA256SUMS"
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        path = root / name.strip()
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != expected:
            raise RuntimeError(f"Checksum mismatch for {path}: {actual} != {expected}")
        print(f"[OK] {path.relative_to(PROJECT_ROOT)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the MeowID-Base training setup")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "configs/experiments/meowid_base.yaml",
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--face-root", type=Path)
    parser.add_argument("--build-model", action="store_true")
    parser.add_argument(
        "--train-step",
        action="store_true",
        help="Run one real optimizer step on two samples; implies --build-model",
    )
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    os.chdir(PROJECT_ROOT)
    overrides = []
    if args.data_root is not None:
        overrides.append(f"data.root={args.data_root.expanduser().resolve()}")
    if args.face_root is not None:
        overrides.append(f"data.face_root={args.face_root.expanduser().resolve()}")
    cfg = apply_overrides(load_config(args.config), overrides)

    verify_training_weights(PROJECT_ROOT / "artifacts/training_init")
    dataset = build_paired_face_hint_split_dataset(
        data_cfg=cfg.data,
        split_name="train",
        is_train=False,
        root_dir=PROJECT_ROOT,
        build_class_to_idx=True,
    )
    if len(dataset) == 0:
        raise RuntimeError("Training dataset is empty")
    sample = dataset[0]
    required = {"image", "face_image", "face_exists", "label", "cat_id"}
    missing = required - set(sample)
    if missing:
        raise RuntimeError(f"Training sample is missing keys: {sorted(missing)}")
    print(
        "[OK] dataset",
        {
            "samples": len(dataset),
            "classes": dataset.num_classes,
            "face_samples": dataset.face_count,
            "first_cat_id": sample["cat_id"],
            "image_shape": tuple(sample["image"].shape),
            "face_shape": tuple(sample["face_image"].shape),
        },
    )

    if not args.build_model and not args.train_step:
        print("Training setup validation passed. Add --build-model for a full forward check.")
        return

    device = torch.device(args.device)
    model = build_model(
        cfg,
        num_classes=dataset.num_classes,
        with_head=True,
        class_to_idx=dataset.class_to_idx,
    ).to(device).eval()
    image = sample["image"].unsqueeze(0).to(device)
    face_image = sample["face_image"].unsqueeze(0).to(device)
    face_mask = torch.as_tensor([bool(sample["face_exists"])], device=device)
    label = torch.as_tensor([int(sample["label"])], device=device)
    with torch.inference_mode():
        outputs = model(
            image,
            label,
            face_images=face_image,
            face_mask=face_mask,
        )
    body = outputs["body_embeddings"]
    if body.shape != (1, int(cfg.model.embedding_dim)):
        raise RuntimeError(f"Unexpected body embedding shape: {tuple(body.shape)}")
    print(
        "[OK] full model forward",
        {
            "device": str(device),
            "body_embedding": tuple(body.shape),
            "face_exists": bool(face_mask.item()),
            "initialization": getattr(model, "initialization_info", {}),
        },
    )

    if not args.train_step:
        return

    train_loader = DataLoader(
        Subset(dataset, range(min(2, len(dataset)))),
        batch_size=min(2, len(dataset)),
        shuffle=False,
        num_workers=0,
    )
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    logger = logging.getLogger("meowid-training-smoke")
    logger.addHandler(logging.NullHandler())
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with tempfile.TemporaryDirectory(prefix="meowid_training_smoke_") as output_dir:
        trainer = MeowIDBaseTrainer(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_query_loader=None,
            val_gallery_loader=None,
            device=device,
            cfg=cfg,
            output_dir=output_dir,
            logger=logger,
            class_to_idx=dataset.class_to_idx,
        )
        metrics = trainer.train_one_epoch(0)
    peak_gib = (
        torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None
    )
    print(
        "[OK] optimizer step",
        {
            "loss": round(float(metrics["loss"]), 6),
            "body_cls_acc": round(float(metrics.get("body_cls_acc", 0.0)), 6),
            "face_cls_acc": round(float(metrics.get("face_cls_acc", 0.0)), 6),
            "peak_cuda_gib": None if peak_gib is None else round(peak_gib, 3),
        },
    )


if __name__ == "__main__":
    main()

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

from cat_recognition.config import apply_overrides, dump_config, load_config
from cat_recognition.models import build_model

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("train_faces", [True, False], ids=["mixed-faces", "body-only"])
def test_training_validation_and_final_test(
    tmp_path: Path, monkeypatch, train_faces: bool
) -> None:
    """Exercise the real training entry point with a small, offline DINOv3."""
    body_root = tmp_path / "body"
    face_root = tmp_path / "faces"
    output_dir = tmp_path / "outputs"
    mapping = {"cat_a": 0, "cat_b": 1}
    for split in ("train", "val", "test"):
        (face_root / split).mkdir(parents=True)
        for cat_id, label in mapping.items():
            for index in range(2):
                relative = Path(split) / cat_id / f"{index:06d}.jpg"
                path = body_root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                color = (
                    (180, 60 + index * 20, 40)
                    if label == 0
                    else (40, 60 + index * 20, 180)
                )
                Image.new("RGB", (40, 32), color).save(path)
                if cat_id == "cat_a" and (split != "train" or train_faces):
                    face_path = face_root / relative
                    face_path.parent.mkdir(parents=True, exist_ok=True)
                    Image.new("RGB", (32, 32), color).save(face_path)

    cfg = apply_overrides(
        load_config(REPO_ROOT / "configs/experiments/meowid_base.yaml"),
        [
            f"data.root={body_root}",
            f"data.face_root={face_root}",
            f"experiment.output_dir={output_dir}",
            "model.embedding_dim=16",
            "model.backbone.hidden_size=32",
            "model.backbone.intermediate_size=64",
            "model.backbone.num_hidden_layers=1",
            "model.backbone.num_attention_heads=4",
            "model.pool.gate_hidden_dim=16",
            "model.meowid.hint_hidden_dim=16",
            "train.epochs=1",
            "train.amp=false",
            "train.log_interval=1",
            "scheduler.warmup_epochs=0",
        ],
    )
    for transforms in (cfg.data.transforms, cfg.data.face_transforms):
        transforms.train.image_size = 32
        transforms.eval.image_size = 32
        transforms.eval.resize_size = 32
    for split in ("train", "val", "test"):
        cfg.data[split].batch_size = 4
        cfg.data[split].num_workers = 0
        cfg.data[split].pin_memory = False

    # Generate compatible expert checkpoints without downloading model weights.
    expert_cfg = apply_overrides(cfg, ["model.type=single"])
    torch.manual_seed(42)
    initial_body_weight = None
    for branch in ("body", "face"):
        expert = build_model(expert_cfg, num_classes=2, with_head=True)
        checkpoint_path = tmp_path / f"{branch}_expert.pth"
        torch.save(
            {"model": expert.state_dict(), "class_to_idx": mapping}, checkpoint_path
        )
        cfg.model.meowid[f"{branch}_checkpoint"] = str(checkpoint_path)
        if branch == "body":
            initial_body_weight = expert.state_dict()[
                "backbone.model.embeddings.patch_embeddings.weight"
            ].clone()
    config_path = tmp_path / "smoke.yaml"
    dump_config(cfg, config_path)

    monkeypatch.syspath_prepend(str(REPO_ROOT / "tools"))
    spec = importlib.util.spec_from_file_location(
        "_meowid_training_smoke", REPO_ROOT / "tools/train.py"
    )
    assert spec is not None and spec.loader is not None
    training = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(training)
    monkeypatch.setattr(training, "build_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(sys, "argv", ["train.py", "--config", str(config_path)])
    training.main()

    metrics = json.loads((output_dir / "metrics.jsonl").read_text().splitlines()[0])
    assert math.isfinite(metrics["train"]["loss"])
    assert metrics["train"]["face_ratio"] == (0.5 if train_faces else 0.0)
    checkpoint = torch.load(
        output_dir / "checkpoints/latest.pth", map_location="cpu", weights_only=False
    )
    trained_body_weight = checkpoint["model"][
        "body_expert.backbone.model.embeddings.patch_embeddings.weight"
    ]
    assert not torch.equal(initial_body_weight, trained_body_weight)
    for name in ("best_body.pth", "best_face.pth", "best_deployment.pth"):
        assert (output_dir / "checkpoints" / name).is_file()
    report = json.loads((output_dir / "test_report_best_experts.json").read_text())
    assert report["protocol"]["single_whole_cat_encoder"]
    assert (
        report["protocol"]["body_epoch"]
        == report["protocol"]["face_hint_body_epoch"]
        == 1
    )
    for route in ("body", "face", "body_no_face", "hard_route"):
        assert report[route]["num_queries"] > 0
        assert math.isfinite(report[route]["mAP"])

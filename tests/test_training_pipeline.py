from __future__ import annotations

from pathlib import Path

from PIL import Image

from cat_recognition.config import apply_overrides, load_config
from cat_recognition.data import build_paired_face_hint_split_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs/experiments/meowid_base.yaml"


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 48), color=color).save(path)


def test_training_config_uses_portable_paths() -> None:
    cfg = load_config(CONFIG)
    assert cfg.model.meowid.body_checkpoint == "artifacts/training_init/body_expert.pth"
    assert cfg.model.meowid.face_checkpoint == "artifacts/training_init/face_expert.pth"
    assert not Path(cfg.data.root).is_absolute()
    assert not Path(cfg.data.face_root).is_absolute()
    assert cfg.data.train.sampler.name == "default"


def test_paired_dataset_keeps_body_only_samples(tmp_path: Path) -> None:
    body_root = tmp_path / "icw_split"
    face_root = tmp_path / "faces"
    _write_image(body_root / "train/cat_a/000000.jpg", (200, 100, 50))
    _write_image(body_root / "train/cat_b/000000.jpg", (50, 100, 200))
    _write_image(face_root / "train/cat_a/000000.jpg", (180, 90, 40))

    cfg = apply_overrides(
        load_config(CONFIG),
        [f"data.root={body_root}", f"data.face_root={face_root}"],
    )
    dataset = build_paired_face_hint_split_dataset(
        data_cfg=cfg.data,
        split_name="train",
        is_train=False,
        root_dir=REPO_ROOT,
        build_class_to_idx=True,
    )

    assert len(dataset) == 2
    assert dataset.num_classes == 2
    assert dataset.face_count == 1
    samples = {dataset[index]["cat_id"]: dataset[index] for index in range(len(dataset))}
    assert bool(samples["cat_a"]["face_exists"])
    assert not bool(samples["cat_b"]["face_exists"])
    assert samples["cat_a"]["image"].shape == samples["cat_a"]["face_image"].shape

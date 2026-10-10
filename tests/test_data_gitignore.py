from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def test_gitignore_keeps_data_source_packages(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    shutil.copyfile(repo_root / ".gitignore", tmp_path / ".gitignore")
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    paths = [
        "data/icw_split/train/cat_a/image.jpg",
        "src/cat_recognition/data/__init__.py",
        "third_party/EdgeCrafter/ecpose/engine/data/__init__.py",
        "third_party/EdgeCrafter/ecdetseg/engine/data/__init__.py",
    ]
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", *paths],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    assert result.stdout.splitlines() == paths[:1]

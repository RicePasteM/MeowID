from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARTIFACTS = REPO_ROOT / "artifacts/MeowID-Base"
DEFAULT_ECSEG_ARTIFACTS = REPO_ROOT / "artifacts/ECSeg"


def parse_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        digest, name = line.split(maxsplit=1)
        checksums[name.strip()] = digest
    return checksums


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifacts(root: Path) -> None:
    expected = parse_checksums(root / "SHA256SUMS")
    for name, digest in expected.items():
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != digest:
            raise RuntimeError(f"Checksum mismatch for {name}: {actual} != {digest}")
        print(f"[OK] {name} ({path.stat().st_size / 1024**2:.1f} MiB)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify MeowID delivery artifacts")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument("--ecseg-artifacts", type=Path, default=DEFAULT_ECSEG_ARTIFACTS)
    parser.add_argument("--skip-ecseg", action="store_true")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--backend", choices=("torch", "onnx", "tensorrt"), default="torch")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    root = args.artifacts.expanduser().resolve()
    verify_artifacts(root)
    print(f"[OK] artifact manifest: {root / 'deployment.json'}")
    ecseg_root = args.ecseg_artifacts.expanduser().resolve()
    if not args.skip_ecseg:
        verify_artifacts(ecseg_root)
        print(f"[OK] ECSeg artifact manifest: {ecseg_root / 'deployment.json'}")
    if args.image is None:
        print("Artifact integrity check passed. Add --image to run inference.")
        return

    from cat_recognition import MeowID

    sdk = MeowID(root, backend=args.backend, device=args.device, batch_size=1)
    result = sdk.embed(args.image, return_aligned=True)[0]
    print(
        "Inference passed:",
        {
            "backend": args.backend,
            "route": result.route,
            "face_detected": result.face_detected,
            "embedding_dim": int(result.embedding.shape[0]),
            "embedding_norm": float((result.embedding**2).sum() ** 0.5),
        },
    )
    if not args.skip_ecseg:
        from cat_recognition import CatCropper

        cropper = CatCropper(ecseg_root, device=args.device, batch_size=1)
        crop_result = cropper(args.image, output_size=256)[0]
        print(
            "Whole-cat crop passed:",
            {
                "cats": len(crop_result.cats),
                "scores": [round(cat.score, 4) for cat in crop_result.cats],
                "crop_sizes": [cat.crop.size for cat in crop_result.cats],
            },
        )


if __name__ == "__main__":
    main()

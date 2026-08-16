from __future__ import annotations

import argparse
from pathlib import Path

from cat_recognition import MeowID

REPO_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_ROOT = Path(__file__).resolve().parent / "icw_sample"
QUERY_NAME = "000000.jpg"


def load_sample() -> list[tuple[str, Path, list[Path]]]:
    """Return the five ICW identities as (cat_id, query, gallery) tuples."""

    identities = []
    for identity_dir in sorted(path for path in SAMPLE_ROOT.iterdir() if path.is_dir()):
        query = identity_dir / QUERY_NAME
        gallery = sorted(path for path in identity_dir.glob("*.jpg") if path != query)
        if not query.is_file() or len(gallery) != 5:
            raise RuntimeError(
                f"Expected one query and five gallery images in {identity_dir}"
            )
        identities.append((identity_dir.name, query, gallery))

    if len(identities) != 5:
        raise RuntimeError(f"Expected five sample identities in {SAMPLE_ROOT}")
    return identities


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register and retrieve five cats sampled from the ICW test split."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=REPO_ROOT / "artifacts/MeowID-Base",
        help="MeowID deployment artifact directory.",
    )
    parser.add_argument(
        "--backend",
        choices=("torch", "onnx", "tensorrt"),
        default="torch",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Inference device, for example cuda:0 or cpu (default: auto).",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=REPO_ROOT / "outputs/icw-five-cat-registry",
        help="Directory used for the example registry.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    identities = load_sample()
    model = MeowID(
        args.model,
        backend=args.backend,
        device=args.device,
        registry=args.registry,
        batch_size=args.batch_size,
    )

    # Rebuild this dedicated example registry on every run for deterministic output.
    model.registry.clear()
    for cat_id, _, gallery in identities:
        model.register(cat_id, gallery, replace=True, save=False)
    model.save_registry()

    correct = 0
    print("expected  predicted  route  score")
    print("--------  ---------  -----  -----")
    for expected, query, _ in identities:
        prediction = model.search(query, top_k=len(identities))[0]
        top_match = prediction.matches[0] if prediction.matches else None
        predicted = top_match.cat_id if top_match is not None else "<none>"
        score = f"{top_match.score:.4f}" if top_match is not None else "n/a"
        correct += int(predicted == expected)
        print(f"{expected:8}  {predicted:9}  {prediction.embedding.route:5}  {score}")

    print(f"\nTop-1 accuracy: {correct}/{len(identities)}")
    print(f"Registry: {args.registry}")


if __name__ == "__main__":
    main()

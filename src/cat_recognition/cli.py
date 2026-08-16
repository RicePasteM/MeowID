from __future__ import annotations

import argparse
import json
from pathlib import Path

from .api import MeowID
from .deployment.export import export_deployment
from .segmentation import CatCropper


def _print(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="meowid", description="MeowID-Base deployment CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Export ECPose and MeowID-Base")
    export_parser.add_argument("--format", choices=["onnx", "tensorrt", "all"], default="all")
    export_parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    export_parser.add_argument("--output-dir", type=Path, default=None)
    export_parser.add_argument("--min-batch", type=int, default=1)
    export_parser.add_argument("--opt-batch", type=int, default=4)
    export_parser.add_argument("--max-batch", type=int, default=16)
    export_parser.add_argument("--workspace-gib", type=float, default=8.0)
    export_parser.add_argument("--fp32", action="store_true", help="Build TensorRT engines in FP32")
    export_parser.add_argument(
        "--no-onnxslim",
        action="store_true",
        help="Skip the default ONNXSlim optimization pass",
    )

    for command in ("embed", "search"):
        child = subparsers.add_parser(command)
        child.add_argument("source", nargs="+")
        child.add_argument("--model", type=Path, default=None)
        child.add_argument("--backend", choices=["torch", "onnx", "tensorrt"], default="torch")
        child.add_argument("--device", default=None)
        child.add_argument("--registry", type=Path, default=None)
        child.add_argument("--batch-size", type=int, default=4)
        if command == "search":
            child.add_argument("--top-k", type=int, default=5)
            child.add_argument("--threshold", type=float, default=None)

    register_parser = subparsers.add_parser("register")
    register_parser.add_argument("cat_id")
    register_parser.add_argument("source", nargs="+")
    register_parser.add_argument("--model", type=Path, default=None)
    register_parser.add_argument("--backend", choices=["torch", "onnx", "tensorrt"], default="torch")
    register_parser.add_argument("--device", default=None)
    register_parser.add_argument("--registry", type=Path, required=True)
    register_parser.add_argument("--replace", action="store_true")

    crop_parser = subparsers.add_parser("crop", help="Extract complete cats with ECSeg")
    crop_parser.add_argument("source", nargs="+")
    crop_parser.add_argument("--model", type=Path, default=Path("artifacts/ECSeg"))
    crop_parser.add_argument("--device", default=None)
    crop_parser.add_argument("--output-dir", type=Path, required=True)
    crop_parser.add_argument("--output-size", type=int, default=512)
    crop_parser.add_argument("--threshold", type=float, default=0.4)
    crop_parser.add_argument("--top-k", type=int, default=1)
    crop_parser.add_argument("--padding", type=float, default=0.06)
    crop_parser.add_argument("--keep-background", action="store_true")
    crop_parser.add_argument("--half", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "export":
        _print(
            export_deployment(
                format=args.format,
                repo_root=args.repo_root,
                output_dir=args.output_dir,
                min_batch=args.min_batch,
                opt_batch=args.opt_batch,
                max_batch=args.max_batch,
                fp16=not args.fp32,
                workspace_gib=args.workspace_gib,
                use_onnxslim=not args.no_onnxslim,
            )
        )
        return

    if args.command == "crop":
        cropper = CatCropper(
            args.model,
            device=args.device,
            threshold=args.threshold,
            half=args.half,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        results = cropper.crop(
            args.source,
            top_k=args.top_k,
            padding=args.padding,
            output_size=args.output_size,
            mask_background=not args.keep_background,
        )
        payload = []
        for image_index, result in enumerate(results):
            metadata = result.to_dict()
            for cat_index, cat in enumerate(result.cats):
                source_stem = (
                    Path(result.source).stem
                    if not result.source.startswith("memory:")
                    else "memory"
                )
                output_path = (
                    args.output_dir
                    / f"{image_index:06d}_{source_stem}_cat_{cat_index:02d}.png"
                )
                cat.crop.save(output_path)
                metadata["cats"][cat_index]["output"] = str(output_path.resolve())
            payload.append(metadata)
        _print(payload)
        return

    sdk = MeowID(
        args.model,
        backend=args.backend,
        device=args.device,
        registry=args.registry,
        batch_size=getattr(args, "batch_size", 4),
    )
    if args.command == "embed":
        _print([item.to_dict(include_embedding=True) for item in sdk.embed(args.source)])
    elif args.command == "register":
        _print(sdk.register(args.cat_id, args.source, replace=args.replace))
    elif args.command == "search":
        _print(
            [
                item.to_dict()
                for item in sdk.search(args.source, top_k=args.top_k, threshold=args.threshold)
            ]
        )


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.nn.parallel import DistributedDataParallel

from common import PROJECT_ROOT, write_json

from cat_recognition.config import apply_overrides, dump_config, load_config
from cat_recognition.data import (
    build_dataloader,
    build_manifest_dataset,
    build_paired_face_hint_split_dataset,
    build_split_dataset,
    resolve_path,
)
from cat_recognition.engine import (
    MeowIDBaseTrainer,
    Trainer,
    compute_expert_route_metrics_by_aggregation,
    evaluate_retrieval,
    extract_expert_embeddings,
)
from cat_recognition.models import MeowIDBase, build_model
from cat_recognition.optim import build_optimizer, build_scheduler
from cat_recognition.utils import (
    barrier,
    destroy_distributed,
    get_rank,
    init_distributed,
    is_main_process,
    load_checkpoint,
    save_checkpoint,
    seed_everything,
    setup_logger,
    unwrap_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train cat recognition model.")
    parser.add_argument("--config", required=True, help="Path to yaml config.")
    parser.add_argument("--resume", default=None, help="Checkpoint path.")
    parser.add_argument(
        "--cfg-options",
        nargs="*",
        default=None,
        help="Override config entries, e.g. optimizer.lr=1e-4 train.epochs=10",
    )
    return parser.parse_args()


def build_device() -> torch.device:
    if torch.cuda.is_available():
        local_rank = int(__import__("os").environ.get("LOCAL_RANK", "0"))
        return torch.device("cuda", local_rank)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _cfg_to_dict(cfg):
    return cfg.to_dict() if hasattr(cfg, "to_dict") else dict(cfg)


def build_eval_case_viz_cfg(cfg, seed_offset: int = 0) -> dict | None:
    debug_cfg = cfg.get("debug", {}) if hasattr(cfg, "get") else {}
    viz_cfg = debug_cfg.get("eval_case_viz", {}) if debug_cfg else {}
    resolved = _cfg_to_dict(viz_cfg) if viz_cfg else {}
    if not resolved or not bool(resolved.get("enabled", False)):
        return None
    resolved["seed"] = int(resolved.get("seed", int(cfg.experiment.seed))) + int(seed_offset)
    return resolved


def _is_split_mode(data_cfg) -> bool:
    return str(_cfg_to_dict(data_cfg).get("source", "manifest")).lower() in {
        "split",
        "paired_face_hint_split",
    }


def _is_paired_face_hint_mode(data_cfg) -> bool:
    return str(_cfg_to_dict(data_cfg).get("source", "manifest")).lower() == "paired_face_hint_split"


def build_retrieval_loaders(data_cfg, split_name: str, train_dataset, distributed: bool):
    if not data_cfg.get(split_name):
        return None, None

    if _is_split_mode(data_cfg):
        dataset_builder = (
            build_paired_face_hint_split_dataset
            if _is_paired_face_hint_mode(data_cfg)
            else build_split_dataset
        )
        dataset = dataset_builder(
            data_cfg=data_cfg,
            split_name=split_name,
            is_train=False,
            root_dir=PROJECT_ROOT,
            class_to_idx=train_dataset.class_to_idx,
        )
        if dataset is None:
            return None, None
        loader = build_dataloader(
            dataset,
            _cfg_to_dict(data_cfg.get(split_name)),
            is_train=False,
            distributed=distributed,
        )
        return loader, loader

    split_cfg = data_cfg.get(split_name)
    query_manifest = split_cfg.get("query_manifest")
    gallery_manifest = split_cfg.get("gallery_manifest")
    manifest = split_cfg.get("manifest")

    if not (query_manifest or gallery_manifest or manifest):
        return None, None

    query_cfg = _cfg_to_dict(split_cfg)
    gallery_cfg = _cfg_to_dict(split_cfg)
    query_cfg["manifest"] = query_manifest or manifest
    gallery_cfg["manifest"] = gallery_manifest or query_manifest or manifest

    query_dataset = build_manifest_dataset(
        split_cfg=query_cfg,
        data_cfg=data_cfg,
        is_train=False,
        root_dir=PROJECT_ROOT,
        class_to_idx=train_dataset.class_to_idx,
    )
    query_loader = build_dataloader(
        query_dataset,
        query_cfg,
        is_train=False,
        distributed=distributed,
    )

    if gallery_cfg["manifest"] == query_cfg["manifest"]:
        return query_loader, query_loader

    gallery_dataset = build_manifest_dataset(
        split_cfg=gallery_cfg,
        data_cfg=data_cfg,
        is_train=False,
        root_dir=PROJECT_ROOT,
        class_to_idx=train_dataset.class_to_idx,
    )
    gallery_loader = build_dataloader(
        gallery_dataset,
        gallery_cfg,
        is_train=False,
        distributed=distributed,
    )
    return query_loader, gallery_loader


def log_retrieval_summary(logger, title: str, split_name: str, metrics: dict[str, float]) -> None:
    logger.info("=" * 88)
    logger.info(title)
    if not metrics:
        logger.info("%s | unavailable", split_name)
        logger.info("=" * 88)
        return

    parts = []
    if "num_queries" in metrics:
        parts.append(f"queries={metrics['num_queries']:.0f}")
    if "num_gallery" in metrics:
        parts.append(f"gallery={metrics['num_gallery']:.0f}")
    if "num_gallery_cats" in metrics:
        parts.append(f"gallery_cats={metrics['num_gallery_cats']:.0f}")
    for key in sorted(metrics):
        if key.startswith("top") and not key.startswith("top_"):
            parts.append(f"{key}={metrics[key]:.4f}")
    if "mAP" in metrics:
        parts.append(f"mAP={metrics['mAP']:.4f}")
    if "auc" in metrics:
        if "auc_gap_threshold" in metrics:
            parts.append(f"auc={metrics['auc']:.4f}@gap={metrics['auc_gap_threshold']:.2f}")
        else:
            parts.append(f"auc={metrics['auc']:.4f}")
    if "best_balanced_acc" in metrics and "best_balanced_acc_threshold" in metrics:
        parts.append(
            f"balanced_acc={metrics['best_balanced_acc']:.4f}"
            f"@{metrics['best_balanced_acc_threshold']:.2f}"
        )
    if "best_hscore" in metrics and "best_hscore_threshold" in metrics:
        parts.append(
            f"hscore={metrics['best_hscore']:.4f}"
            f"@{metrics['best_hscore_threshold']:.2f}"
        )
    logger.info("%s | %s", split_name, " | ".join(parts))
    logger.info("=" * 88)


def run_meowid_base_final_test(
    model,
    test_query_loader,
    test_gallery_loader,
    device: torch.device,
    cfg,
    output_dir: str,
    logger,
) -> None:
    checkpoint_dir = Path(output_dir) / "checkpoints"
    body_checkpoint_path = checkpoint_dir / "best_body.pth"
    face_checkpoint_path = checkpoint_dir / "best_face.pth"
    missing = [
        str(path)
        for path in (body_checkpoint_path, face_checkpoint_path)
        if not path.is_file()
    ]
    barrier()
    if missing:
        if is_main_process():
            logger.warning("Skipping MeowID-Base final test; missing checkpoints=%s", missing)
        return

    use_amp = bool(cfg.train.get("amp", False) and device.type == "cuda")
    gallery_loader = test_gallery_loader or test_query_loader

    unwrapped = unwrap_model(model)
    if bool(getattr(unwrapped, "dynamic_best_body_hint", False)):
        # Load the face/fusion parameters together with the frozen teacher they
        # were selected against, then replace the live body branch with the
        # independently selected best body checkpoint.
        face_checkpoint = load_checkpoint(
            str(face_checkpoint_path),
            model=model,
            map_location="cpu",
            strict=True,
        )
        body_checkpoint = torch.load(str(body_checkpoint_path), map_location="cpu")
        unwrapped.load_body_branch_state(body_checkpoint["model"])
        if not unwrapped.best_body_hint_matches_body():
            raise RuntimeError(
                "best_face.pth was not validated with the final best_body.pth hint; "
                "refusing to export an inconsistent deployment checkpoint"
            )

        body_epoch = int(body_checkpoint.get("epoch", -1))
        face_epoch = int(face_checkpoint.get("epoch", -1))
        face_summary = (face_checkpoint.get("best_summary") or {}).get("face") or {}
        raw_hint_body_epoch = face_summary.get("hint_body_epoch")
        hint_body_epoch = int(raw_hint_body_epoch) if raw_hint_body_epoch is not None else -1
        if hint_body_epoch != body_epoch:
            raise RuntimeError(
                f"Face checkpoint hint epoch {hint_body_epoch + 1} does not match "
                f"body checkpoint epoch {body_epoch + 1}"
            )

        # From here onward the single body_expert is used both as fallback and
        # as the face hint. Removing the teacher also removes its parameters
        # from state_dict, so deployment has no duplicate whole-cat encoder.
        unwrapped.prepare_for_single_body_deployment()
        deployment_checkpoint_path = checkpoint_dir / "best_deployment.pth"
        if is_main_process():
            deployment_extra = dict(face_checkpoint.get("extra") or {})
            deployment_extra.update(
                {
                    "selected_branch": "body+face",
                    "single_body_deployment": True,
                    "body_epoch": body_epoch,
                    "face_epoch": face_epoch,
                    "face_hint_body_epoch": hint_body_epoch,
                }
            )
            save_checkpoint(
                deployment_checkpoint_path,
                model=model,
                epoch=max(body_epoch, face_epoch),
                best_summary=face_checkpoint.get("best_summary"),
                config=cfg.to_dict(),
                extra=deployment_extra,
            )
        barrier()

        query_routes = extract_expert_embeddings(
            model,
            test_query_loader,
            device,
            use_amp=use_amp,
            routes=("body", "face"),
        )
        if gallery_loader is test_query_loader:
            gallery_routes = query_routes
        else:
            gallery_routes = extract_expert_embeddings(
                model,
                gallery_loader,
                device,
                use_amp=use_amp,
                routes=("body", "face"),
            )
        route_metrics = compute_expert_route_metrics_by_aggregation(
            body_query_data=query_routes["body"],
            body_gallery_data=gallery_routes["body"],
            face_query_data=query_routes["face"],
            face_gallery_data=gallery_routes["face"],
            eval_cfg=cfg.evaluation,
        )
        primary = str(cfg.evaluation.get("aggregation", "max")).lower()
        report = {
            "protocol": {
                "deployment_checkpoint": str(deployment_checkpoint_path),
                "single_whole_cat_encoder": True,
                "body_epoch": body_epoch + 1,
                "face_epoch": face_epoch + 1,
                "face_hint_body_epoch": hint_body_epoch + 1,
                "face_query_route": "face + shared best body hint -> face gallery",
                "no_face_query_route": "same best body expert -> whole-image body gallery",
                "cross_space_similarity": False,
                "aggregation": primary,
            },
            "body": route_metrics.get("body", {}).get(primary, {}),
            "face": route_metrics.get("face", {}).get(primary, {}),
            "body_no_face": route_metrics.get("body_no_face", {}).get(primary, {}),
            "hard_route": route_metrics.get("hard_route", {}).get(primary, {}),
            "aggregations": route_metrics,
        }
        if is_main_process():
            log_retrieval_summary(logger, "Final Body Expert Test", "body", report["body"])
            log_retrieval_summary(logger, "Final Face Expert Test", "face", report["face"])
            log_retrieval_summary(
                logger,
                "Final Single-Body Hard-Route Test",
                "hard_route",
                report["hard_route"],
            )
            write_json(str(Path(output_dir) / "test_report_best_experts.json"), report)
        del body_checkpoint, face_checkpoint
        return

    raise RuntimeError("MeowID-Base final test requires its best-body hint teacher")


def main():
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args.cfg_options)
    if args.resume:
        cfg.train.resume = args.resume

    distributed = init_distributed(backend=str(cfg.distributed.get("backend", "nccl")))
    device = build_device()
    output_dir = resolve_path(PROJECT_ROOT, cfg.experiment.output_dir)
    logger = setup_logger(output_dir=output_dir, rank=get_rank())
    seed_everything(int(cfg.experiment.seed) + get_rank(), deterministic=bool(cfg.experiment.get("deterministic", False)))

    if is_main_process():
        dump_config(cfg, output_dir + "/resolved_config.yaml")

    if _is_paired_face_hint_mode(cfg.data):
        train_dataset = build_paired_face_hint_split_dataset(
            data_cfg=cfg.data,
            split_name="train",
            is_train=True,
            root_dir=PROJECT_ROOT,
            build_class_to_idx=True,
        )
    elif _is_split_mode(cfg.data):
        train_dataset = build_split_dataset(
            data_cfg=cfg.data,
            split_name="train",
            is_train=True,
            root_dir=PROJECT_ROOT,
            build_class_to_idx=True,
        )
    else:
        train_dataset = build_manifest_dataset(
            split_cfg=cfg.data.train,
            data_cfg=cfg.data,
            is_train=True,
            root_dir=PROJECT_ROOT,
            build_class_to_idx=True,
        )
    train_loader = build_dataloader(train_dataset, cfg.data.train, is_train=True, distributed=distributed)

    val_query_loader, val_gallery_loader = build_retrieval_loaders(
        data_cfg=cfg.data,
        split_name="val",
        train_dataset=train_dataset,
        distributed=distributed,
    )
    test_query_loader, test_gallery_loader = build_retrieval_loaders(
        data_cfg=cfg.data,
        split_name="test",
        train_dataset=train_dataset,
        distributed=distributed,
    )

    model = build_model(
        cfg,
        num_classes=train_dataset.num_classes,
        with_head=True,
        class_to_idx=train_dataset.class_to_idx,
    ).to(device)
    is_meowid_base = isinstance(model, MeowIDBase)

    if is_main_process() and hasattr(model, "initialization_info"):
        logger.info("expert_initialization=%s", getattr(model, "initialization_info"))

    if distributed and bool(cfg.distributed.get("sync_bn", False)) and device.type == "cuda":
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if distributed:
        ddp_kwargs = {}
        if device.type == "cuda":
            ddp_kwargs["device_ids"] = [device.index]
            ddp_kwargs["output_device"] = device.index
        ddp_kwargs["find_unused_parameters"] = bool(cfg.train.get("find_unused_parameters", False))
        model = DistributedDataParallel(model, **ddp_kwargs)

    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)

    trainer_class = MeowIDBaseTrainer if is_meowid_base else Trainer
    trainer = trainer_class(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_query_loader=val_query_loader,
        val_gallery_loader=val_gallery_loader,
        device=device,
        cfg=cfg,
        output_dir=output_dir,
        logger=logger,
        class_to_idx=train_dataset.class_to_idx,
    )

    if cfg.train.get("resume"):
        checkpoint = load_checkpoint(
            cfg.train.resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=trainer.scaler,
            map_location="cpu",
            strict=True,
        )
        trainer.start_epoch = int(checkpoint.get("epoch", -1)) + 1
        if hasattr(trainer, "restore_checkpoint_metadata"):
            trainer.restore_checkpoint_metadata(checkpoint)
        else:
            trainer.best_metric = checkpoint.get("best_metric")
            trainer.best_epoch = checkpoint.get("best_epoch")
            best_summary = checkpoint.get("best_summary") or {}
            if isinstance(best_summary, dict):
                trainer.best_train_metrics = dict(best_summary.get("train", {}))
                trainer.best_val_metrics = dict(best_summary.get("val", {}))
                if trainer.best_epoch is None and best_summary.get("epoch") is not None:
                    trainer.best_epoch = int(best_summary["epoch"])
        logger.info("Resumed from %s at epoch %d", cfg.train.resume, trainer.start_epoch)

    if is_main_process():
        write_json(output_dir + "/class_to_idx.json", train_dataset.class_to_idx)
        logger.info("num_classes=%d train_samples=%d", train_dataset.num_classes, len(train_dataset))
        if hasattr(train_dataset, "face_count"):
            logger.info(
                "train_face_coverage=%d/%d (%.2f%%)",
                train_dataset.face_count,
                len(train_dataset),
                100.0 * train_dataset.face_count / max(len(train_dataset), 1),
            )
        if val_query_loader is not None:
            logger.info("val_query_samples=%d", len(val_query_loader.dataset))
            if hasattr(val_query_loader.dataset, "face_count"):
                logger.info(
                    "val_face_coverage=%d/%d",
                    val_query_loader.dataset.face_count,
                    len(val_query_loader.dataset),
                )
        if val_gallery_loader is not None:
            logger.info("val_gallery_samples=%d", len(val_gallery_loader.dataset))
        if test_query_loader is not None:
            logger.info("test_query_samples=%d", len(test_query_loader.dataset))
            if hasattr(test_query_loader.dataset, "face_count"):
                logger.info(
                    "test_face_coverage=%d/%d",
                    test_query_loader.dataset.face_count,
                    len(test_query_loader.dataset),
                )
        if test_gallery_loader is not None:
            logger.info("test_gallery_samples=%d", len(test_gallery_loader.dataset))

    trainer.fit()

    should_test_after_fit = bool(cfg.train.get("test_after_fit", False))
    if should_test_after_fit and test_query_loader is not None:
        if is_meowid_base:
            run_meowid_base_final_test(
                model=model,
                test_query_loader=test_query_loader,
                test_gallery_loader=test_gallery_loader,
                device=device,
                cfg=cfg,
                output_dir=output_dir,
                logger=logger,
            )
            destroy_distributed()
            return
        checkpoint_name = str(cfg.train.get("test_checkpoint_name", "best.pth"))
        checkpoint_path = resolve_path(output_dir, f"checkpoints/{checkpoint_name}")
        checkpoint_file = Path(checkpoint_path) if checkpoint_path is not None else None
        barrier()
        if checkpoint_file is None:
            if is_main_process():
                logger.warning("Skipping final test: invalid checkpoint path for %s", checkpoint_name)
        elif not checkpoint_file.exists():
            if is_main_process():
                logger.warning("Skipping final test: checkpoint not found at %s", checkpoint_file)
        else:
            load_checkpoint(str(checkpoint_file), model=model, map_location="cpu", strict=True)
            test_case_viz_cfg = build_eval_case_viz_cfg(cfg, seed_offset=1000)
            test_metrics = evaluate_retrieval(
                model=model,
                query_loader=test_query_loader,
                gallery_loader=test_gallery_loader or test_query_loader,
                device=device,
                eval_cfg=cfg.evaluation,
                use_amp=bool(cfg.train.get("amp", False) and device.type == "cuda"),
                visualization_dir=(
                    Path(output_dir) / "test_cases" / f"final_{checkpoint_file.stem}"
                    if test_case_viz_cfg is not None
                    else None
                ),
                visualization_cfg=test_case_viz_cfg,
                visualization_metadata={
                    "split": "test",
                    "stage": "final",
                    "checkpoint": checkpoint_file.name,
                },
            )
            if is_main_process():
                log_retrieval_summary(logger, "Final Test Summary", "test", test_metrics)
                write_json(output_dir + f"/test_report_{checkpoint_name.rsplit('.', 1)[0]}.json", test_metrics)
    destroy_distributed()


if __name__ == "__main__":
    main()

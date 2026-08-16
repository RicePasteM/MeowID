from __future__ import annotations

import json
from datetime import datetime
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn

from cat_recognition.engine.evaluator import evaluate_retrieval_aggregations
from cat_recognition.losses import build_aux_loss
from cat_recognition.utils import (
    is_main_process,
    reduce_dict_sum,
    save_checkpoint,
    save_training_batch_preview,
    unwrap_model,
)


def _autocast_context(use_amp: bool):
    if use_amp:
        return torch.cuda.amp.autocast()
    return nullcontext()


def _cfg_to_dict(cfg) -> dict:
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    return dict(cfg)


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        train_loader,
        val_query_loader,
        val_gallery_loader,
        device: torch.device,
        cfg,
        output_dir: str | Path,
        logger,
        class_to_idx: dict[str, int] | None = None,
        validation_fn=None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_query_loader = val_query_loader
        self.val_gallery_loader = val_gallery_loader
        self.device = device
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.logger = logger
        self.class_to_idx = class_to_idx or {}
        self.validation_fn = validation_fn
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=float(cfg.train.get("label_smoothing", 0.0))
        )
        self.aux_loss_cfg = cfg.train.get("aux_loss", {})
        self.aux_criterion = build_aux_loss(self.aux_loss_cfg)
        self.use_amp = bool(cfg.train.get("amp", False) and device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.start_epoch = 0
        self.best_metric = None
        self.best_epoch = None
        self.best_train_metrics: dict[str, float] = {}
        self.best_val_metrics: dict[str, float] = {}
        self.backbone_tuning_state: str | None = None
        self.train_batch_viz_cfg = cfg.get("debug", {}).get("train_batch_viz", {})
        self.eval_case_viz_cfg = cfg.get("debug", {}).get("eval_case_viz", {})
        self.train_batch_viz_dir = self._create_train_batch_viz_dir()

    def restore_checkpoint_metadata(self, checkpoint: dict) -> None:
        self.best_metric = checkpoint.get("best_metric")
        self.best_epoch = checkpoint.get("best_epoch")
        summary = checkpoint.get("best_summary") or {}
        self.best_train_metrics = dict(summary.get("train", {}))
        self.best_val_metrics = dict(summary.get("val", {}))

    def _format_metrics(self, metrics: dict[str, float]) -> str:
        if not metrics:
            return "{}"

        formatted = []
        for key in sorted(metrics.keys()):
            value = metrics[key]
            if isinstance(value, float):
                formatted.append(f"{key}={value:.4f}")
            else:
                formatted.append(f"{key}={value}")
        return " ".join(formatted)

    def _format_train_metrics(self, metrics: dict[str, float]) -> str:
        ordered = []
        for key in (
            "loss",
            "cls_loss",
            "aux_loss",
            "cls_acc",
        ):
            if key in metrics:
                ordered.append(f"{key}={metrics[key]:.4f}")
        known = {
            "loss", "cls_loss", "aux_loss", "cls_acc",
        }
        extras = sorted(key for key in metrics.keys() if key not in known)
        for key in extras:
            value = metrics[key]
            ordered.append(f"{key}={value:.4f}" if isinstance(value, float) else f"{key}={value}")
        return " | ".join(ordered)

    def _format_val_metrics(self, metrics: dict[str, float]) -> str:
        if not metrics:
            return "skipped"
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
        for key in ("macro_auc", "pooled_auc"):
            if key in metrics:
                parts.append(f"{key}={metrics[key]:.4f}")
        return " | ".join(parts)

    def _log_epoch_summary(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        val_aggregation_metrics: dict[str, dict[str, float]],
        val_ran: bool,
        val_interval: int,
    ) -> None:
        if not is_main_process():
            return

        self.logger.info("=" * 88)
        self.logger.info("Epoch %03d Summary", epoch + 1)
        self.logger.info("train | %s", self._format_train_metrics(train_metrics))
        if val_ran:
            primary_aggregation = self._get_primary_aggregation()
            if len(val_aggregation_metrics) <= 1:
                self.logger.info("val[%s] | %s", primary_aggregation, self._format_val_metrics(val_metrics))
            else:
                for aggregation, metrics in val_aggregation_metrics.items():
                    self.logger.info("val[%s] | %s", aggregation, self._format_val_metrics(metrics))
        else:
            self.logger.info("val   | skipped (val_interval=%d)", val_interval)
        self.logger.info("=" * 88)

    def _get_val_interval(self) -> int:
        return int(self.cfg.train.get("val_interval", self.cfg.train.get("eval_interval", 1)))

    def _get_primary_aggregation(self) -> str:
        return str(self.cfg.evaluation.get("aggregation", "max")).lower()

    def _get_monitor_aggregation(self) -> str:
        return str(self.cfg.train.get("monitor_aggregation", self._get_primary_aggregation())).lower()

    def _create_train_batch_viz_dir(self) -> Path | None:
        if not is_main_process() or not bool(self.train_batch_viz_cfg.get("enabled", False)):
            return None
        base_dir = self.output_dir / "tmp"
        base_dir.mkdir(parents=True, exist_ok=True)
        prefix = str(self.train_batch_viz_cfg.get("prefix", "train_batch_viz_"))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_dir = base_dir / f"{prefix}{timestamp}"
        suffix = 1
        while debug_dir.exists():
            debug_dir = base_dir / f"{prefix}{timestamp}_{suffix:02d}"
            suffix += 1
        debug_dir.mkdir(parents=True, exist_ok=False)
        self.logger.info("train_batch_viz_dir=%s", debug_dir)
        return debug_dir

    def _dump_train_batch_preview(self, epoch: int, step: int, batch) -> None:
        if self.train_batch_viz_dir is None:
            return

        max_epochs = int(self.train_batch_viz_cfg.get("num_epochs", 3))
        if epoch + 1 > max_epochs:
            return

        max_steps = int(self.train_batch_viz_cfg.get("num_steps", 5))
        if step > max_steps:
            return

        normalize_cfg = self.cfg.data.transforms.train.get("normalize", {})
        mean = list(normalize_cfg.get("mean", [0.485, 0.456, 0.406]))
        std = list(normalize_cfg.get("std", [0.229, 0.224, 0.225]))
        epoch_dir = self.train_batch_viz_dir / f"epoch_{epoch + 1:03d}"
        output_path = epoch_dir / f"step_{step:04d}.jpg"

        save_training_batch_preview(
            images=batch["image"],
            labels=batch["label"],
            cat_ids=batch["cat_id"],
            paths=batch["path"],
            output_path=output_path,
            mean=mean,
            std=std,
            max_images=int(self.train_batch_viz_cfg.get("max_images", 16)),
            nrow=int(self.train_batch_viz_cfg.get("nrow", 4)),
            metadata={"epoch": epoch + 1, "step": step},
        )

    def _is_better(self, current: float) -> bool:
        if self.best_metric is None:
            return True
        if self.cfg.train.get("monitor_mode", "max") == "min":
            return current < self.best_metric
        return current > self.best_metric

    def _build_eval_case_viz_cfg(self, seed_offset: int = 0) -> dict | None:
        cfg = _cfg_to_dict(self.eval_case_viz_cfg or {})
        if not cfg or not bool(cfg.get("enabled", False)):
            return None
        cfg["seed"] = int(cfg.get("seed", int(self.cfg.experiment.seed))) + int(seed_offset)
        return cfg

    def _build_best_summary(self) -> dict | None:
        if self.best_epoch is None:
            return None
        return {
            "epoch": int(self.best_epoch),
            "train": dict(self.best_train_metrics),
            "val": dict(self.best_val_metrics),
        }

    def _apply_backbone_tuning(self, epoch: int) -> None:
        model = unwrap_model(self.model)
        backbone = getattr(model, "backbone", None)
        if backbone is None or not hasattr(backbone, "set_trainable_mode"):
            return

        backbone_cfg = self.cfg.model.get("backbone", {})
        freeze_epochs = int(backbone_cfg.get("freeze_epochs", 0))
        unfreeze_last_n_blocks = int(backbone_cfg.get("unfreeze_last_n_blocks", 0))
        full_unfreeze_epoch = int(backbone_cfg.get("full_unfreeze_epoch", 0))

        if freeze_epochs > 0 and epoch < freeze_epochs:
            state = backbone.set_trainable_mode("frozen")
        elif full_unfreeze_epoch > 0 and (epoch + 1) >= full_unfreeze_epoch:
            state = backbone.set_trainable_mode("full")
        elif unfreeze_last_n_blocks > 0:
            state = backbone.set_trainable_mode("partial", unfreeze_last_n_blocks=unfreeze_last_n_blocks)
        else:
            state = backbone.set_trainable_mode("full")

        if state != self.backbone_tuning_state:
            self.backbone_tuning_state = state
            if is_main_process():
                self.logger.info("backbone_tuning | epoch=%d | mode=%s", epoch + 1, state)

    def _log_final_summary(self) -> None:
        if not is_main_process():
            return

        self.logger.info("=" * 88)
        self.logger.info("Training Complete")
        if self.best_epoch is None:
            self.logger.info("best | unavailable (validation did not run)")
            self.logger.info("=" * 88)
            return

        monitor = str(self.cfg.train.get("monitor", "top1"))
        monitor_aggregation = self._get_monitor_aggregation()
        if self.best_metric is None:
            self.logger.info("best[%s] | epoch=%03d", monitor_aggregation, self.best_epoch + 1)
        else:
            self.logger.info(
                "best[%s] | epoch=%03d | %s=%.4f",
                monitor_aggregation,
                self.best_epoch + 1,
                monitor,
                self.best_metric,
            )
        if self.best_train_metrics:
            self.logger.info("best_train | %s", self._format_train_metrics(self.best_train_metrics))
        if self.best_val_metrics:
            self.logger.info("best_val   | %s", self._format_val_metrics(self.best_val_metrics))
        self.logger.info("=" * 88)

    def _save_metrics(
        self,
        epoch: int,
        train_metrics: dict,
        val_metrics: dict,
        val_aggregation_metrics: dict[str, dict[str, float]] | None = None,
    ) -> None:
        if not is_main_process():
            return
        path = self.output_dir / "metrics.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "val_aggregations": val_aggregation_metrics or {},
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def _save_val_report(
        self,
        epoch: int,
        val_metrics: dict[str, float],
        val_aggregation_metrics: dict[str, dict[str, float]] | None = None,
    ) -> None:
        if not is_main_process() or not val_metrics:
            return
        report_dir = self.output_dir / "val_metrics"
        report_dir.mkdir(parents=True, exist_ok=True)
        with (report_dir / f"epoch_{epoch + 1:03d}.json").open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "primary_aggregation": self._get_primary_aggregation(),
                    "monitor_aggregation": self._get_monitor_aggregation(),
                    "metrics": val_metrics,
                    "aggregations": val_aggregation_metrics or {},
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )

    def save(self, epoch: int, name: str) -> None:
        if not is_main_process():
            return
        save_checkpoint(
            self.output_dir / "checkpoints" / name,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            epoch=epoch,
            best_metric=self.best_metric,
            best_epoch=self.best_epoch,
            best_summary=self._build_best_summary(),
            config=self.cfg.to_dict(),
            extra={
                "class_to_idx": self.class_to_idx,
                "embedding_dim": unwrap_model(self.model).embedding_dim,
            },
        )

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        self._apply_backbone_tuning(epoch)
        if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)
        if hasattr(self.train_loader, "batch_sampler") and hasattr(self.train_loader.batch_sampler, "set_epoch"):
            self.train_loader.batch_sampler.set_epoch(epoch)

        accumulation_steps = int(self.cfg.train.get("accumulation_steps", 1))
        grad_clip_norm = float(self.cfg.train.get("grad_clip_norm", 0.0))
        log_interval = int(self.cfg.train.get("log_interval", 20))

        self.optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0
        running_cls_loss = 0.0
        running_aux_loss = 0.0
        running_correct = 0.0
        running_count = 0

        for step, batch in enumerate(self.train_loader, start=1):
            if is_main_process():
                self._dump_train_batch_preview(epoch, step, batch)

            images = batch["image"].to(self.device, non_blocking=True)
            labels = batch["label"].to(self.device, non_blocking=True).long()
            if (labels < 0).any():
                raise ValueError("Training set contains samples without valid labels.")

            with _autocast_context(self.use_amp):
                outputs = self.model(images, labels)
                cls_loss = self.criterion(outputs["logits"], labels)
                aux_loss = cls_loss.new_zeros(())
                if self.aux_criterion is not None:
                    aux_loss = self.aux_criterion(outputs["embeddings"], labels)
                aux_weight = float(self.aux_loss_cfg.get("weight", 0.0))
                loss = cls_loss + aux_weight * aux_loss
                loss_to_backward = loss / accumulation_steps

            self.scaler.scale(loss_to_backward).backward()

            if step % accumulation_steps == 0 or step == len(self.train_loader):
                if grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            preds = outputs["logits"].argmax(dim=1)
            running_loss += float(loss.detach().item()) * labels.size(0)
            running_cls_loss += float(cls_loss.detach().item()) * labels.size(0)
            running_aux_loss += float(aux_loss.detach().item()) * labels.size(0)
            running_correct += float((preds == labels).sum().item())
            running_count += labels.size(0)

            if is_main_process() and step % log_interval == 0:
                self.logger.info(
                    "epoch=%d step=%d/%d loss=%.4f cls_acc=%.4f",
                    epoch + 1,
                    step,
                    len(self.train_loader),
                    running_loss / max(running_count, 1),
                    running_correct / max(running_count, 1),
                )

        reduced = reduce_dict_sum(
            {
                "loss_sum": running_loss,
                "cls_loss_sum": running_cls_loss,
                "aux_loss_sum": running_aux_loss,
                "correct_sum": running_correct,
                "count_sum": float(running_count),
            },
            device=self.device,
        )

        metrics = {
            "loss": reduced["loss_sum"] / max(reduced["count_sum"], 1.0),
            "cls_loss": reduced["cls_loss_sum"] / max(reduced["count_sum"], 1.0),
            "aux_loss": reduced["aux_loss_sum"] / max(reduced["count_sum"], 1.0),
            "cls_acc": reduced["correct_sum"] / max(reduced["count_sum"], 1.0),
        }
        return metrics

    def fit(self) -> None:
        for epoch in range(self.start_epoch, int(self.cfg.train.epochs)):
            train_metrics = self.train_one_epoch(epoch)

            val_metrics: dict[str, float] = {}
            val_aggregation_metrics: dict[str, dict[str, float]] = {}
            val_interval = max(1, self._get_val_interval())
            val_ran = False
            has_validation = self.validation_fn is not None or self.val_query_loader is not None
            if has_validation and (epoch + 1) % val_interval == 0:
                val_ran = True
                if self.validation_fn is not None:
                    val_metrics = dict(self.validation_fn())
                    val_aggregation_metrics = {
                        self._get_primary_aggregation(): dict(val_metrics)
                    }
                else:
                    val_case_viz_cfg = self._build_eval_case_viz_cfg(seed_offset=epoch)
                    val_aggregation_metrics = evaluate_retrieval_aggregations(
                        model=self.model,
                        query_loader=self.val_query_loader,
                        gallery_loader=self.val_gallery_loader or self.val_query_loader,
                        device=self.device,
                        eval_cfg=self.cfg.evaluation,
                        use_amp=self.use_amp,
                        visualization_dir=(
                            self.output_dir / "val_cases" / f"epoch_{epoch + 1:03d}"
                            if val_case_viz_cfg is not None
                            else None
                        ),
                        visualization_cfg=val_case_viz_cfg,
                        visualization_metadata={
                            "split": "val",
                            "epoch": epoch + 1,
                        },
                        visualization_aggregation=self._get_primary_aggregation(),
                    )
                val_metrics = dict(val_aggregation_metrics.get(self._get_primary_aggregation(), {}))
                monitor_metrics = val_aggregation_metrics.get(self._get_monitor_aggregation(), val_metrics)
                monitor = str(self.cfg.train.get("monitor", "top1"))
                current_metric = float(monitor_metrics.get(monitor, float("-inf")))
                if self._is_better(current_metric):
                    self.best_metric = current_metric
                    self.best_epoch = epoch
                    self.best_train_metrics = dict(train_metrics)
                    self.best_val_metrics = dict(monitor_metrics)
                    self.save(epoch, "best.pth")
                self._save_val_report(epoch, val_metrics, val_aggregation_metrics)

            if self.scheduler is not None:
                self.scheduler.step()

            self._save_metrics(epoch, train_metrics, val_metrics, val_aggregation_metrics)
            self._log_epoch_summary(
                epoch,
                train_metrics,
                val_metrics,
                val_aggregation_metrics,
                val_ran,
                val_interval,
            )

            if (epoch + 1) % int(self.cfg.train.get("save_interval", 1)) == 0:
                self.save(epoch, "latest.pth")

        self._log_final_summary()

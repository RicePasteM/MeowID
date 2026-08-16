from __future__ import annotations

import json
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from cat_recognition.engine.evaluator import (
    compute_expert_route_metrics_by_aggregation,
    compute_retrieval_metrics_by_aggregation,
    extract_expert_embeddings,
)
from cat_recognition.losses import build_aux_loss
from cat_recognition.utils import (
    is_main_process,
    reduce_dict_sum,
    save_checkpoint,
    unwrap_model,
)


def _autocast_context(use_amp: bool):
    if use_amp:
        return torch.cuda.amp.autocast()
    return nullcontext()


class MeowIDBaseTrainer:
    """Train MeowID-Base and couple face selection to the best body teacher."""

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
        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=float(cfg.train.get("label_smoothing", 0.0))
        )
        self.aux_loss_cfg = cfg.train.get("aux_loss", {})
        self.aux_criterion = build_aux_loss(self.aux_loss_cfg)
        self.use_amp = bool(cfg.train.get("amp", False) and device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        self.start_epoch = 0
        self.best_metrics: dict[str, float | None] = {"body": None, "face": None}
        self.best_epochs: dict[str, int | None] = {"body": None, "face": None}
        self.best_val_metrics: dict[str, dict[str, float]] = {"body": {}, "face": {}}
        self.best_train_metrics: dict[str, dict[str, float]] = {"body": {}, "face": {}}
        self.face_hint_body_epoch: int | None = None

    def restore_checkpoint_metadata(self, checkpoint: dict) -> None:
        summary = checkpoint.get("best_summary") or {}
        if not isinstance(summary, dict):
            return
        for branch in ("body", "face"):
            branch_summary = summary.get(branch) or {}
            if not isinstance(branch_summary, dict):
                continue
            if branch_summary.get("metric") is not None:
                self.best_metrics[branch] = float(branch_summary["metric"])
            if branch_summary.get("epoch") is not None:
                self.best_epochs[branch] = int(branch_summary["epoch"])
            self.best_val_metrics[branch] = dict(branch_summary.get("val", {}))
            self.best_train_metrics[branch] = dict(branch_summary.get("train", {}))
        face_summary = summary.get("face") or {}
        if isinstance(face_summary, dict) and face_summary.get("hint_body_epoch") is not None:
            self.face_hint_body_epoch = int(face_summary["hint_body_epoch"])

    def _best_summary(self) -> dict:
        summary = {
            branch: {
                "epoch": self.best_epochs[branch],
                "metric": self.best_metrics[branch],
                "train": dict(self.best_train_metrics[branch]),
                "val": dict(self.best_val_metrics[branch]),
            }
            for branch in ("body", "face")
        }
        summary["face"]["hint_body_epoch"] = self.face_hint_body_epoch
        return summary

    def _primary_aggregation(self) -> str:
        return str(self.cfg.evaluation.get("aggregation", "max")).lower()

    def _monitor_aggregation(self) -> str:
        return str(
            self.cfg.train.get("monitor_aggregation", self._primary_aggregation())
        ).lower()

    def _monitor_name(self) -> str:
        return str(self.cfg.train.get("monitor", "mAP"))

    def _is_better(self, branch: str, current: float) -> bool:
        previous = self.best_metrics[branch]
        if previous is None:
            return True
        if str(self.cfg.train.get("monitor_mode", "max")).lower() == "min":
            return current < previous
        return current > previous

    @staticmethod
    def _format_metrics(metrics: dict[str, float]) -> str:
        parts = []
        for key in ("num_queries", "num_gallery", "num_gallery_cats"):
            if key in metrics:
                parts.append(f"{key}={metrics[key]:.0f}")
        for key in sorted(metrics):
            if key.startswith("top") and not key.startswith("top_"):
                parts.append(f"{key}={metrics[key]:.4f}")
        if "mAP" in metrics:
            parts.append(f"mAP={metrics['mAP']:.4f}")
        return " | ".join(parts) if parts else "unavailable"

    @staticmethod
    def _format_train(metrics: dict[str, float]) -> str:
        order = (
            "loss",
            "body_cls_loss",
            "face_cls_loss",
            "body_aux_loss",
            "face_aux_loss",
            "body_face_align_loss",
            "face_preserve_loss",
            "body_cls_acc",
            "face_cls_acc",
            "face_ratio",
            "hint_gate_mean",
            "hint_norm_mean",
        )
        return " | ".join(
            f"{key}={metrics[key]:.4f}" for key in order if key in metrics
        )

    def save(self, epoch: int, name: str, branch: str | None = None) -> None:
        if not is_main_process():
            return
        include_training_state = name == "latest.pth"
        branch_metric = self.best_metrics.get(branch) if branch else None
        branch_epoch = self.best_epochs.get(branch) if branch else None
        model = unwrap_model(self.model)
        save_checkpoint(
            self.output_dir / "checkpoints" / name,
            model=self.model,
            optimizer=self.optimizer if include_training_state else None,
            scheduler=self.scheduler if include_training_state else None,
            scaler=self.scaler if include_training_state else None,
            epoch=epoch,
            best_metric=branch_metric,
            best_epoch=branch_epoch,
            best_summary=self._best_summary(),
            config=self.cfg.to_dict(),
            extra={
                "class_to_idx": self.class_to_idx,
                "face_class_to_idx": getattr(model, "face_class_to_idx", {}),
                "embedding_dim": model.embedding_dim,
                "architecture": "MeowID-Base",
                "selected_branch": branch,
            },
        )

    def train_one_epoch(self, epoch: int) -> dict[str, float]:
        self.model.train()
        if hasattr(self.train_loader, "sampler") and hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(epoch)
        if hasattr(self.train_loader, "batch_sampler") and hasattr(
            self.train_loader.batch_sampler, "set_epoch"
        ):
            self.train_loader.batch_sampler.set_epoch(epoch)

        accumulation_steps = int(self.cfg.train.get("accumulation_steps", 1))
        grad_clip_norm = float(self.cfg.train.get("grad_clip_norm", 0.0))
        log_interval = int(self.cfg.train.get("log_interval", 20))
        loss_cfg = self.cfg.train.get("meowid_loss", {})
        body_cls_weight = float(loss_cfg.get("body_cls_weight", 1.0))
        face_cls_weight = float(loss_cfg.get("face_cls_weight", 1.0))
        aux_weight = float(self.aux_loss_cfg.get("weight", 0.0))
        body_aux_weight = float(loss_cfg.get("body_aux_weight", aux_weight))
        face_aux_weight = float(loss_cfg.get("face_aux_weight", aux_weight))
        align_weight = float(loss_cfg.get("body_face_align_weight", 0.0))
        preserve_weight = float(loss_cfg.get("face_preserve_weight", 0.0))

        sums = {
            "loss_sum": 0.0,
            "body_cls_loss_sum": 0.0,
            "face_cls_loss_sum": 0.0,
            "body_aux_loss_sum": 0.0,
            "face_aux_loss_sum": 0.0,
            "body_face_align_loss_sum": 0.0,
            "face_preserve_loss_sum": 0.0,
            "body_correct_sum": 0.0,
            "face_correct_sum": 0.0,
            "body_count_sum": 0.0,
            "face_count_sum": 0.0,
            "hint_gate_sum": 0.0,
            "hint_norm_sum": 0.0,
        }
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(self.train_loader, start=1):
            images = batch["image"].to(self.device, non_blocking=True)
            face_images = batch["face_image"].to(self.device, non_blocking=True)
            face_mask = batch["face_exists"].to(self.device, non_blocking=True).bool()
            labels = batch["label"].to(self.device, non_blocking=True).long()
            if bool((labels < 0).any()):
                raise ValueError("Training set contains samples without valid labels")

            with _autocast_context(self.use_amp):
                outputs = self.model(
                    images,
                    labels,
                    face_images=face_images,
                    face_mask=face_mask,
                )
                body_cls_loss = self.criterion(outputs["body_logits"], labels)
                zero = body_cls_loss.new_zeros(())
                face_cls_loss = zero
                body_aux_loss = zero
                face_aux_loss = zero
                body_face_align_loss = zero
                face_preserve_loss = zero

                if self.aux_criterion is not None:
                    body_aux_loss = self.aux_criterion(outputs["body_embeddings"], labels)

                if bool(face_mask.any()):
                    face_labels = outputs["face_labels"]
                    face_route = outputs["face_route_embeddings"][face_mask]
                    raw_face = outputs["face_embeddings"][face_mask]
                    hint_body = outputs["hint_body_embeddings"][face_mask]
                    face_cls_loss = self.criterion(outputs["face_logits"], face_labels)
                    if self.aux_criterion is not None:
                        face_aux_loss = self.aux_criterion(face_route, face_labels)
                    body_face_align_loss = (
                        1.0 - F.cosine_similarity(hint_body, raw_face.detach())
                    ).mean()
                    face_preserve_loss = (
                        1.0 - F.cosine_similarity(face_route, raw_face.detach())
                    ).mean()

                loss = (
                    body_cls_weight * body_cls_loss
                    + face_cls_weight * face_cls_loss
                    + body_aux_weight * body_aux_loss
                    + face_aux_weight * face_aux_loss
                    + align_weight * body_face_align_loss
                    + preserve_weight * face_preserve_loss
                )
                loss_to_backward = loss / accumulation_steps

            self.scaler.scale(loss_to_backward).backward()
            if step % accumulation_steps == 0 or step == len(self.train_loader):
                if grad_clip_norm > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            body_count = labels.size(0)
            face_count = int(face_mask.sum().item())
            sums["loss_sum"] += float(loss.detach()) * body_count
            sums["body_cls_loss_sum"] += float(body_cls_loss.detach()) * body_count
            sums["body_aux_loss_sum"] += float(body_aux_loss.detach()) * body_count
            sums["body_correct_sum"] += float(
                (outputs["body_logits"].argmax(dim=1) == labels).sum().item()
            )
            sums["body_count_sum"] += body_count
            if face_count > 0:
                sums["face_cls_loss_sum"] += float(face_cls_loss.detach()) * face_count
                sums["face_aux_loss_sum"] += float(face_aux_loss.detach()) * face_count
                sums["body_face_align_loss_sum"] += (
                    float(body_face_align_loss.detach()) * face_count
                )
                sums["face_preserve_loss_sum"] += float(face_preserve_loss.detach()) * face_count
                sums["face_correct_sum"] += float(
                    (
                        outputs["face_logits"].argmax(dim=1)
                        == outputs["face_labels"]
                    ).sum().item()
                )
                sums["face_count_sum"] += face_count
                sums["hint_gate_sum"] += float(outputs["hint_gates"].detach().sum().item())
                sums["hint_norm_sum"] += float(outputs["hint_norms"].detach().sum().item())

            if is_main_process() and step % log_interval == 0:
                self.logger.info(
                    "epoch=%d step=%d/%d loss=%.4f body_acc=%.4f face_acc=%.4f",
                    epoch + 1,
                    step,
                    len(self.train_loader),
                    sums["loss_sum"] / max(sums["body_count_sum"], 1.0),
                    sums["body_correct_sum"] / max(sums["body_count_sum"], 1.0),
                    sums["face_correct_sum"] / max(sums["face_count_sum"], 1.0),
                )

        reduced = reduce_dict_sum(sums, device=self.device)
        body_count = max(reduced["body_count_sum"], 1.0)
        face_count = max(reduced["face_count_sum"], 1.0)
        return {
            "loss": reduced["loss_sum"] / body_count,
            "body_cls_loss": reduced["body_cls_loss_sum"] / body_count,
            "face_cls_loss": reduced["face_cls_loss_sum"] / face_count,
            "body_aux_loss": reduced["body_aux_loss_sum"] / body_count,
            "face_aux_loss": reduced["face_aux_loss_sum"] / face_count,
            "body_face_align_loss": reduced["body_face_align_loss_sum"] / face_count,
            "face_preserve_loss": reduced["face_preserve_loss_sum"] / face_count,
            "body_cls_acc": reduced["body_correct_sum"] / body_count,
            "face_cls_acc": reduced["face_correct_sum"] / face_count,
            "face_ratio": reduced["face_count_sum"] / body_count,
            "hint_gate_mean": reduced["hint_gate_sum"] / face_count,
            "hint_norm_mean": reduced["hint_norm_sum"] / face_count,
        }

    def _save_epoch_reports(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        route_metrics: dict[str, dict[str, dict[str, float]]],
    ) -> None:
        if not is_main_process():
            return
        payload = {"epoch": epoch, "train": train_metrics, "val": route_metrics}
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with (self.output_dir / "metrics.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        report_dir = self.output_dir / "val_metrics"
        report_dir.mkdir(parents=True, exist_ok=True)
        with (report_dir / f"epoch_{epoch + 1:03d}.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def _extract_route(
        self,
        loader,
        route: str,
    ) -> dict[str, list | object]:
        return extract_expert_embeddings(
            self.model,
            loader,
            self.device,
            use_amp=self.use_amp,
            routes=(route,),
        )[route]

    def _evaluate_with_dynamic_best_body_hint(
        self,
        epoch: int,
        train_metrics: dict[str, float],
        monitor_aggregation: str,
        monitor_name: str,
    ) -> dict[str, dict[str, dict[str, float]]]:
        """Select body first, refresh its teacher, then evaluate face.

        Resetting the face selection window whenever the teacher changes makes
        every saved best_face checkpoint compatible with the final best body.
        """

        gallery_loader = self.val_gallery_loader or self.val_query_loader
        body_query = self._extract_route(self.val_query_loader, "body")
        body_gallery = (
            body_query
            if gallery_loader is self.val_query_loader
            else self._extract_route(gallery_loader, "body")
        )
        body_metrics = compute_retrieval_metrics_by_aggregation(
            body_query,
            body_gallery,
            self.cfg.evaluation,
        )
        selected_body = body_metrics.get(monitor_aggregation, {})
        current_body = float(selected_body.get(monitor_name, float("-inf")))
        if self._is_better("body", current_body):
            self.best_metrics["body"] = current_body
            self.best_epochs["body"] = epoch
            self.best_val_metrics["body"] = dict(selected_body)
            self.best_train_metrics["body"] = dict(train_metrics)

            model = unwrap_model(self.model)
            model.refresh_best_body_hint()
            self.face_hint_body_epoch = epoch
            # Face scores from a previous teacher are not comparable deployment
            # candidates. The face model is evaluated immediately below with
            # the refreshed teacher, so a compatible best_face is always saved.
            self.best_metrics["face"] = None
            self.best_epochs["face"] = None
            self.best_val_metrics["face"] = {}
            self.best_train_metrics["face"] = {}
            self.save(epoch, "best_body.pth", branch="body")
            if is_main_process():
                self.logger.info(
                    "best body teacher refreshed | epoch=%03d | %s=%.4f; "
                    "face best-selection window reset",
                    epoch + 1,
                    monitor_name,
                    current_body,
                )

        face_query = self._extract_route(self.val_query_loader, "face")
        face_gallery = (
            face_query
            if gallery_loader is self.val_query_loader
            else self._extract_route(gallery_loader, "face")
        )
        route_metrics = compute_expert_route_metrics_by_aggregation(
            body_query_data=body_query,
            body_gallery_data=body_gallery,
            face_query_data=face_query,
            face_gallery_data=face_gallery,
            eval_cfg=self.cfg.evaluation,
        )

        selected_face = route_metrics.get("face", {}).get(monitor_aggregation, {})
        current_face = float(selected_face.get(monitor_name, float("-inf")))
        if self._is_better("face", current_face):
            self.best_metrics["face"] = current_face
            self.best_epochs["face"] = epoch
            self.best_val_metrics["face"] = dict(selected_face)
            self.best_train_metrics["face"] = dict(train_metrics)
            self.save(epoch, "best_face.pth", branch="face")
        return route_metrics

    def fit(self) -> None:
        val_interval = max(
            1,
            int(self.cfg.train.get("val_interval", self.cfg.train.get("eval_interval", 1))),
        )
        monitor_aggregation = self._monitor_aggregation()
        monitor_name = self._monitor_name()

        for epoch in range(self.start_epoch, int(self.cfg.train.epochs)):
            train_metrics = self.train_one_epoch(epoch)
            route_metrics: dict[str, dict[str, dict[str, float]]] = {}

            if self.val_query_loader is not None and (epoch + 1) % val_interval == 0:
                route_metrics = self._evaluate_with_dynamic_best_body_hint(
                    epoch,
                    train_metrics,
                    monitor_aggregation,
                    monitor_name,
                )

            if self.scheduler is not None:
                self.scheduler.step()
            self._save_epoch_reports(epoch, train_metrics, route_metrics)

            if is_main_process():
                self.logger.info("=" * 88)
                self.logger.info("Epoch %03d Summary", epoch + 1)
                self.logger.info("train | %s", self._format_train(train_metrics))
                for route in ("body", "face", "body_no_face", "hard_route"):
                    metrics = route_metrics.get(route, {}).get(self._primary_aggregation(), {})
                    self.logger.info("val[%s] | %s", route, self._format_metrics(metrics))
                self.logger.info("=" * 88)

            if (epoch + 1) % int(self.cfg.train.get("save_interval", 1)) == 0:
                self.save(epoch, "latest.pth")

        if is_main_process():
            self.logger.info("=" * 88)
            self.logger.info("MeowID-Base Training Complete")
            for branch in ("body", "face"):
                epoch = self.best_epochs[branch]
                metric = self.best_metrics[branch]
                if epoch is None or metric is None:
                    self.logger.info("best_%s | unavailable", branch)
                else:
                    self.logger.info(
                        "best_%s | epoch=%03d | %s=%.4f",
                        branch,
                        epoch + 1,
                        monitor_name,
                        metric,
                    )
            self.logger.info("=" * 88)

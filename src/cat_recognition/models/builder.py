from __future__ import annotations

from pathlib import Path

import torch

from .arcface import ArcMarginProduct, SubCenterArcMarginProduct
from .backbones import HFDinoV3Backbone, TimmBackbone
from .meowid_base import MeowIDBase
from .pooling import build_pool
from .recognizer import CatRecognizer


def _build_head(model_cfg, num_classes: int | None, with_head: bool):
    if not with_head or not bool(model_cfg.head.get("enabled", True)) or not num_classes:
        return None
    head_name = str(model_cfg.head.get("name", "arcface")).lower()
    head_kwargs = {
        "in_features": int(model_cfg.embedding_dim),
        "out_features": int(num_classes),
        "scale": float(model_cfg.head.get("scale", 64.0)),
        "margin": float(model_cfg.head.get("margin", 0.5)),
        "easy_margin": bool(model_cfg.head.get("easy_margin", False)),
        "ls_eps": float(model_cfg.head.get("ls_eps", 0.0)),
    }
    if head_name == "arcface":
        return ArcMarginProduct(**head_kwargs)
    if head_name in {"subcenter_arcface", "sub_center_arcface"}:
        return SubCenterArcMarginProduct(
            **head_kwargs,
            num_subcenters=int(model_cfg.head.get("num_subcenters", 3)),
        )
    raise ValueError(f"Unsupported head type: {model_cfg.head.get('name')}")


def _build_recognizer(model_cfg, train_image_size: int, head=None) -> CatRecognizer:
    backbone_cfg = model_cfg.backbone
    backend = str(backbone_cfg.get("backend", "timm")).lower()
    if backend == "hf_dinov3":
        backbone = HFDinoV3Backbone(
            pretrained=bool(backbone_cfg.get("pretrained", True)),
            pretrained_ckpt=backbone_cfg.get("pretrained_ckpt"),
            hidden_size=int(backbone_cfg.get("hidden_size", 768)),
            image_size=int(train_image_size),
            patch_size=int(backbone_cfg.get("patch_size", 16)),
            intermediate_size=int(backbone_cfg.get("intermediate_size", 3072)),
            num_hidden_layers=int(backbone_cfg.get("num_hidden_layers", 12)),
            num_attention_heads=int(backbone_cfg.get("num_attention_heads", 12)),
            num_register_tokens=int(backbone_cfg.get("num_register_tokens", 4)),
            hidden_act=str(backbone_cfg.get("hidden_act", "gelu")),
            layer_norm_eps=float(backbone_cfg.get("layer_norm_eps", 1.0e-5)),
            rope_theta=float(backbone_cfg.get("rope_theta", 100.0)),
            query_bias=bool(backbone_cfg.get("query_bias", True)),
            key_bias=bool(backbone_cfg.get("key_bias", False)),
            value_bias=bool(backbone_cfg.get("value_bias", True)),
            proj_bias=bool(backbone_cfg.get("proj_bias", True)),
            mlp_bias=bool(backbone_cfg.get("mlp_bias", True)),
            use_gated_mlp=bool(backbone_cfg.get("use_gated_mlp", False)),
            layerscale_value=float(backbone_cfg.get("layerscale_value", 1.0)),
            drop_path_rate=float(backbone_cfg.get("drop_path_rate", 0.0)),
        )
    else:
        backbone = TimmBackbone(
            name=backbone_cfg.name,
            pretrained=bool(backbone_cfg.get("pretrained", True)),
            pretrained_ckpt=backbone_cfg.get("pretrained_ckpt"),
            drop_path_rate=float(backbone_cfg.get("drop_path_rate", 0.0)),
            drop_rate=float(backbone_cfg.get("drop_rate", 0.0)),
            feature_mode=str(backbone_cfg.get("feature_mode", "auto")),
        )
    pool = build_pool(model_cfg.pool, input_dim=int(backbone.out_features))
    pooled_dim = (
        int(pool.get_output_dim(backbone.out_features))
        if hasattr(pool, "get_output_dim")
        else int(backbone.out_features)
    )
    return CatRecognizer(
        backbone=backbone,
        pool=pool,
        backbone_dim=pooled_dim,
        embedding_dim=int(model_cfg.embedding_dim),
        dropout=float(model_cfg.get("dropout", 0.0)),
        head=head,
        neck_cfg=model_cfg.get("neck", {}),
    )


def _checkpoint_mapping(checkpoint: dict) -> dict[str, int]:
    mapping = checkpoint.get("class_to_idx")
    if mapping is None:
        mapping = checkpoint.get("extra", {}).get("class_to_idx")
    if not mapping:
        raise ValueError("Expert checkpoint does not contain class_to_idx")
    return {str(key): int(value) for key, value in mapping.items()}


def _load_expert_state(expert: CatRecognizer, checkpoint: dict, checkpoint_path: str | Path) -> None:
    expert_state = {
        key: value for key, value in checkpoint["model"].items() if not key.startswith("head.")
    }
    incompatible = expert.load_state_dict(expert_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            f"Failed to load expert {checkpoint_path}: "
            f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
        )


def _load_expert_checkpoint(expert: CatRecognizer, checkpoint_path: str | Path) -> dict:
    if not checkpoint_path:
        raise ValueError("Expert checkpoint path is required")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
    _load_expert_state(expert, checkpoint, checkpoint_path)
    return checkpoint


def _copy_head_rows(destination_head, destination_mapping: dict[str, int], checkpoint: dict) -> int:
    if destination_head is None:
        return 0
    source_weight = checkpoint["model"].get("head.weight")
    if source_weight is None:
        raise ValueError("Expert checkpoint is missing head.weight")
    source_mapping = _checkpoint_mapping(checkpoint)
    if tuple(destination_head.weight.shape[1:]) != tuple(source_weight.shape[1:]):
        raise ValueError(
            f"ArcFace head shape mismatch: destination={tuple(destination_head.weight.shape)} "
            f"source={tuple(source_weight.shape)}"
        )
    copied = 0
    with torch.no_grad():
        for cat_id, destination_index in destination_mapping.items():
            source_index = source_mapping.get(cat_id)
            if source_index is None:
                continue
            destination_head.weight[int(destination_index)].copy_(source_weight[source_index])
            copied += 1
    return copied


def _build_meowid_base(cfg, num_classes, with_head, class_to_idx):
    meowid_cfg = cfg.model.get("meowid", {})
    if not class_to_idx:
        raise ValueError("MeowID-Base requires class_to_idx")

    train_image_size = int(cfg.data.transforms.train.get("image_size", 256))
    body_expert = _build_recognizer(cfg.model, train_image_size=train_image_size, head=None)
    face_expert = _build_recognizer(cfg.model, train_image_size=train_image_size, head=None)
    best_body_hint_expert = _build_recognizer(
        cfg.model,
        train_image_size=train_image_size,
        head=None,
    )

    body_checkpoint = _load_expert_checkpoint(
        body_expert,
        meowid_cfg.get("body_checkpoint"),
    )
    _load_expert_state(
        best_body_hint_expert,
        body_checkpoint,
        meowid_cfg.get("body_checkpoint"),
    )
    face_checkpoint = _load_expert_checkpoint(
        face_expert,
        meowid_cfg.get("face_checkpoint"),
    )
    face_class_to_idx = _checkpoint_mapping(face_checkpoint)

    body_head = _build_head(cfg.model, num_classes=num_classes, with_head=with_head)
    face_head = _build_head(
        cfg.model,
        num_classes=len(face_class_to_idx),
        with_head=with_head,
    )

    lookup_size = max(int(index) for index in class_to_idx.values()) + 1
    face_label_lookup = torch.full((lookup_size,), -1, dtype=torch.long)
    for cat_id, body_index in class_to_idx.items():
        face_index = face_class_to_idx.get(str(cat_id))
        if face_index is not None:
            face_label_lookup[int(body_index)] = int(face_index)

    model = MeowIDBase(
        body_expert=body_expert,
        face_expert=face_expert,
        best_body_hint_expert=best_body_hint_expert,
        body_head=body_head,
        face_head=face_head,
        face_label_lookup=face_label_lookup,
        embedding_dim=int(cfg.model.embedding_dim),
        hint_hidden_dim=int(meowid_cfg.get("hint_hidden_dim", cfg.model.embedding_dim)),
        hint_scale=float(meowid_cfg.get("hint_scale", 0.1)),
        dynamic_best_body_hint=True,
        freeze_body_expert=bool(meowid_cfg.get("freeze_body_expert", False)),
        freeze_face_expert=bool(meowid_cfg.get("freeze_face_expert", False)),
        freeze_body_head=bool(meowid_cfg.get("freeze_body_head", False)),
        freeze_face_head=bool(meowid_cfg.get("freeze_face_head", False)),
    )

    body_rows = _copy_head_rows(model.body_head, class_to_idx, body_checkpoint)
    face_rows = _copy_head_rows(model.face_head, face_class_to_idx, face_checkpoint)
    model._apply_freeze_policy()
    model.initialization_info = {
        "architecture": "MeowID-Base",
        "body_checkpoint": str(meowid_cfg.get("body_checkpoint")),
        "face_checkpoint": str(meowid_cfg.get("face_checkpoint")),
        "body_epoch": int(body_checkpoint.get("epoch", -1)),
        "face_epoch": int(face_checkpoint.get("epoch", -1)),
        "body_head_rows": body_rows,
        "face_head_rows": face_rows,
        "body_num_classes": int(num_classes or 0),
        "face_num_classes": len(face_class_to_idx),
        "face_missing_body_classes": int((face_label_lookup < 0).sum().item()),
        "dynamic_best_body_hint": True,
        "disabled_unused_parameters": list(model.disabled_unused_parameters),
    }
    model.face_class_to_idx = face_class_to_idx
    del body_checkpoint, face_checkpoint
    return model


def build_model(
    cfg,
    num_classes: int | None = None,
    with_head: bool = True,
    class_to_idx: dict[str, int] | None = None,
):
    model_type = str(cfg.model.get("type", "single")).lower()
    if model_type == "meowid_base":
        return _build_meowid_base(cfg, num_classes, with_head, class_to_idx)
    head = _build_head(cfg.model, num_classes=num_classes, with_head=with_head)
    image_size = int(
        cfg.data.transforms.train.get("image_size", cfg.model.backbone.get("image_size", 224))
    )
    return _build_recognizer(cfg.model, train_image_size=image_size, head=head)


def build_meowid_deployment_model(cfg, state_dict: dict[str, torch.Tensor]) -> MeowIDBase:
    """Build MeowID-Base directly from a single-body deployment checkpoint.

    Training builds the two experts from their initialization checkpoints and
    includes large classification heads.  Deployment needs neither dependency:
    the supplied state dict is authoritative and only the embedding graph is
    instantiated here.
    """

    image_size = int(
        cfg.data.transforms.train.get(
            "image_size",
            cfg.model.backbone.get("image_size", 256),
        )
    )
    meowid_cfg = cfg.model.get("meowid", {})
    body_expert = _build_recognizer(cfg.model, train_image_size=image_size, head=None)
    face_expert = _build_recognizer(cfg.model, train_image_size=image_size, head=None)
    face_label_lookup = state_dict.get("face_label_lookup")
    if face_label_lookup is None:
        face_label_lookup = torch.empty(0, dtype=torch.long)

    model = MeowIDBase(
        body_expert=body_expert,
        face_expert=face_expert,
        best_body_hint_expert=None,
        body_head=None,
        face_head=None,
        face_label_lookup=face_label_lookup,
        embedding_dim=int(cfg.model.embedding_dim),
        hint_hidden_dim=int(meowid_cfg.get("hint_hidden_dim", cfg.model.embedding_dim)),
        hint_scale=float(meowid_cfg.get("hint_scale", 0.1)),
        dynamic_best_body_hint=False,
        freeze_body_expert=False,
        freeze_face_expert=False,
    )
    deploy_state = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(("body_head.", "face_head."))
    }
    incompatible = model.load_state_dict(deploy_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Invalid MeowID-Base deployment checkpoint: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    return model

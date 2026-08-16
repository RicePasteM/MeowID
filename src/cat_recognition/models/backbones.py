from __future__ import annotations

from pathlib import Path
import re
import warnings

import timm
import torch
import torch.nn as nn


def _unwrap_checkpoint(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a state dict or a dict containing a state dict.")

    for key in ("state_dict", "model_state_dict", "model"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            checkpoint = value
            break
    return checkpoint


def _strip_prefix_from_state_dict(state_dict: dict, prefix: str) -> dict:
    matched = {key[len(prefix):]: value for key, value in state_dict.items() if key.startswith(prefix)}
    return matched or state_dict


def _prepare_local_state_dict(state_dict: dict) -> dict:
    state_dict = dict(state_dict)
    state_dict = _strip_prefix_from_state_dict(state_dict, "module.")
    state_dict = _strip_prefix_from_state_dict(state_dict, "backbone.model.")
    state_dict = _strip_prefix_from_state_dict(state_dict, "backbone.")
    return state_dict


def _looks_like_hf_dinov3_state_dict(state_dict: dict) -> bool:
    return (
        "embeddings.patch_embeddings.weight" in state_dict
        and "embeddings.cls_token" in state_dict
        and any(key.startswith("layer.0.attention.") for key in state_dict)
    )


def _resolve_model_key(model_state_dict: dict, *candidates: str) -> str | None:
    for candidate in candidates:
        if candidate in model_state_dict:
            return candidate
    return None


def _convert_hf_dinov3_state_dict(state_dict: dict, model: nn.Module) -> dict:
    model_state_dict = model.state_dict()
    converted = {}

    def assign_if_present(target_key: str | None, source_key: str) -> None:
        if target_key is not None and source_key in state_dict:
            converted[target_key] = state_dict[source_key]

    assign_if_present(
        _resolve_model_key(model_state_dict, "cls_token", "embeddings.cls_token"),
        "embeddings.cls_token",
    )
    assign_if_present(
        _resolve_model_key(model_state_dict, "reg_token", "reg_tokens", "register_tokens", "embeddings.register_tokens"),
        "embeddings.register_tokens",
    )
    assign_if_present(
        _resolve_model_key(model_state_dict, "mask_token", "embeddings.mask_token"),
        "embeddings.mask_token",
    )
    assign_if_present(
        _resolve_model_key(model_state_dict, "patch_embed.proj.weight", "embeddings.patch_embeddings.weight"),
        "embeddings.patch_embeddings.weight",
    )
    assign_if_present(
        _resolve_model_key(model_state_dict, "patch_embed.proj.bias", "embeddings.patch_embeddings.bias"),
        "embeddings.patch_embeddings.bias",
    )
    assign_if_present(
        _resolve_model_key(model_state_dict, "norm.weight", "fc_norm.weight"),
        "norm.weight",
    )
    assign_if_present(
        _resolve_model_key(model_state_dict, "norm.bias", "fc_norm.bias"),
        "norm.bias",
    )

    layer_indices = sorted(
        {
            int(match.group(1))
            for key in state_dict.keys()
            if (match := re.match(r"layer\.(\d+)\.", key))
        }
    )

    for idx in layer_indices:
        prefix = f"layer.{idx}"

        q_weight = state_dict.get(f"{prefix}.attention.q_proj.weight")
        k_weight = state_dict.get(f"{prefix}.attention.k_proj.weight")
        v_weight = state_dict.get(f"{prefix}.attention.v_proj.weight")
        if q_weight is not None and k_weight is not None and v_weight is not None:
            converted[f"blocks.{idx}.attn.qkv.weight"] = torch.cat([q_weight, k_weight, v_weight], dim=0)

        q_bias = state_dict.get(f"{prefix}.attention.q_proj.bias")
        k_bias = state_dict.get(f"{prefix}.attention.k_proj.bias")
        v_bias = state_dict.get(f"{prefix}.attention.v_proj.bias")
        if q_bias is not None or k_bias is not None or v_bias is not None:
            reference = q_bias if q_bias is not None else (k_bias if k_bias is not None else v_bias)
            zeros = torch.zeros_like(reference)
            converted[f"blocks.{idx}.attn.qkv.bias"] = torch.cat(
                [
                    q_bias if q_bias is not None else zeros,
                    k_bias if k_bias is not None else zeros,
                    v_bias if v_bias is not None else zeros,
                ],
                dim=0,
            )

        assign_if_present(f"blocks.{idx}.attn.proj.weight", f"{prefix}.attention.o_proj.weight")
        assign_if_present(f"blocks.{idx}.attn.proj.bias", f"{prefix}.attention.o_proj.bias")
        assign_if_present(f"blocks.{idx}.norm1.weight", f"{prefix}.norm1.weight")
        assign_if_present(f"blocks.{idx}.norm1.bias", f"{prefix}.norm1.bias")
        assign_if_present(f"blocks.{idx}.norm2.weight", f"{prefix}.norm2.weight")
        assign_if_present(f"blocks.{idx}.norm2.bias", f"{prefix}.norm2.bias")
        assign_if_present(f"blocks.{idx}.mlp.fc1.weight", f"{prefix}.mlp.up_proj.weight")
        assign_if_present(f"blocks.{idx}.mlp.fc1.bias", f"{prefix}.mlp.up_proj.bias")
        assign_if_present(f"blocks.{idx}.mlp.fc2.weight", f"{prefix}.mlp.down_proj.weight")
        assign_if_present(f"blocks.{idx}.mlp.fc2.bias", f"{prefix}.mlp.down_proj.bias")
        assign_if_present(f"blocks.{idx}.ls1.gamma", f"{prefix}.layer_scale1.lambda1")
        assign_if_present(f"blocks.{idx}.ls2.gamma", f"{prefix}.layer_scale2.lambda1")

        bias_mask_key = f"blocks.{idx}.attn.qkv.bias_mask"
        if bias_mask_key in model_state_dict:
            converted[bias_mask_key] = model_state_dict[bias_mask_key]

    return converted


def _resolve_feature_mode(name: str, feature_mode: str) -> str:
    mode = feature_mode.lower()
    if mode != "auto":
        return mode

    lowered = name.lower()
    if "dinov3" in lowered and "vit" in lowered:
        return "dinov3_tokens"
    return "tensor"


def load_local_pretrained_weights(model: nn.Module, checkpoint_path: str | Path, strict: bool = False) -> None:
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Local pretrained checkpoint not found: {checkpoint_path}")

    if checkpoint_path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "Loading '.safetensors' requires `safetensors`. Install it with `pip install safetensors`."
            ) from exc
        state_dict = load_file(str(checkpoint_path))
    else:
        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        state_dict = _unwrap_checkpoint(checkpoint) if isinstance(checkpoint, dict) else checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    state_dict = _prepare_local_state_dict(state_dict)
    model_state_dict = model.state_dict()
    should_convert_hf_dinov3 = (
        _looks_like_hf_dinov3_state_dict(state_dict)
        and "blocks.0.attn.qkv.weight" in model_state_dict
    )
    if should_convert_hf_dinov3:
        warnings.warn(
            f"Detected Hugging Face DINOv3 checkpoint format, converting keys for timm: {checkpoint_path}",
            stacklevel=2,
        )
        state_dict = _convert_hf_dinov3_state_dict(state_dict, model)

    incompatible = model.load_state_dict(state_dict, strict=strict)

    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    if missing or unexpected:
        missing_preview = missing[:10]
        unexpected_preview = unexpected[:10]
        warnings.warn(
            "Loaded local pretrained weights with strict=False. "
            f"missing_keys={len(missing)} unexpected_keys={len(unexpected)} "
            f"missing_preview={missing_preview} unexpected_preview={unexpected_preview} "
            f"checkpoint={checkpoint_path}",
            stacklevel=2,
        )


class HFDinoV3Backbone(nn.Module):
    def __init__(
        self,
        pretrained: bool = True,
        pretrained_ckpt: str | None = None,
        hidden_size: int = 768,
        image_size: int = 224,
        patch_size: int = 16,
        intermediate_size: int = 3072,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        num_register_tokens: int = 4,
        hidden_act: str = "gelu",
        layer_norm_eps: float = 1.0e-5,
        rope_theta: float = 100.0,
        query_bias: bool = True,
        key_bias: bool = False,
        value_bias: bool = True,
        proj_bias: bool = True,
        mlp_bias: bool = True,
        use_gated_mlp: bool = False,
        layerscale_value: float = 1.0,
        drop_path_rate: float = 0.0,
    ) -> None:
        super().__init__()
        try:
            from transformers import DINOv3ViTConfig, DINOv3ViTModel
        except ImportError as exc:
            raise ImportError(
                "Hugging Face DINOv3 backbone requires `transformers>=4.56.0`."
            ) from exc

        self.config = DINOv3ViTConfig(
            hidden_size=hidden_size,
            image_size=image_size,
            patch_size=patch_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_register_tokens=num_register_tokens,
            hidden_act=hidden_act,
            layer_norm_eps=layer_norm_eps,
            rope_theta=rope_theta,
            query_bias=query_bias,
            key_bias=key_bias,
            value_bias=value_bias,
            proj_bias=proj_bias,
            mlp_bias=mlp_bias,
            use_gated_mlp=use_gated_mlp,
            layerscale_value=layerscale_value,
            drop_path_rate=drop_path_rate,
        )
        self.model = DINOv3ViTModel(self.config)
        self.out_features = int(hidden_size)

        if pretrained and pretrained_ckpt:
            load_local_pretrained_weights(self.model, pretrained_ckpt, strict=True)

    def set_trainable_mode(self, mode: str, unfreeze_last_n_blocks: int = 0) -> str:
        mode = mode.lower()
        if mode not in {"full", "frozen", "partial"}:
            raise ValueError(f"Unsupported backbone trainable mode: {mode}")

        if mode == "full":
            for parameter in self.model.parameters():
                parameter.requires_grad = True
            self.model.train()
            return "full"

        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()

        if mode == "frozen":
            return "frozen"

        blocks = getattr(self.model, "layer", None)
        if blocks is None and hasattr(self.model, "encoder"):
            blocks = getattr(self.model.encoder, "layer", None)
        if blocks is None or len(blocks) == 0 or unfreeze_last_n_blocks <= 0:
            for parameter in self.model.parameters():
                parameter.requires_grad = True
            self.model.train()
            return "full"

        trainable_blocks = list(blocks)[-min(int(unfreeze_last_n_blocks), len(blocks)) :]
        for block in trainable_blocks:
            block.train()
            for parameter in block.parameters():
                parameter.requires_grad = True

        norm = getattr(self.model, "norm", None)
        if isinstance(norm, nn.Module):
            norm.train()
            for parameter in norm.parameters():
                parameter.requires_grad = True

        return f"partial_last_{len(trainable_blocks)}"

    def forward(self, x):
        outputs = self.model(pixel_values=x, return_dict=True)
        sequence = outputs.last_hidden_state
        num_register_tokens = int(getattr(self.config, "num_register_tokens", 0))
        prefix_tokens = 1 + num_register_tokens
        return {
            "sequence": sequence,
            "cls_token": sequence[:, 0],
            "register_tokens": sequence[:, 1:prefix_tokens],
            "patch_tokens": sequence[:, prefix_tokens:],
            "num_prefix_tokens": prefix_tokens,
            "num_register_tokens": num_register_tokens,
        }


class TimmBackbone(nn.Module):
    def __init__(
        self,
        name: str,
        pretrained: bool = True,
        pretrained_ckpt: str | None = None,
        drop_path_rate: float = 0.0,
        drop_rate: float = 0.0,
        feature_mode: str = "auto",
    ) -> None:
        super().__init__()
        self.name = name
        self.feature_mode = _resolve_feature_mode(name, feature_mode)
        use_timm_pretrained = bool(pretrained and not pretrained_ckpt)
        kwargs = {
            "pretrained": use_timm_pretrained,
            "num_classes": 0,
            "drop_path_rate": drop_path_rate,
            "drop_rate": drop_rate,
        }

        try:
            self.model = timm.create_model(name, global_pool="", **kwargs)
        except TypeError:
            self.model = timm.create_model(name, **kwargs)

        self.out_features = getattr(self.model, "num_features", None)
        if self.out_features is None:
            raise ValueError(f"Backbone does not expose num_features: {name}")

        if pretrained_ckpt:
            load_local_pretrained_weights(self.model, pretrained_ckpt)

    def _split_token_sequence(self, sequence: torch.Tensor) -> dict[str, torch.Tensor | int]:
        if sequence.ndim != 3:
            raise ValueError(f"Expected token sequence with shape (B, N, C), got: {tuple(sequence.shape)}")

        num_prefix_tokens = int(getattr(self.model, "num_prefix_tokens", 1))
        num_reg_tokens = int(getattr(self.model, "num_reg_tokens", max(num_prefix_tokens - 1, 0)))
        cls_token = sequence[:, 0]
        register_tokens = sequence[:, 1:num_prefix_tokens] if num_prefix_tokens > 1 else sequence[:, :0]
        patch_tokens = sequence[:, num_prefix_tokens:]
        return {
            "sequence": sequence,
            "cls_token": cls_token,
            "register_tokens": register_tokens,
            "patch_tokens": patch_tokens,
            "num_prefix_tokens": num_prefix_tokens,
            "num_register_tokens": num_reg_tokens,
        }

    def _format_dinov3_features(self, features):
        if isinstance(features, torch.Tensor):
            return self._split_token_sequence(features)

        if not isinstance(features, dict):
            raise TypeError(f"Unsupported DINOv3 feature output type: {type(features)!r}")

        def _first_tensor(*keys):
            for key in keys:
                value = features.get(key)
                if isinstance(value, torch.Tensor):
                    return value
            return None

        sequence = None
        for key in ("x_prenorm", "sequence", "x"):
            value = features.get(key)
            if isinstance(value, torch.Tensor):
                sequence = value
                break

        cls_token = _first_tensor("x_norm_clstoken", "cls_token")
        patch_tokens = _first_tensor("x_norm_patchtokens", "patch_tokens")
        register_tokens = _first_tensor("x_norm_regtokens", "register_tokens")

        if sequence is None:
            if patch_tokens is None or cls_token is None:
                raise ValueError("DINOv3 features did not contain enough token tensors to build pooled features.")
            if cls_token.ndim == 2:
                cls_prefix = cls_token.unsqueeze(1)
            else:
                cls_prefix = cls_token
            prefix_tokens = [cls_prefix]
            if register_tokens is not None:
                prefix_tokens.append(register_tokens)
            sequence = torch.cat(prefix_tokens + [patch_tokens], dim=1)

        token_features = self._split_token_sequence(sequence)
        if isinstance(cls_token, torch.Tensor):
            token_features["cls_token"] = cls_token.squeeze(1) if cls_token.ndim == 3 else cls_token
        if isinstance(patch_tokens, torch.Tensor):
            token_features["patch_tokens"] = patch_tokens
        if isinstance(register_tokens, torch.Tensor):
            token_features["register_tokens"] = register_tokens
        return token_features

    def set_trainable_mode(self, mode: str, unfreeze_last_n_blocks: int = 0) -> str:
        mode = mode.lower()
        if mode not in {"full", "frozen", "partial"}:
            raise ValueError(f"Unsupported backbone trainable mode: {mode}")

        if mode == "full":
            for parameter in self.model.parameters():
                parameter.requires_grad = True
            self.model.train()
            return "full"

        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.model.eval()

        if mode == "frozen":
            return "frozen"

        blocks = getattr(self.model, "blocks", None)
        if blocks is None or len(blocks) == 0 or unfreeze_last_n_blocks <= 0:
            for parameter in self.model.parameters():
                parameter.requires_grad = True
            self.model.train()
            return "full"

        trainable_blocks = list(blocks)[-min(int(unfreeze_last_n_blocks), len(blocks)) :]
        for block in trainable_blocks:
            block.train()
            for parameter in block.parameters():
                parameter.requires_grad = True

        for attr in ("norm", "fc_norm"):
            module = getattr(self.model, attr, None)
            if isinstance(module, nn.Module):
                module.train()
                for parameter in module.parameters():
                    parameter.requires_grad = True

        for attr in ("cls_token", "reg_token", "register_tokens", "pos_embed"):
            parameter = getattr(self.model, attr, None)
            if isinstance(parameter, nn.Parameter):
                parameter.requires_grad = True

        return f"partial_last_{len(trainable_blocks)}"

    def forward(self, x):
        if hasattr(self.model, "forward_features"):
            features = self.model.forward_features(x)
        else:
            features = self.model(x)

        if self.feature_mode == "dinov3_tokens":
            return self._format_dinov3_features(features)
        return features

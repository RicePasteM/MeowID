from __future__ import annotations

from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F


class MeowIDBase(nn.Module):
    """Final MeowID-Base whole-cat/face identification architecture.

    The body expert is supervised on every whole-cat image.  The face expert
    and its face-space hint modules are supervised only where an aligned face
    exists. During training, the face branch uses a frozen, validation-selected
    best-body snapshot as its whole-cat hint. The snapshot is discarded after
    its matching best body weights are installed for export, leaving one
    whole-cat encoder for fallback and hint inference.

    Retrieval intentionally keeps two spaces:

    * body route: the raw normalized body-expert embedding;
    * face route: ``normalize(face + gated_body_hint)``.

    They must be evaluated against their matching galleries rather than mixed
    in a single cosine-similarity index.
    """

    supports_expert_routes = True

    def __init__(
        self,
        body_expert: nn.Module,
        face_expert: nn.Module,
        best_body_hint_expert: nn.Module | None,
        body_head: nn.Module | None,
        face_head: nn.Module | None,
        face_label_lookup: torch.Tensor,
        embedding_dim: int,
        hint_hidden_dim: int = 512,
        hint_scale: float = 0.1,
        dynamic_best_body_hint: bool = True,
        freeze_body_expert: bool = False,
        freeze_face_expert: bool = False,
        freeze_body_head: bool = False,
        freeze_face_head: bool = False,
    ) -> None:
        super().__init__()
        self.body_expert = body_expert
        self.face_expert = face_expert
        self.best_body_hint_expert = best_body_hint_expert
        self.body_head = body_head
        self.face_head = face_head
        self.embedding_dim = int(embedding_dim)
        self.hint_scale = float(hint_scale)
        self.dynamic_best_body_hint = bool(dynamic_best_body_hint)
        if self.dynamic_best_body_hint and self.best_body_hint_expert is None:
            raise ValueError("MeowID-Base training requires best_body_hint_expert")
        if not self.dynamic_best_body_hint and self.best_body_hint_expert is not None:
            raise ValueError("best_body_hint_expert requires dynamic_best_body_hint=True")
        self.freeze_body_expert = bool(freeze_body_expert)
        self.freeze_face_expert = bool(freeze_face_expert)
        self.freeze_body_head = bool(freeze_body_head)
        self.freeze_face_head = bool(freeze_face_head)
        self.register_buffer("face_label_lookup", face_label_lookup.long(), persistent=True)
        self.disabled_unused_parameters: list[str] = []

        # This projector belongs only to the face hint path.  It never replaces
        # the body embedding returned by the body route.
        self.body_projector = nn.Linear(self.embedding_dim, self.embedding_dim, bias=False)
        self.hint_adapter = nn.Sequential(
            nn.Linear(self.embedding_dim, int(hint_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hint_hidden_dim), self.embedding_dim),
        )
        self.hint_gate = nn.Sequential(
            nn.Linear(self.embedding_dim * 4, int(hint_hidden_dim)),
            nn.GELU(),
            nn.Linear(int(hint_hidden_dim), 1),
        )
        self.reset_fusion_parameters()
        self._apply_freeze_policy()

    def reset_fusion_parameters(self) -> None:
        with torch.no_grad():
            self.body_projector.weight.copy_(torch.eye(self.embedding_dim))
        final_adapter = self.hint_adapter[-1]
        nn.init.zeros_(final_adapter.weight)
        nn.init.zeros_(final_adapter.bias)
        final_gate = self.hint_gate[-1]
        nn.init.zeros_(final_gate.weight)
        nn.init.constant_(final_gate.bias, -2.0)

    @staticmethod
    def _set_requires_grad(module: nn.Module | None, enabled: bool) -> None:
        if module is None:
            return
        for parameter in module.parameters():
            parameter.requires_grad = enabled

    def _apply_freeze_policy(self) -> None:
        self._set_requires_grad(self.body_expert, not self.freeze_body_expert)
        self._set_requires_grad(self.face_expert, not self.freeze_face_expert)
        # This is a validation-selected teacher, never an independently
        # optimized expert. It exists only while training and is removed from
        # the deployment checkpoint.
        self._set_requires_grad(self.best_body_hint_expert, False)
        self._set_requires_grad(self.body_head, not self.freeze_body_head)
        self._set_requires_grad(self.face_head, not self.freeze_face_head)
        if self.freeze_body_expert:
            self.body_expert.eval()
        if self.freeze_face_expert:
            self.face_expert.eval()
        if self.best_body_hint_expert is not None:
            self.best_body_hint_expert.eval()
        if self.freeze_body_head and self.body_head is not None:
            self.body_head.eval()
        if self.freeze_face_head and self.face_head is not None:
            self.face_head.eval()
        self._disable_structurally_unused_parameters()

    def _disable_structurally_unused_parameters(self) -> None:
        """Freeze masked-modeling tokens that image retrieval never consumes.

        Hugging Face DINOv3 exposes a trainable ``mask_token`` for pretraining,
        but its ordinary image forward does not use that parameter. Leaving it
        trainable makes DDP wait for a gradient that can never exist. Freezing
        just that token avoids the expensive global unused-parameter scan.
        """

        disabled = []
        experts = (
            ("body_expert", self.body_expert),
            ("face_expert", self.face_expert),
            ("best_body_hint_expert", self.best_body_hint_expert),
        )
        for prefix, expert in experts:
            if expert is None:
                continue
            for name, parameter in expert.named_parameters():
                if name.endswith("mask_token"):
                    parameter.requires_grad = False
                    disabled.append(f"{prefix}.{name}")
        self.disabled_unused_parameters = disabled

    def train(self, mode: bool = True):
        super().train(mode)
        self._apply_freeze_policy()
        return self

    @staticmethod
    def _encode_expert(expert: nn.Module, images: torch.Tensor, frozen: bool) -> torch.Tensor:
        context = torch.no_grad() if frozen else nullcontext()
        # A routed subset can contain a single face even when the original
        # batch is larger. BatchNorm cannot estimate batch statistics from one
        # sample, so only its running-stat behavior is used for that forward;
        # affine parameters and the rest of the expert remain trainable.
        batch_norms = []
        training_states = []
        if images.shape[0] == 1 and expert.training:
            batch_norms = [
                module
                for module in expert.modules()
                if isinstance(module, nn.modules.batchnorm._BatchNorm)
            ]
            training_states = [module.training for module in batch_norms]
            for module in batch_norms:
                module.eval()
        try:
            with context:
                return expert.encode(images)
        finally:
            for module, was_training in zip(batch_norms, training_states):
                module.train(was_training)

    @staticmethod
    def _zero_parameter_dependency(
        reference: torch.Tensor,
        modules: tuple[nn.Module | None, ...],
    ) -> torch.Tensor:
        dependency = reference.new_zeros(())
        for module in modules:
            if module is None:
                continue
            for parameter in module.parameters():
                if parameter.requires_grad:
                    dependency = dependency + parameter.reshape(-1)[0] * 0.0
        return dependency

    def encode_body(self, images: torch.Tensor) -> torch.Tensor:
        return self._encode_expert(self.body_expert, images, self.freeze_body_expert)

    def encode_deployment(
        self,
        images: torch.Tensor,
        face_images: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return both hard-route embeddings from one deployable graph.

        ``images`` and ``face_images`` have the same batch size.  Missing faces
        are represented by any placeholder image and are ignored by the caller.
        Keeping routing outside the graph avoids data-dependent control flow and
        makes the exact same graph exportable to ONNX and TensorRT.  Importantly,
        the body embedding used by the fallback route is also the hint consumed
        by the face route, so the deployment graph contains one body encoder.
        """

        body_embeddings = self.encode_body(images)
        face_embeddings = self._encode_expert(
            self.face_expert,
            face_images,
            self.freeze_face_expert,
        )
        body_for_hint = body_embeddings.detach()
        projected_body = F.normalize(
            self.body_projector(body_for_hint),
            p=2,
            dim=1,
        )
        face_embeddings = face_embeddings.to(dtype=projected_body.dtype)
        gate_inputs = torch.cat(
            [
                face_embeddings,
                projected_body,
                torch.abs(face_embeddings - projected_body),
                face_embeddings * projected_body,
            ],
            dim=1,
        )
        gates = torch.sigmoid(self.hint_gate(gate_inputs))
        hints = self.hint_adapter(body_for_hint).to(dtype=projected_body.dtype)
        face_route_embeddings = F.normalize(
            face_embeddings + self.hint_scale * gates * hints,
            p=2,
            dim=1,
        )
        return body_embeddings, face_route_embeddings

    @torch.no_grad()
    def refresh_best_body_hint(self) -> None:
        """Make the frozen hint teacher an exact snapshot of the body expert."""

        if not self.dynamic_best_body_hint or self.best_body_hint_expert is None:
            raise RuntimeError("Dynamic best-body hint is not enabled")
        self.best_body_hint_expert.load_state_dict(self.body_expert.state_dict(), strict=True)
        self.best_body_hint_expert.eval()
        self._set_requires_grad(self.best_body_hint_expert, False)

    @torch.no_grad()
    def best_body_hint_matches_body(self) -> bool:
        if self.best_body_hint_expert is None:
            return not self.dynamic_best_body_hint
        body_state = self.body_expert.state_dict()
        hint_state = self.best_body_hint_expert.state_dict()
        return body_state.keys() == hint_state.keys() and all(
            torch.equal(body_state[key], hint_state[key])
            for key in body_state
        )

    @torch.no_grad()
    def load_body_branch_state(self, checkpoint_state: dict[str, torch.Tensor]) -> None:
        """Load only the deployable body branch from a full training checkpoint."""

        for module_name in ("body_expert", "body_head"):
            module = getattr(self, module_name)
            if module is None:
                continue
            prefix = f"{module_name}."
            module_state = {
                key[len(prefix) :]: value
                for key, value in checkpoint_state.items()
                if key.startswith(prefix)
            }
            module.load_state_dict(module_state, strict=True)

    def prepare_for_single_body_deployment(self) -> None:
        """Reuse body_expert as the hint and discard the training-only teacher."""

        self.best_body_hint_expert = None
        self.dynamic_best_body_hint = False

    def encode_routes(
        self,
        images: torch.Tensor,
        face_images: torch.Tensor | None = None,
        face_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        batch_size = images.shape[0]
        if face_mask is None:
            face_mask = torch.zeros(batch_size, dtype=torch.bool, device=images.device)
        else:
            face_mask = face_mask.to(device=images.device, dtype=torch.bool).reshape(-1)
        if face_mask.numel() != batch_size:
            raise ValueError(f"face_mask length {face_mask.numel()} != batch size {batch_size}")
        has_face = bool(face_mask.any())
        if has_face and face_images is None:
            raise ValueError("face_images are required when face_mask contains True")

        body_embeddings = self.encode_body(images)

        # If a rare training batch has no faces, attach every face-route
        # parameter to the body loss with a zero-valued edge.  DDP can then run
        # with find_unused_parameters=False without a graph traversal.
        if not has_face and self.training:
            zero_dependency = self._zero_parameter_dependency(
                body_embeddings,
                (
                    self.face_expert,
                    self.face_head,
                    self.body_projector,
                    self.hint_adapter,
                    self.hint_gate,
                ),
            )
            body_embeddings = body_embeddings + zero_dependency

        route_embeddings = body_embeddings.clone()
        face_embeddings = body_embeddings.new_zeros((batch_size, self.embedding_dim))
        face_route_embeddings = body_embeddings.new_zeros((batch_size, self.embedding_dim))
        hint_body_embeddings = body_embeddings.new_zeros((batch_size, self.embedding_dim))
        gates = body_embeddings.new_zeros((batch_size,))
        hint_norms = body_embeddings.new_zeros((batch_size,))

        if has_face:
            selected_faces = self._encode_expert(
                self.face_expert,
                face_images[face_mask],
                self.freeze_face_expert,
            )
            if self.dynamic_best_body_hint:
                selected_body_for_hint = self._encode_expert(
                    self.best_body_hint_expert,
                    images[face_mask],
                    True,
                )
            else:
                selected_body_for_hint = body_embeddings[face_mask].detach()
            projected_body = F.normalize(
                self.body_projector(selected_body_for_hint),
                p=2,
                dim=1,
            )
            selected_faces = selected_faces.to(dtype=projected_body.dtype)
            gate_inputs = torch.cat(
                [
                    selected_faces,
                    projected_body,
                    torch.abs(selected_faces - projected_body),
                    selected_faces * projected_body,
                ],
                dim=1,
            )
            selected_gates = torch.sigmoid(self.hint_gate(gate_inputs)).squeeze(1)
            selected_gates = selected_gates.to(dtype=projected_body.dtype)
            selected_hints = self.hint_adapter(selected_body_for_hint)
            selected_hints = selected_hints.to(dtype=projected_body.dtype)
            selected_fused = F.normalize(
                selected_faces
                + self.hint_scale * selected_gates.unsqueeze(1) * selected_hints,
                p=2,
                dim=1,
            )

            route_embeddings[face_mask] = selected_fused
            face_embeddings[face_mask] = selected_faces
            face_route_embeddings[face_mask] = selected_fused
            hint_body_embeddings[face_mask] = projected_body
            gates[face_mask] = selected_gates
            hint_norms[face_mask] = selected_hints.norm(p=2, dim=1)

        return {
            # ``embeddings`` is retained for generic inference compatibility,
            # but route-aware evaluation uses the two explicit fields below.
            "embeddings": route_embeddings,
            "route_embeddings": route_embeddings,
            "body_embeddings": body_embeddings,
            "face_embeddings": face_embeddings,
            "face_route_embeddings": face_route_embeddings,
            "hint_body_embeddings": hint_body_embeddings,
            "face_mask": face_mask,
            "hint_gates": gates,
            "hint_norms": hint_norms,
        }

    def encode(
        self,
        images: torch.Tensor,
        face_images: torch.Tensor | None = None,
        face_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.encode_routes(images, face_images=face_images, face_mask=face_mask)[
            "route_embeddings"
        ]

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
        face_images: torch.Tensor | None = None,
        face_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        outputs = self.encode_routes(images, face_images=face_images, face_mask=face_mask)
        if labels is None:
            return outputs

        if self.body_head is not None:
            outputs["body_logits"] = self.body_head(outputs["body_embeddings"], labels)

        selected_mask = outputs["face_mask"]
        if self.face_head is not None and bool(selected_mask.any()):
            selected_body_labels = labels[selected_mask].long()
            if selected_body_labels.min() < 0 or selected_body_labels.max() >= self.face_label_lookup.numel():
                raise ValueError("Body labels are outside face_label_lookup")
            face_labels = self.face_label_lookup[selected_body_labels]
            if bool((face_labels < 0).any()):
                raise ValueError("A face sample has no corresponding face-expert class")
            outputs["face_labels"] = face_labels
            outputs["face_logits"] = self.face_head(
                outputs["face_route_embeddings"][selected_mask],
                face_labels,
            )
        return outputs

    def get_optimizer_param_groups(self, cfg) -> list[dict]:
        base_lr = float(cfg.optimizer.lr)
        meowid_cfg = cfg.model.get("meowid", {})
        groups: list[dict] = []

        def add(name: str, module: nn.Module | None, multiplier: float) -> None:
            if module is None:
                return
            params = [parameter for parameter in module.parameters() if parameter.requires_grad]
            if params:
                groups.append({"name": name, "params": params, "lr": base_lr * multiplier})

        add(
            "body_expert",
            self.body_expert,
            float(meowid_cfg.get("body_expert_lr_multiplier", 0.02)),
        )
        add(
            "face_expert",
            self.face_expert,
            float(meowid_cfg.get("face_expert_lr_multiplier", 0.01)),
        )
        add(
            "body_head",
            self.body_head,
            float(meowid_cfg.get("body_head_lr_multiplier", 0.1)),
        )
        add(
            "face_head",
            self.face_head,
            float(meowid_cfg.get("face_head_lr_multiplier", 0.1)),
        )
        add(
            "body_projector",
            self.body_projector,
            float(meowid_cfg.get("body_projector_lr_multiplier", 1.0)),
        )
        add(
            "hint_adapter",
            self.hint_adapter,
            float(meowid_cfg.get("hint_lr_multiplier", 1.0)),
        )
        add(
            "hint_gate",
            self.hint_gate,
            float(meowid_cfg.get("hint_lr_multiplier", 1.0)),
        )
        return groups

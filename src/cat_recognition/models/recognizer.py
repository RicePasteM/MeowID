from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CatRecognizer(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        pool: nn.Module,
        backbone_dim: int,
        embedding_dim: int = 512,
        dropout: float = 0.0,
        head: nn.Module | None = None,
        neck_cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        neck_cfg = dict(neck_cfg or {})
        self.neck_name = str(neck_cfg.get("name", "linear_bn")).lower()
        self.embedding_hidden = None
        self.embedding_activation = nn.Identity()
        if self.neck_name == "linear_bn":
            self.embedding = nn.Linear(backbone_dim, embedding_dim, bias=True)
        elif self.neck_name == "mlp_bn":
            hidden_dim = int(neck_cfg.get("hidden_dim", backbone_dim))
            self.embedding_hidden = nn.Linear(backbone_dim, hidden_dim, bias=True)
            self.embedding_activation = nn.GELU()
            self.embedding = nn.Linear(hidden_dim, embedding_dim, bias=True)
        else:
            raise ValueError(f"Unsupported neck type: {neck_cfg.get('name')}")
        self.bn = nn.BatchNorm1d(embedding_dim)
        self.head = head
        self.embedding_dim = embedding_dim

    def get_optimizer_param_groups(self, cfg) -> list[dict]:
        base_lr = float(cfg.optimizer.lr)
        groups = []
        seen: set[int] = set()

        def add_group(name: str, module: nn.Module | None, lr_multiplier: float) -> None:
            if module is None:
                return
            params = []
            for parameter in module.parameters():
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                params.append(parameter)
            if params:
                groups.append(
                    {
                        "name": name,
                        "params": params,
                        "lr": base_lr * float(lr_multiplier),
                    }
                )

        add_group("backbone", self.backbone, float(cfg.optimizer.get("backbone_lr_multiplier", 1.0)))
        add_group("pool", self.pool, float(cfg.optimizer.get("pool_lr_multiplier", 1.0)))
        embedding_modules = [self.embedding, self.bn]
        if self.embedding_hidden is not None:
            embedding_modules.insert(0, self.embedding_hidden)
        add_group("embedding", nn.ModuleList(embedding_modules), float(cfg.optimizer.get("embedding_lr_multiplier", 1.0)))
        add_group("head", self.head, float(cfg.optimizer.get("head_lr_multiplier", 1.0)))

        leftovers = [parameter for parameter in self.parameters() if id(parameter) not in seen]
        if leftovers:
            groups.append(
                {
                    "name": "other",
                    "params": leftovers,
                    "lr": base_lr,
                }
            )
        return groups

    def encode(self, images: torch.Tensor) -> torch.Tensor:
        features = self.backbone(images)
        pooled = self.pool(features)
        pooled = self.dropout(pooled)
        if self.embedding_hidden is not None:
            pooled = self.embedding_hidden(pooled)
            pooled = self.embedding_activation(pooled)
        embeddings = self.embedding(pooled)
        embeddings = self.bn(embeddings)
        return F.normalize(embeddings, p=2, dim=1)

    def forward(self, images: torch.Tensor, labels: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        embeddings = self.encode(images)
        outputs = {"embeddings": embeddings}
        if self.head is not None and labels is not None:
            outputs["logits"] = self.head(embeddings, labels)
        return outputs

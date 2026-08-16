from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml


class ConfigNode(dict):
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        super().__init__()
        for key, value in (data or {}).items():
            self[key] = value

    def __getattr__(self, item: str) -> Any:
        try:
            return self[item]
        except KeyError as exc:
            raise AttributeError(item) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, self._wrap(value))

    @classmethod
    def _wrap(cls, value: Any) -> Any:
        if isinstance(value, ConfigNode):
            return value
        if isinstance(value, dict):
            return cls(value)
        if isinstance(value, list):
            return [cls._wrap(item) for item in value]
        return value

    def to_dict(self) -> dict[str, Any]:
        return _to_plain_dict(self)


def _to_plain_dict(value: Any) -> Any:
    if isinstance(value, ConfigNode):
        return {key: _to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {key: _to_plain_dict(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain_dict(item) for item in value]
    return value


def deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in update.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return data


def _load_config_dict(path: Path) -> dict[str, Any]:
    raw_cfg = _load_yaml_dict(path)
    bases = raw_cfg.pop("base", None)
    if bases is None:
        return raw_cfg

    if not isinstance(bases, list):
        bases = [bases]

    merged: dict[str, Any] = {}
    for base in bases:
        base_path = (path.parent / base).resolve()
        merged = deep_merge(merged, _load_config_dict(base_path))
    return deep_merge(merged, raw_cfg)


def load_config(path: str | Path) -> ConfigNode:
    path = Path(path).resolve()
    return ConfigNode(_load_config_dict(path))


def set_by_dotted_path(cfg: dict[str, Any], dotted_key: str, value: Any) -> None:
    node = cfg
    parts = dotted_key.split(".")
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def apply_overrides(cfg: ConfigNode, overrides: Iterable[str] | None) -> ConfigNode:
    if not overrides:
        return cfg

    plain = cfg.to_dict()
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override format: {item}")
        key, raw_value = item.split("=", 1)
        set_by_dotted_path(plain, key, yaml.safe_load(raw_value))
    return ConfigNode(plain)


def dump_config(cfg: ConfigNode, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(cfg.to_dict(), handle, sort_keys=False, allow_unicode=False)

"""YAML config loading with dotted-key overrides.

Intentionally small. Avoids pulling Hydra/OmegaConf as required deps so
the package stays light. Drop in your own if you want fancier merging.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path}: top-level YAML must be a mapping (got {type(data).__name__})")
    return data


def apply_overrides(cfg: dict[str, Any], overrides: Iterable[str]) -> dict[str, Any]:
    """Apply ``"a.b.c=value"`` style overrides in place. Values are YAML-parsed.

    Example
    -------
    >>> apply_overrides(cfg, ["fwi.epochs=50", "model.dh=[10.0, 10.0]"])
    """
    for ov in overrides:
        if "=" not in ov:
            raise ValueError(f"override must be 'key=value'; got {ov!r}")
        key, _, raw = ov.partition("=")
        value = yaml.safe_load(raw)
        _set_dotted(cfg, key.strip().split("."), value)
    return cfg


def _set_dotted(d: dict[str, Any], keys: list[str], value: Any) -> None:
    cur = d
    for k in keys[:-1]:
        nxt = cur.get(k)
        if not isinstance(nxt, dict):
            nxt = {}
            cur[k] = nxt
        cur = nxt
    cur[keys[-1]] = value


__all__ = ["load_config", "apply_overrides"]

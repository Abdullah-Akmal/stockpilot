"""Configuration loading and shared paths for StockPilot."""
from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the config: explicit path > $STOCKPILOT_CONFIG > configs/config.yaml."""
    path = path or os.environ.get("STOCKPILOT_CONFIG") or DEFAULT_CONFIG
    with open(resolve(path), "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(rel: str | Path) -> Path:
    """Resolve a repo-relative path from the config."""
    p = Path(rel)
    return p if p.is_absolute() else ROOT / p


def merged(cfg: dict, overrides: dict | None) -> dict:
    """Deep-merge overrides into a copy of cfg."""
    out = copy.deepcopy(cfg)
    for k, v in (overrides or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = merged(out[k], v)
        else:
            out[k] = v
    return out

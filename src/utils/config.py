"""Lightweight YAML config loader.

Usage:
    from src.utils.config import load
    aoi = load("aoi")  # loads configs/aoi.yml
"""
from __future__ import annotations
from pathlib import Path
from functools import lru_cache
from typing import Any
import yaml

from . import paths


@lru_cache(maxsize=None)
def load(name: str) -> dict[str, Any]:
    """Load a YAML config from configs/<name>.yml."""
    p = paths.CONFIGS / f"{name}.yml"
    if not p.exists():
        raise FileNotFoundError(f"Config {p} not found")
    with p.open() as f:
        return yaml.safe_load(f)


def seed(component: str = "master_seed") -> int:
    """Return the deterministic seed for a component (master_seed by default)."""
    cfg = load("seed")
    if component == "master_seed":
        return int(cfg["master_seed"])
    if component in cfg.get("splits", {}):
        return int(cfg["splits"][component])
    if component in cfg.get("sampling", {}):
        return int(cfg["sampling"][component])
    raise KeyError(f"Unknown seed component: {component}")

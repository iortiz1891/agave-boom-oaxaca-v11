"""Canonical project paths. Single source of truth.

Other modules should import these constants instead of constructing paths inline.
"""
from __future__ import annotations
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[2]

CONFIGS = ROOT / "configs"
DATA = ROOT / "data"
DATA_RAW = DATA / "raw"
DATA_INTERIM = DATA / "interim"
DATA_PROCESSED = DATA / "processed"
DATA_REFERENCE = DATA / "reference"
DATA_EXTERNAL = DATA / "external"
NOTEBOOKS = ROOT / "notebooks"
SRC = ROOT / "src"
GEE = ROOT / "gee"
FIGURES = ROOT / "figures"
PAPER = ROOT / "paper"
DASHBOARD = ROOT / "dashboard"
DOCS = ROOT / "docs"
REPORTS = ROOT / "reports"
LOGS = ROOT / "logs"

# Common config files
AOI_YML = CONFIGS / "aoi.yml"
DATES_YML = CONFIGS / "dates.yml"
SEED_YML = CONFIGS / "seed.yml"
MODEL_YML = CONFIGS / "model.yml"

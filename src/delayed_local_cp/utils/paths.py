"""Project and data locations (inside the container by default)."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(os.environ.get("DALOCP_ROOT", Path(__file__).resolve().parents[3]))
RUNS_DIR = Path(os.environ.get("DALOCP_RUNS", PROJECT_ROOT / "runs"))
CONFIGS_DIR = PROJECT_ROOT / "configs"
REPORTS_DIR = PROJECT_ROOT / "reports"
RESULTS_DIR = PROJECT_ROOT / "results"
PAPER_DIR = PROJECT_ROOT / "paper"

DATA_RAW = Path(os.environ.get("DALOCP_DATA_RAW", PROJECT_ROOT / "datasets" / "raw"))
DATA_PROCESSED = Path(os.environ.get("DALOCP_DATA_PROCESSED", PROJECT_ROOT / "datasets" / "processed"))
DATA_MANIFESTS = Path(os.environ.get("DALOCP_DATA_MANIFESTS", PROJECT_ROOT / "datasets" / "manifests"))

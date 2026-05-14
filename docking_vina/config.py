# src/docking_vina/config.py
# -*- coding: utf-8 -*-

import os
import sys
import yaml
from typing import Any, Dict

from ipz_core.config_loader import ConfigLoader


CONFIG_PATH = "config.yaml"

DEFAULT_CONFIG = {
    "docking": {
        "known_compounds": "data/known_compounds.csv",
        "candidates_file": "data/candidates.csv",
        "grids_file": "docking_grids.json",
        "results_file": "output/docking_results.csv",
        "exhaustiveness": 8,
        "n_poses": 1,
    },
    "project": {"data_dir": "data"},
}


def read_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        sys.exit("[CRITICAL]: config.yaml not found.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    out = {"docking": dict(DEFAULT_CONFIG["docking"])}
    out["docking"].update(raw.get("docking", {}) or {})
    out["project"] = raw.get("project", {}) or {}
    return out


def load_data_dir(config_path: str, cfg: Dict[str, Any]) -> str:
    # prefer ConfigLoader, fallback na cfg
    try:
        loader = ConfigLoader(config_path)
        project_cfg = loader.project()
        return project_cfg.data_dir
    except Exception:
        return cfg.get("project", {}).get("data_dir", DEFAULT_CONFIG["project"]["data_dir"])


def compute_cpu(cfg_dock: Dict[str, Any]) -> int:
    n_poses = cfg_dock.get("n_poses", os.cpu_count() or 1)
    try:
        return max(1, int(n_poses))
    except Exception:
        return max(1, int(os.cpu_count() or 1))

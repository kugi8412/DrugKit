# -*- coding: utf-8 -*-

import os
import sys
from typing import Any, Dict

import yaml

from pocket_logic.config_loader import ConfigLoader


def read_yaml(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        sys.exit("[CRITICAL]: config.yaml not found.")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_section(raw: Dict[str, Any], section: str, defaults: Dict[str, Any]) -> Dict[str, Any]:
    out = {section: dict(defaults)}
    out[section].update(raw.get(section, {}) or {})
    out["project"] = raw.get("project", {}) or {}
    return out


def load_data_dir(config_path: str, cfg: Dict[str, Any], default_data_dir: str = "data") -> str:
    try:
        loader = ConfigLoader(config_path)
        project_cfg = loader.project()
        return project_cfg.data_dir
    except Exception:
        return cfg.get("project", {}).get("data_dir", default_data_dir)


def compute_cpu(cfg_section: Dict[str, Any], key: str = "n_cpu") -> int:
    value = cfg_section.get(key, os.cpu_count() or 1)
    try:
        return max(1, int(value))
    except Exception:
        return max(1, int(os.cpu_count() or 1))

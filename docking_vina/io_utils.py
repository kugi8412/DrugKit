# src/docking_vina/io_utils.py
# -*- coding: utf-8 -*-

import os
import json
from typing import Any, Dict, Optional


def ensure_dir(directory: str) -> None:
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def sanitize_key(key: str) -> str:
    return key.replace("/", "_").replace("\\", "_")


def load_grids(grids_file: str, logger) -> Optional[Dict[str, Any]]:
    if not os.path.exists(grids_file):
        logger.error(f"Missing grids file: {grids_file}")
        return None
    try:
        with open(grids_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Can't load JSON grids: {grids_file} ({e})")
        return None


def get_receptor_path(target_key: str, data_dir: str) -> Optional[str]:
    safe = sanitize_key(target_key)
    possibilities = [
        f"output/{target_key}.pdbqt",
        f"output/{safe}.pdbqt",
        f"data/{target_key}.pdbqt",
        f"data/{safe}.pdbqt",
        f"{target_key}.pdbqt",
        f"{safe}.pdbqt",
        os.path.join(data_dir, f"{target_key}.pdbqt"),
        os.path.join(data_dir, f"{safe}.pdbqt"),
    ]

    for p in possibilities:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return os.path.abspath(p)
    return None


def build_receptor_map(grids: Dict[str, Any], data_dir: str, logger) -> Dict[str, str]:
    rec_map: Dict[str, str] = {}
    for target_key in grids.keys():
        path = get_receptor_path(target_key, data_dir)
        if path:
            rec_map[target_key] = path
            logger.info(f"Receptor {target_key} -> {path}")
        else:
            logger.warning(f"No PDBQT file for {target_key}!")
    return rec_map

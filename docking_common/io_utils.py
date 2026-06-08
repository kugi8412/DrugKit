# -*- coding: utf-8 -*-

import json
import os
from typing import Any, Dict, Optional


def ensure_dir(directory: str) -> None:
    if directory and not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def safe_filename(value: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in str(value))


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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict, Optional

from docking_common.io_utils import ensure_dir, load_grids
from docking_common.receptor_io import build_receptor_map, get_receptor_path, sanitize_key

__all__ = [
    "ensure_dir",
    "load_grids",
    "sanitize_key",
    "get_receptor_path",
    "build_receptor_map",
]


def get_receptor_path_pdbqt(target_key: str, data_dir: str) -> Optional[str]:
    return get_receptor_path(target_key, data_dir, extensions=(".pdbqt",))


def build_receptor_map_pdbqt(grids: Dict[str, Any], data_dir: str, logger) -> Dict[str, str]:
    return build_receptor_map(grids, data_dir, logger, extensions=(".pdbqt",))

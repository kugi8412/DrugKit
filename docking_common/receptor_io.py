#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Any, Dict, Optional, Tuple


def sanitize_key(key: str) -> str:
    return key.replace("/", "_").replace("\\", "_")


def _search_paths(target_key: str, data_dir: str, ext: str) -> Tuple[str, ...]:
    safe = sanitize_key(target_key)
    return (
        f"output/{target_key}{ext}",
        f"output/{safe}{ext}",
        f"data/{target_key}{ext}",
        f"data/{safe}{ext}",
        f"{target_key}{ext}",
        f"{safe}{ext}",
        os.path.join(data_dir, f"{target_key}{ext}"),
        os.path.join(data_dir, f"{safe}{ext}"),
    )


def get_receptor_path(
    target_key: str,
    data_dir: str,
    extensions: Tuple[str, ...] = (".pdbqt", ".pdb"),
) -> Optional[str]:
    found: Dict[str, str] = {}

    for ext in extensions:
        for path in _search_paths(target_key, data_dir, ext):
            if os.path.exists(path) and os.path.getsize(path) > 0:
                found[ext] = os.path.abspath(path)
                break

    if ".pdbqt" in found:
        return found[".pdbqt"]
    if ".pdb" in found:
        return found[".pdb"]
    for ext in extensions:
        if ext in found:
            return found[ext]
    return None


def build_receptor_map(
    grids: Dict[str, Any],
    data_dir: str,
    logger,
    extensions: Tuple[str, ...] = (".pdbqt", ".pdb"),
) -> Dict[str, str]:
    rec_map: Dict[str, str] = {}
    for target_key in grids.keys():
        path = get_receptor_path(target_key, data_dir, extensions=extensions)
        if path:
            rec_map[target_key] = path
            logger.info(f"Receptor {target_key} -> {path}")
        else:
            logger.warning(f"No receptor file for {target_key}!")
    return rec_map

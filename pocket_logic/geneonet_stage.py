# pocket_logic/geneonet_stage.py
# -*- coding: utf-8 -*-

import os
import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import requests

from p2rank_stage import validate_quality
from geometry_utils import calculate_box_from_radius, check_proximity_to_chain
from merge_utils import is_duplicate


def find_geneonet_file(search_dir: str,
                       keywords: List[str]) -> Optional[str]:
    if not os.path.exists(search_dir):
        return None

    for f in os.listdir(search_dir):
        if f.endswith(".csv"):
            if any(k.lower() in f.lower() for k in keywords):
                return os.path.join(search_dir, f)
    return None


def parse_geneonet_csv(filepath: str) -> List[dict]:
    pockets: List[dict] = []
    df = pd.read_csv(filepath)
    df.columns = [c.strip() for c in df.columns]
    if "Score" in df.columns:
        df = df.sort_values(by="Score", ascending=False)

    for idx, row in df.iterrows():
        try:
            c_str = str(row["Center"]).replace("[", "").replace("]", "").replace('"', "").strip()
            center = [float(x) for x in c_str.split(",")]
            pockets.append({
                "id_suffix": f"rank{row.get('Pocket', idx + 1)}",
                "center": center,
                "radius": float(row["Radius"]),
                "score": float(row["Score"])
            })
        except Exception:
            continue

    return pockets


def is_official_pdb_entry(pdb_id: str, timeout: float = 5.0) -> bool:
    if not pdb_id:
        return False

    url = f"https://data.rcsb.org/rest/v1/core/entry/{pdb_id}"
    try:
        r = requests.get(url, timeout=timeout)
        return r.status_code == 200
    except requests.RequestException:
        return False


def should_use_geneonet(pdb_id: str, gn_file: Optional[str]) -> bool:
    if not gn_file:
        return False
    return is_official_pdb_entry(pdb_id)


def add_geneonet_pockets(chain_pockets: List[dict],
                         structure,
                         chain_id: str,
                         gn_global: List[dict],
                         geneonet_top_n: int,
                         buffer_size: float,
                         overlap_threshold: float,
                         p2rank_global: List[dict],
                         logger: logging.Logger) -> int:
    added_gn = 0
    stats_gn = {"Chain": 0, "Overlap": 0, "LowConf": 0}

    for gn_p in gn_global:
        if added_gn >= geneonet_top_n:
            break

        if not check_proximity_to_chain(gn_p["center"], structure, chain_id, buffer_size * 2):
            stats_gn["Chain"] += 1
            continue

        if is_duplicate(gn_p["center"], chain_pockets, overlap_threshold):
            stats_gn["Overlap"] += 1
            continue

        val = validate_quality(gn_p["center"], p2rank_global, chain_id)
        if "Low" in val["status"]:
            stats_gn["LowConf"] += 1
            continue

        size = calculate_box_from_radius(gn_p["center"], gn_p["radius"], buffer_size)
        chain_pockets.append({
            "id": f"{chain_id}_geneonet_{gn_p['id_suffix']}",
            "center": [round(x, 3) for x in gn_p["center"]],
            "size": [round(x, 3) for x in size],
            "source": "GeneoNet",
            "validation": val
        })
        added_gn += 1

    logger.info(f"    + Added {added_gn}/{geneonet_top_n} GeneoNet pockets.")
    if added_gn < geneonet_top_n:
        logger.info(
            f"      (Skipped: {stats_gn['Chain']} Chain, {stats_gn['Overlap']} Overlap, {stats_gn['LowConf']} LowConf)"
        )
    return added_gn

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pocket_logic/p2rank_stage.py

import os
import shutil
import subprocess
import logging
from typing import List, Dict, Set, Optional, Any

import numpy as np
import pandas as pd

from pocket_logic.geometry_utils import get_residue_coords, calculate_centered_box
from pocket_logic.merge_utils import is_duplicate


def run_or_load_p2rank(pdb_path: str,
                       output_dir: str,
                       p2rank_exec: Optional[str],
                       logger: logging.Logger) -> List[dict]:
    pockets: List[dict] = []
    pdb_name = os.path.basename(pdb_path)
    csv_file = os.path.join(output_dir, f"{pdb_name}_predictions.csv")

    if not os.path.exists(csv_file):
        if not p2rank_exec or (not os.path.exists(p2rank_exec) and not shutil.which(p2rank_exec)):
            return pockets

        logger.info(f"  [P2Rank] Running on {pdb_name}...")
        cmd = [p2rank_exec, "predict", "-f", pdb_path, "-o", output_dir]
        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
            csv_file = os.path.join(output_dir, f"{pdb_name}_predictions.csv")
        except Exception:
            return pockets

    if csv_file and os.path.exists(csv_file):
        try:
            df = pd.read_csv(csv_file, skipinitialspace=True)
            df.columns = [c.strip() for c in df.columns]
            for _, row in df.iterrows():
                res_raw = str(row.get("residue_ids", ""))
                res_ids: List[int] = []
                chains: Set[str] = set()
                for token in res_raw.split():
                    if "_" in token:
                        c, r = token.split("_")
                        chains.add(c)
                        if r.isdigit():
                            res_ids.append(int(r))
                    elif token.isdigit():
                        res_ids.append(int(token))

                pockets.append({
                    "rank": int(row["rank"]),
                    "score": float(row["score"]),
                    "center": [float(row["center_x"]), float(row["center_y"]), float(row["center_z"])],
                    "residues": set(res_ids),
                    "chains": chains
                })
        except Exception:
            pass

    return pockets


def validate_quality(target_center,
                     p2rank_pockets,
                     target_chain=None) -> Dict[str, Any]:
    if not p2rank_pockets:
        return {"status": "Unknown", "dist": 999.9}

    t_c = np.array(target_center)
    best_dist = 999.9
    for p in p2rank_pockets:
        if target_chain and p["chains"] and target_chain not in p["chains"]:
            continue
        dist = np.linalg.norm(t_c - np.array(p["center"]))
        if dist < best_dist:
            best_dist = dist

    if best_dist < 5.0:
        status = "High Confidence"
    elif best_dist < 10.0:
        status = "Medium Confidence"
    else:
        status = "Low Confidence"

    return {"status": status, "distance": round(best_dist, 2)}


def add_p2rank_pockets(chain_pockets: List[dict],
                       structure,
                       chain_id: str,
                       p2rank_global: List[dict],
                       p2rank_top_n: int,
                       buffer_size: float,
                       overlap_threshold: float,
                       logger: logging.Logger) -> int:
    added_p2 = 0
    stats_p2 = {"Chain": 0, "Overlap": 0}

    for p2 in p2rank_global:
        if added_p2 >= p2rank_top_n:
            break

        if p2["chains"] and chain_id not in p2["chains"]:
            stats_p2["Chain"] += 1
            continue

        if is_duplicate(p2["center"], chain_pockets, overlap_threshold):
            stats_p2["Overlap"] += 1
            continue

        coords = get_residue_coords(structure, chain_id, p2["residues"])
        if coords.size > 0:
            center, size = calculate_centered_box(coords, buffer_size)
        else:
            center, size = p2["center"], [30.0, 30.0, 30.0]

        chain_pockets.append({
            "id": f"{chain_id}_p2rank_r{p2['rank']}",
            "center": [round(x, 3) for x in center],
            "size": [round(x, 3) for x in size],
            "source": "P2Rank",
            "validation": {"status": "High (Source)", "dist": 0.0}
        })
        added_p2 += 1

    logger.info(f"    + Added {added_p2}/{p2rank_top_n} P2Rank pockets.")
    if added_p2 < p2rank_top_n:
        logger.info(f"      (Skipped: {stats_p2['Chain']} Chain, {stats_p2['Overlap']} Overlap)")
    return added_p2

# src/docking_vina/workers.py
# -*- coding: utf-8 -*-

import os
import numpy as np
from typing import Any, Dict, List, Tuple

from docking_vina.ligands import prepare_ligand
from docking_vina.vina_engine import run_vina_scoring
from docking_vina.io_utils import sanitize_key


def worker_dock_known(
    item: Dict[str, Any],
    rec_map: Dict[str, str],
    grids: Dict[str, Any],
    base_exhaustiveness: int,
) -> List[Dict[str, Any]]:
    smi = item.get("SMILES")
    name = item.get("Name", "Unknown")
    if not smi:
        return []

    prep = prepare_ligand(smi, name)
    if not prep:
        return []
    lig_pdbqt, _ = prep

    results: List[Dict[str, Any]] = []
    for target_key, pockets in grids.items():
        rec_path = rec_map.get(target_key)
        if not rec_path:
            continue

        for p in pockets:
            score, _ = run_vina_scoring(
                lig_pdbqt, rec_path, p["center"], p["size"], base_exhaustiveness
            )
            if not np.isnan(score):
                results.append(
                    {
                        "Name": name,
                        "SMILES": smi,
                        "Original_Target": item.get("Target"),
                        "Docked_Target": target_key,
                        "Pocket_ID": p["id"],
                        "Energy": score,
                    }
                )
    return results


def worker_dock_candidate(
    item: Dict[str, Any],
    rec_map: Dict[str, str],
    grids: Dict[str, Any],
    thresholds: Dict[str, float],
    output_poses_dir: str,
    base_exhaustiveness: int,
) -> Tuple[List[Dict[str, Any]], bool]:
    smi = item.get("SMILES")
    name = item.get("Name")
    if not smi:
        return [], False

    prep = prepare_ligand(smi, name)
    if not prep:
        return [], False
    lig_pdbqt, _ = prep

    results: List[Dict[str, Any]] = []
    hit_found = False

    for target_key, pockets in grids.items():
        rec_path = rec_map.get(target_key)
        if not rec_path:
            continue

        thresh = thresholds.get(target_key, -7.0)

        for p in pockets:
            score, pose = run_vina_scoring(
                lig_pdbqt, rec_path, p["center"], p["size"], base_exhaustiveness
            )

            if np.isnan(score):
                continue

            is_hit = score < thresh
            if is_hit:
                hit_found = True
                safe_name = "".join([c if c.isalnum() else "_" for c in (name or "Ligand")])
                safe_target = sanitize_key(target_key)
                fname = f"{safe_name}_{safe_target}_{p['id']}.pdbqt"
                try:
                    with open(os.path.join(output_poses_dir, fname), "w", encoding="utf-8") as f:
                        f.write(pose)
                except Exception:
                    pass

            results.append(
                {
                    "Name": name,
                    "SMILES": smi,
                    "Target": target_key,
                    "Pocket_ID": p["id"],
                    "Energy": score,
                    "Beat_Baseline": is_hit,
                    "Baseline_Value": thresh,
                    "Cluster_ID": item.get("Cluster_ID", ""),
                }
            )

    return results, hit_found

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from typing import Any, Dict, List, Tuple

import numpy as np

from docking_common.ligands import prepare_ligand
from docking_common.receptor_io import sanitize_key
from docking_smina.smina_engine import run_smina_scoring


def worker_dock_known(
    item: Dict[str, Any],
    rec_map: Dict[str, str],
    grids: Dict[str, Any],
    base_exhaustiveness: int,
    smina_exe: str,
    num_modes: int,
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

        for pocket in pockets:
            score, _ = run_smina_scoring(
                rec_path,
                lig_pdbqt,
                pocket["center"],
                pocket["size"],
                base_exhaustiveness,
                smina_exe=smina_exe,
                num_modes=num_modes,
            )
            if not np.isnan(score):
                results.append(
                    {
                        "Name": name,
                        "SMILES": smi,
                        "Original_Target": item.get("Target"),
                        "Docked_Target": target_key,
                        "Pocket_ID": pocket["id"],
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
    smina_exe: str,
    num_modes: int,
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

        for pocket in pockets:
            score, pose = run_smina_scoring(
                rec_path,
                lig_pdbqt,
                pocket["center"],
                pocket["size"],
                base_exhaustiveness,
                smina_exe=smina_exe,
                num_modes=num_modes,
            )

            if np.isnan(score):
                continue

            is_hit = score < thresh
            if is_hit:
                hit_found = True
                safe_name = "".join(c if c.isalnum() else "_" for c in (name or "Ligand"))
                safe_target = sanitize_key(target_key)
                fname = f"{safe_name}_{safe_target}_{pocket['id']}.pdbqt"
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
                    "Pocket_ID": pocket["id"],
                    "Energy": score,
                    "Beat_Baseline": is_hit,
                    "Baseline_Value": thresh,
                    "Cluster_ID": item.get("Cluster_ID", ""),
                }
            )

    return results, hit_found

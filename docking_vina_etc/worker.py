#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import numpy as np
from typing import Dict, Any, List, Optional

from docking_vina_etc.ligand_prep import prepare_ligand
from docking_vina_etc.vina_runner import run_vina_scoring
from docking_vina_etc.fs_utils import safe_filename


def worker_dock_one_ligand_all_targets(
    member: Dict[str, Any],
    pocket_map_for_cluster: Dict[str, Optional[str]],
    rec_map: Dict[str, str],
    grids: Dict[str, List[dict]],
    baselines: Dict[str, float],
    base_exhaustiveness: int,
    poses_dir: str,
) -> List[Dict[str, Any]]:
    """
    Return he list of rows (long format):
      Name,SMILES,Target,Pocket_ID,Energy,Beat_Baseline,Baseline_Value,Cluster_ID
    """
    smi = member.get("SMILES")
    name = member.get("Name")
    cid = member.get("Cluster_ID")

    if not smi or not name or not cid:
        return []

    if not pocket_map_for_cluster:
        return []

    prep = prepare_ligand(str(smi), str(name))
    if not prep:
        return []

    lig_pdbqt, _ = prep
    rows: List[Dict[str, Any]] = []

    for target_id, pocket_id in pocket_map_for_cluster.items():
        if not pocket_id:
            continue

        target_id = str(target_id)
        pocket_id = str(pocket_id)

        rec_path = rec_map.get(target_id)
        if not rec_path:
            continue

        grid_list = grids.get(target_id, [])
        grid_data = next((p for p in grid_list if str(p.get("id")) == pocket_id), None)
        if not grid_data:
            continue

        score, pose = run_vina_scoring(
            pdbqt_ligand=lig_pdbqt,
            receptor_path=rec_path,
            center=grid_data["center"],
            size=grid_data["size"],
            base_exhaustiveness=base_exhaustiveness,
        )

        if np.isnan(score):
            continue

        baseline = float(baselines.get(target_id, -7.0))
        beat = bool(score < baseline)

        if beat and poses_dir:
            try:
                safe_name = safe_filename(name)
                fname = f"{safe_name}_{target_id}_{pocket_id}.pdbqt"
                with open(os.path.join(poses_dir, fname), "w", encoding="utf-8") as f:
                    f.write(pose)
            except Exception:
                pass

        rows.append({
            "Name": str(name),
            "SMILES": str(smi),
            "Target": target_id,
            "Pocket_ID": pocket_id,
            "Energy": float(score),
            "Beat_Baseline": beat,
            "Baseline_Value": baseline,
            "Cluster_ID": str(cid),
        })

    return rows

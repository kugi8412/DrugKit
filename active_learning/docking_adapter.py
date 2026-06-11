#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Adapter that docks a list of compounds via the docking_smina pipeline.
"""

import os
import tempfile
from typing import Any, Dict, List

import pandas as pd

from docking_smina.pipeline import run_candidates


def dock_compounds(records: List[Dict[str, Any]], rec_map: Dict[str, str],
                   grids: Dict[str, Any], cfg: Dict[str, Any],
                   output_poses_dir: str, logger) -> pd.DataFrame:
    """Dock `records` (each with Name, SMILES) against all receptors in `grids`.

    Returns a DataFrame of per-pocket results (Name, SMILES, Target, Pocket_ID,
    Energy, ...). Empty DataFrame if nothing docked.
    """
    if not records:
        return pd.DataFrame()

    os.makedirs(output_poses_dir, exist_ok=True)
    thresholds = {k: float(cfg["default_baseline"]) for k in grids.keys()}

    rows = []
    for r in records:
        rows.append({
            "Name": r.get("Name"),
            "SMILES": r.get("SMILES"),
            "Target": r.get("Target", ""),
            "Cluster_ID": r.get("Cluster_ID", "Unknown"),
        })

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", prefix="al_dock_", delete=False)
    try:
        pd.DataFrame(rows).to_csv(tmp.name, index=False)
        tmp.close()
        results, _ = run_candidates(
            candidates_file=tmp.name,
            rec_map=rec_map,
            grids=grids,
            thresholds=thresholds,
            output_poses_dir=output_poses_dir,
            n_cpu=int(cfg.get("n_cpu", 2)),
            base_exhaustiveness=int(cfg["exhaustiveness"]),
            smina_exe=cfg["smina_exe"],
            num_modes=int(cfg["num_modes"]),
            logger=logger,
        )
    finally:
        if os.path.exists(tmp.name):
            os.remove(tmp.name)

    return pd.DataFrame(results) if results else pd.DataFrame()

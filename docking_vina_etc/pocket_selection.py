# -*- coding: utf-8 -*-

import os
import pandas as pd
from typing import Dict, Optional, Tuple, List


def infer_targets_from_reps(df_reps: pd.DataFrame) -> List[str]:
    if "Target" not in df_reps.columns:
        return []
    return sorted([str(x) for x in df_reps["Target"].dropna().unique().tolist()])


def identify_best_pockets_and_baselines(
    reps_file: str,
    known_res_file: str,
    default_baseline: float,
    logger,
) -> Tuple[Dict[str, Dict[str, Optional[str]]], Dict[str, float]]:
    """
    Return:
      cluster_pockets: Cluster_ID -> { target_id -> best_pocket_id }
      baselines: target_id -> baseline_energy
    """
    logger.info("Selecting best pockets per cluster (from reps results) and baselines...")

    if not os.path.exists(reps_file):
        logger.error(f"Missing reps_file: {reps_file}")
        return {}, {}

    df_reps = pd.read_csv(reps_file)
    df_reps.columns = [c.strip() for c in df_reps.columns]

    targets = infer_targets_from_reps(df_reps)
    if not targets:
        logger.error("No targets found in reps_file (missing/empty 'Target' column).")
        return {}, {}

    baselines = {t: float(default_baseline) for t in targets}

    if known_res_file and os.path.exists(known_res_file):
        df_k = pd.read_csv(known_res_file)
        df_k.columns = [c.strip() for c in df_k.columns]

        docked_col = "Docked_Target" if "Docked_Target" in df_k.columns else ("Target" if "Target" in df_k.columns else None)
        if docked_col and "Energy" in df_k.columns:
            for t in targets:
                sub = df_k[df_k[docked_col].astype(str) == str(t)]
                if not sub.empty:
                    baselines[t] = float(sub["Energy"].min())

    logger.info(f"Baselines resolved for {len(baselines)} targets.")

    cluster_pockets: Dict[str, Dict[str, Optional[str]]] = {}
    required = {"Cluster_ID", "Target", "Pocket_ID", "Energy"}
    if not required.issubset(set(df_reps.columns)):
        logger.error("reps_file must contain: Cluster_ID, Target, Pocket_ID, Energy")
        return {}, baselines

    for cid in df_reps["Cluster_ID"].dropna().unique():
        cdf = df_reps[df_reps["Cluster_ID"] == cid]
        best_map: Dict[str, Optional[str]] = {}

        for t in targets:
            tdf = cdf[cdf["Target"].astype(str) == str(t)]
            if tdf.empty:
                best_map[t] = None
                continue
            best_row = tdf.sort_values("Energy", ascending=True).iloc[0]
            best_map[t] = str(best_row["Pocket_ID"]) if pd.notna(best_row["Pocket_ID"]) else None

        if any(v is not None for v in best_map.values()):
            cluster_pockets[str(cid)] = best_map

    logger.info(f"Best pockets mapped for {len(cluster_pockets)} clusters.")
    return cluster_pockets, baselines

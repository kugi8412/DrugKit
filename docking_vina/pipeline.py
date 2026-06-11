#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# src/docking_vina/pipeline.py
# -*- coding: utf-8 -*-

import sys
import pandas as pd
from typing import Any, Dict, List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

from docking_vina.workers import worker_dock_known, worker_dock_candidate


def run_baseline(
    known_file: str,
    output_results: str,
    rec_map: Dict[str, str],
    grids: Dict[str, Any],
    n_cpu: int,
    base_exhaustiveness: int,
    logger,
) -> Dict[str, float]:
    logger.info("=== Step 1: Baseline & Cross-Docking ===")

    thresholds: Dict[str, float] = {k: -7.0 for k in grids.keys()}

    if not known_file or not pd.io.common.file_exists(known_file):
        logger.warning(f"No file {known_file}.")
        return thresholds

    df_known = pd.read_csv(known_file)
    df_known.columns = [c.strip() for c in df_known.columns]
    known_items = df_known.to_dict("records")
    logger.info(f"Docking {len(known_items)} known compounds...")

    all_known_res: List[Dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=n_cpu) as executor:
        futures = [
            executor.submit(worker_dock_known, i, rec_map, grids, base_exhaustiveness)
            for i in known_items
        ]
        for f in as_completed(futures):
            all_known_res.extend(f.result())

    if not all_known_res:
        logger.warning("No baseline ressult (Vina error?). Use default.")
        return thresholds

    df_k = pd.DataFrame(all_known_res)
    df_k.to_csv(output_results.replace(".csv", "_known_compounds.csv"), index=False)

    mins = df_k.groupby("Docked_Target")["Energy"].min()
    logger.info(f"New baselines:\n{mins}")

    for target_key, val in mins.items():
        try:
            thresholds[str(target_key)] = float(val)
        except Exception:
            pass

    return thresholds


def run_candidates(
    candidates_file: str,
    rec_map: Dict[str, str],
    grids: Dict[str, Any],
    thresholds: Dict[str, float],
    output_poses_dir: str,
    n_cpu: int,
    base_exhaustiveness: int,
    logger,
) -> Tuple[List[Dict[str, Any]], int]:
    logger.info(f"=== Step 2: Docking candidates (baseline: {thresholds}) ===")

    if not candidates_file or not pd.io.common.file_exists(candidates_file):
        logger.error(f"No file {candidates_file}")
        return [], 0

    df_cand = pd.read_csv(candidates_file)
    df_cand.columns = [c.strip() for c in df_cand.columns]
    cand_items = df_cand.to_dict("records")
    logger.info(f"Candidates number: {len(cand_items)}")

    final_results: List[Dict[str, Any]] = []
    hits_count = 0
    processed = 0

    with ProcessPoolExecutor(max_workers=n_cpu) as executor:
        futures = {
            executor.submit(
                worker_dock_candidate,
                i,
                rec_map,
                grids,
                thresholds,
                output_poses_dir,
                base_exhaustiveness,
            ): i.get("Name")
            for i in cand_items
        }

        for f in as_completed(futures):
            res, is_hit = f.result()
            final_results.extend(res)
            if is_hit:
                hits_count += 1
            processed += 1
            if processed % 10 == 0:
                print(
                    f"Progres: {processed}/{len(cand_items)} | Hits: {hits_count}    ",
                    end="\r",
                )

    print("")
    return final_results, hits_count


def save_results_and_summary(
    final_results: List[Dict[str, Any]],
    output_results: str,
    targets: List[str],
    logger,
) -> None:
    if not final_results:
        logger.warning("No result for candidates.")
        return

    df_res = pd.DataFrame(final_results)
    df_res.to_csv(output_results, index=False)
    logger.info(f"Results saved: {output_results}")

    try:
        pivot = (
            df_res.groupby(["Name", "SMILES", "Target"])["Energy"]
            .min()
            .unstack("Target")
            .reset_index()
        )
        pivot.columns = [f"Energy_{c}" if c in targets else c for c in pivot.columns]
        summary_file = output_results.replace(".csv", "_summary_matrix.csv")
        pivot.to_csv(summary_file, index=False)
        logger.info(f"The summary matrix saved: {summary_file}")
    except Exception:
        pass



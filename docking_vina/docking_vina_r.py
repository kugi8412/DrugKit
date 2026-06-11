#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import os

from pocket_logic.logging_utils import setup_logging

from docking_common.config_utils import compute_cpu, load_data_dir, read_yaml
from docking_common.io_utils import ensure_dir, load_grids
from docking_vina.config import CONFIG_PATH, merge_config
from docking_vina.io_utils import build_receptor_map_pdbqt as build_receptor_map
from docking_vina.pipeline import run_baseline, run_candidates, save_results_and_summary


def main() -> None:
    raw = read_yaml(CONFIG_PATH)
    cfg = merge_config(raw)

    logger = setup_logging(log_dir="logs", log_file="docking_vina.log")
    logger.info("--- START Docking Pipeline (Vina) ---")

    data_dir = load_data_dir(CONFIG_PATH, cfg)

    cfg_dock = cfg["docking"]
    grids_file = cfg_dock.get("grids_file", "docking_grids.json")
    output_results = cfg_dock.get("results_file", "output/docking_results.csv")
    known_file = cfg_dock.get("known_compounds")
    candidates_file = cfg_dock.get("candidates_file")

    base_exhaustiveness = int(cfg_dock.get("exhaustiveness", 8))
    n_cpu = compute_cpu(cfg_dock, key="n_poses")

    output_dir = os.path.dirname(output_results)
    output_poses_dir = os.path.join(output_dir, "top_poses")
    ensure_dir(output_dir)
    ensure_dir(output_poses_dir)

    grids = load_grids(grids_file, logger)
    if not grids:
        return

    rec_map = build_receptor_map(grids, data_dir, logger)
    if not rec_map:
        logger.error("No receptors!")
        sys.exit(1)

    thresholds = run_baseline(
        known_file=known_file,
        output_results=output_results,
        rec_map=rec_map,
        grids=grids,
        n_cpu=n_cpu,
        base_exhaustiveness=base_exhaustiveness,
        logger=logger,
    )

    final_results, hits_count = run_candidates(
        candidates_file=candidates_file,
        rec_map=rec_map,
        grids=grids,
        thresholds=thresholds,
        output_poses_dir=output_poses_dir,
        n_cpu=n_cpu,
        base_exhaustiveness=base_exhaustiveness,
        logger=logger,
    )

    save_results_and_summary(
        final_results=final_results,
        output_results=output_results,
        targets=list(rec_map.keys()),
        logger=logger,
    )

    if final_results:
        logger.info(f"Founded {hits_count} hits. Strukturs in {output_poses_dir}")


if __name__ == "__main__":
    main()

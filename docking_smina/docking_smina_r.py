# -*- coding: utf-8 -*-

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from pocket_logic.logging_utils import setup_logging

from docking_common.config_utils import compute_cpu, load_data_dir, read_yaml
from docking_common.io_utils import ensure_dir, load_grids
from docking_common.receptor_io import build_receptor_map
from docking_smina.config import CONFIG_PATH, merge_config
from docking_smina.pipeline import run_baseline, run_candidates, save_results_and_summary


def main() -> None:
    raw = read_yaml(CONFIG_PATH)
    cfg = merge_config(raw)

    logger = setup_logging(log_dir="logs", log_file="docking_smina.log")
    logger.info("--- START Docking Pipeline (Smina) ---")

    data_dir = load_data_dir(CONFIG_PATH, cfg)
    cfg_dock = cfg["docking_smina"]

    grids_file = cfg_dock.get("grids_file", "docking_grids.json")
    output_results = cfg_dock.get("results_file", "output/docking_smina_results.csv")
    known_file = cfg_dock.get("known_compounds")
    candidates_file = cfg_dock.get("candidates_file")

    base_exhaustiveness = int(cfg_dock.get("exhaustiveness", 16))
    num_modes = int(cfg_dock.get("num_modes", 1))
    smina_exe = cfg_dock.get("smina_exe", "smina")
    default_baseline = float(cfg_dock.get("default_baseline", -7.0))
    n_cpu = compute_cpu(cfg_dock)

    output_dir = os.path.dirname(output_results)
    output_poses_dir = os.path.join(output_dir, "smina_top_poses")
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
        smina_exe=smina_exe,
        num_modes=num_modes,
        default_baseline=default_baseline,
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
        smina_exe=smina_exe,
        num_modes=num_modes,
        logger=logger,
    )

    save_results_and_summary(
        final_results=final_results,
        output_results=output_results,
        targets=list(rec_map.keys()),
        logger=logger,
    )

    if final_results:
        logger.info(f"Found {hits_count} hits. Structures in {output_poses_dir}")


if __name__ == "__main__":
    main()

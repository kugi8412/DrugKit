# -*- coding: utf-8 -*-

import os
import pandas as pd

from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, Any, List

from ipz_core.config_loader import ConfigLoader
from ipz_core.logging_utils import setup_logging

from docking_vina_etc.fs_utils import ensure_dir
from docking_vina_etc.grids_io import load_grids_json
from docking_vina_etc.receptor_io import get_receptor_path
from docking_vina_etc.pocket_selection import identify_best_pockets_and_baselines
from docking_vina_etc.selectivity import build_target_to_protein_map, compute_selectivity_min_for_target
from docking_vina_etc.worker import worker_dock_one_ligand_all_targets


def main() -> None:
    cfg_path = "config.yaml"
    if not os.path.exists(cfg_path) and os.path.exists(f"../{cfg_path}"):
        cfg_path = f"../{cfg_path}"

    logger = setup_logging(log_dir="logs", log_file="docking_vina_etc.log")

    loader = ConfigLoader(cfg_path)
    project_cfg = loader.project()
    structures = loader.structures()

    docking_cfg = (loader.raw or {}).get("docking", {})
    data_dir = project_cfg.data_dir

    grids_file = docking_cfg.get("grids_file", "docking_grids.json")
    reps_file = docking_cfg.get("results_reps_file", "output/docking_results.csv")
    known_file = docking_cfg.get("results_known_file", "output/docking_results_known_compounds.csv")
    members_file = docking_cfg.get("cluster_members_file", "data/cluster_members.csv")

    energies_file = docking_cfg.get("results_energies_file", "output/docking_energies_etc.csv")
    final_file = docking_cfg.get("results_etc_file", "output/docking_results_etc.csv")

    poses_dir = os.path.join(os.path.dirname(final_file), "etc_poses")

    base_exhaustiveness = int(docking_cfg.get("exhaustiveness_etc", 16))
    n_cpu = max(1, int(docking_cfg.get("n_cpu", os.cpu_count() or 1)))
    default_baseline = float(docking_cfg.get("default_baseline", -7.0))

    logger.info("--- START Docking Expansion (multi-protein, multi-conformation) ---")

    ensure_dir(os.path.dirname(final_file))
    ensure_dir(os.path.dirname(energies_file))
    ensure_dir(poses_dir)

    cluster_pockets, baselines = identify_best_pockets_and_baselines(
        reps_file=reps_file,
        known_res_file=known_file,
        default_baseline=default_baseline,
        logger=logger,
    )
    if not cluster_pockets:
        logger.error("No cluster pocket mapping. Stop.")
        return

    if not os.path.exists(members_file):
        logger.error(f"Missing cluster_members_file: {members_file}")
        return

    df_members = pd.read_csv(members_file)
    df_members.columns = [c.strip() for c in df_members.columns]
    if "Cluster_ID" not in df_members.columns:
        logger.error("cluster_members_file must contain 'Cluster_ID'.")
        return

    targets_df = df_members[df_members["Cluster_ID"].astype(str).isin(set(cluster_pockets.keys()))].copy()
    logger.info(f"Ligands to expand: {len(targets_df)}")

    if not os.path.exists(grids_file):
        logger.error(f"Missing grids_file: {grids_file}")
        return

    grids = load_grids_json(grids_file)

    rec_map: Dict[str, str] = {}
    for target_id in grids.keys():
        p = get_receptor_path(str(target_id), data_dir=data_dir)
        if p:
            rec_map[str(target_id)] = p

    if not rec_map:
        logger.error("No receptors found (.pdbqt). Stop.")
        return

    items = targets_df.to_dict("records")
    all_rows: List[Dict[str, Any]] = []
    processed = 0

    with ProcessPoolExecutor(max_workers=n_cpu) as executor:
        futures = []
        for item in items:
            cid = str(item.get("Cluster_ID"))
            pocket_map = cluster_pockets.get(cid, {})
            futures.append(executor.submit(
                worker_dock_one_ligand_all_targets,
                item,
                pocket_map,
                rec_map,
                grids,
                baselines,
                base_exhaustiveness,
                poses_dir,
            ))

        for f in as_completed(futures):
            rows = f.result() or []
            if rows:
                all_rows.extend(rows)

            processed += 1
            if processed % 10 == 0:
                logger.info(f"Progress: {processed}/{len(items)} ligands processed")

    if not all_rows:
        logger.warning("No docking rows produced.")
        return

    df_energy = pd.DataFrame(all_rows)
    df_energy = df_energy.sort_values(by=["Cluster_ID", "Name", "Target"], ascending=[True, True, True])
    df_energy.to_csv(energies_file, index=False)
    logger.info(f"Saved energies per target: {energies_file}")

    target_to_protein = build_target_to_protein_map(structures)

    out_rows: List[Dict[str, Any]] = []
    group_cols = ["Name", "SMILES", "Cluster_ID"]
    for (name, smi, cid), g in df_energy.groupby(group_cols, dropna=False):
        energies_by_target: Dict[str, float] = {}
        for _, r in g.iterrows():
            t = str(r["Target"])
            e = float(r["Energy"])
            if t not in energies_by_target or e < energies_by_target[t]:
                energies_by_target[t] = e

        for _, r in g.iterrows():
            t = str(r["Target"])
            sel = compute_selectivity_min_for_target(
                energies_by_target=energies_by_target,
                target_id=t,
                target_to_protein=target_to_protein,
            )
            row = dict(r)
            row["Selectivity_Min"] = sel
            out_rows.append(row)

    df_final = pd.DataFrame(out_rows)

    if "Beat_Baseline" in df_final.columns:
        df_final = df_final.sort_values(
            by=["Beat_Baseline", "Selectivity_Min", "Energy"],
            ascending=[False, False, True],
        )

    df_final.to_csv(final_file, index=False)
    logger.info(f"Saved final (energies + selectivity): {final_file}")
    logger.info(f"Saved poses (hits): {poses_dir}")


if __name__ == "__main__":
    main()

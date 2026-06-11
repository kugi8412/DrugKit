#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Iterative active-learning orchestration loop.
"""

import os
from typing import Any, Callable, Dict, List, Optional

import pandas as pd

from siamese_GNN.featurization import smiles_to_graph_gine, feature_dims
from siamese_GNN.trainer import build_labeled_graphs, train_ranknet
from active_learning.acquisition import select_top_uncertain
from active_learning.selectivity import compute_selectivity
from active_learning.uncertainty import mc_dropout_predict


DockFn = Callable[[List[Dict[str, Any]]], pd.DataFrame]


def _train_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "elite_count": cfg["elite_count"],
        "elite_penalty": cfg["elite_penalty"],
        "val_target_ratio": cfg["val_target_ratio"],
        "batch_size": cfg["batch_size"],
        "epochs": cfg["epochs"],
        "learning_rate": cfg["learning_rate"],
        "hidden_dim": cfg["hidden_dim"],
        "dropout": cfg["dropout"],
        "seed": cfg["seed"],
    }


def _label_from_docking(records: List[Dict[str, Any]], dock_fn: DockFn,
                        cfg: Dict[str, Any]) -> pd.DataFrame:
    results = dock_fn(records)
    labeled = compute_selectivity(results, cfg["on_targets"], cfg["off_targets"])
    if not labeled.empty:
        labeled["Cluster_ID"] = "Unknown"
    return labeled


def _build_default_dock_fn(cfg, logger):
    from docking_common.io_utils import load_grids, ensure_dir
    from docking_common.receptor_io import build_receptor_map
    from docking_common.config_utils import load_data_dir
    from active_learning.docking_adapter import dock_compounds
    from active_learning.config import CONFIG_PATH

    grids = load_grids(cfg["grids_file"], logger)
    if not grids:
        raise RuntimeError(f"Could not load grids: {cfg['grids_file']}")
    data_dir = load_data_dir(CONFIG_PATH, {"project": {}})
    rec_map = build_receptor_map(grids, data_dir, logger)
    if not rec_map:
        raise RuntimeError("No receptor files found for the configured grids.")
    poses_dir = os.path.join(cfg["output_dir"], "poses")
    ensure_dir(poses_dir)

    def dock_fn(records):
        return dock_compounds(records, rec_map, grids, cfg, poses_dir, logger)

    return dock_fn


def run_active_learning(cfg: Dict[str, Any], logger=None,
                        dock_fn: Optional[DockFn] = None,
                        device: str = "cpu") -> Dict[str, Any]:
    if not cfg.get("off_targets"):
        raise ValueError("Selectivity requires at least one off_target receptor; "
                         "set active_learning.off_targets in config.yaml.")

    os.makedirs(cfg["output_dir"], exist_ok=True)
    if dock_fn is None:
        dock_fn = _build_default_dock_fn(cfg, logger)

    def log(msg):
        if logger:
            logger.info(msg)

    # Validate input files
    seed_path = cfg["seed_file"]
    pool_path = cfg["pool_file"]
    if not os.path.exists(seed_path):
        raise FileNotFoundError(f"Seed file not found: {seed_path}")
    if not os.path.exists(pool_path):
        raise FileNotFoundError(f"Pool file not found: {pool_path}")

    seed_df = pd.read_csv(seed_path)
    seed_df.columns = [c.strip() for c in seed_df.columns]
    for col in ("Name", "SMILES"):
        if col not in seed_df.columns:
            raise ValueError(f"Seed file missing required column: '{col}'. "
                             f"Found: {list(seed_df.columns)}")

    seed_df = seed_df.head(int(cfg["seed_size"]))
    seed_records = seed_df[["Name", "SMILES"]].to_dict("records")
    log(f"Bootstrap docking {len(seed_records)} seed compounds...")
    labeled = _label_from_docking(seed_records, dock_fn, cfg)
    if len(labeled) < 2:
        raise RuntimeError("Bootstrap produced fewer than 2 labeled compounds; "
                           "cannot train. Check docking / receptor setup.")
    attempted = set(seed_df["Name"].tolist())

    pool_df = pd.read_csv(pool_path)
    pool_df.columns = [c.strip() for c in pool_df.columns]
    for col in ("Name", "SMILES"):
        if col not in pool_df.columns:
            raise ValueError(f"Pool file missing required column: '{col}'. "
                             f"Found: {list(pool_df.columns)}")
    log(f"Pool loaded: {len(pool_df)} candidate compounds")

    node_dim, edge_dim = feature_dims()
    history: List[Dict[str, Any]] = []
    model = None

    for rnd in range(int(cfg["rounds"])):
        log(f"=== Active-learning round {rnd + 1}/{cfg['rounds']} "
            f"(labeled={len(labeled)}) ===")

        graphs, clusters = build_labeled_graphs(
            labeled, sel_col="selectivity", cluster_col="Cluster_ID",
            elite_count=cfg["elite_count"])
        model, train_hist = train_ranknet(
            graphs, clusters, node_dim, edge_dim, _train_cfg(cfg), device,
            logger=logger)

        candidates = pool_df[~pool_df["Name"].isin(attempted)]
        cand_graphs, cand_names = [], []
        for _, row in candidates.iterrows():
            g = smiles_to_graph_gine(row["SMILES"])
            if g is not None:
                cand_graphs.append(g)
                cand_names.append(row["Name"])
        if not cand_graphs:
            log("Pool exhausted; stopping early.")
            history.append({"round": rnd + 1, "labeled_size": len(labeled),
                            "selected": 0, "newly_labeled": 0,
                            "val_rho": train_hist["best_val_rho"],
                            "mean_uncertainty": 0.0, "max_uncertainty": 0.0})
            break

        means, stds = mc_dropout_predict(
            model, cand_graphs, int(cfg["mc_samples"]), device,
            batch_size=cfg["batch_size"], seed=cfg["seed"])
        picked = select_top_uncertain(cand_names, stds,
                                      int(cfg["acquisition_batch"]), attempted)
        log(f"Selected {len(picked)} high-uncertainty compounds: {picked}")

        name_to_smiles = dict(zip(candidates["Name"], candidates["SMILES"]))
        records = [{"Name": n, "SMILES": name_to_smiles[n]} for n in picked]
        new_labeled = _label_from_docking(records, dock_fn, cfg)
        attempted.update(picked)

        if not new_labeled.empty:
            labeled = pd.concat([labeled, new_labeled], ignore_index=True)
            labeled = labeled.drop_duplicates(subset=["SMILES"], keep="first")

        unc_map = dict(zip(cand_names, stds))
        sel_unc = [unc_map[n] for n in picked] or [0.0]
        history.append({
            "round": rnd + 1,
            "labeled_size": len(labeled),
            "selected": len(picked),
            "newly_labeled": int(len(new_labeled)),
            "val_rho": float(train_hist["best_val_rho"]),
            "mean_uncertainty": float(sum(sel_unc) / len(sel_unc)),
            "max_uncertainty": float(max(sel_unc)),
        })

    labeled_path = os.path.join(cfg["output_dir"], "labeled.csv")
    history_path = os.path.join(cfg["output_dir"], "al_history.csv")
    model_path = os.path.join(cfg["output_dir"], "al_model.pth")
    labeled.to_csv(labeled_path, index=False)
    pd.DataFrame(history).to_csv(history_path, index=False)
    if model is not None:
        import torch
        torch.save(model.state_dict(), model_path)

    log(f"Done. Final labeled={len(labeled)}; outputs in {cfg['output_dir']}")
    return {"model": model, "labeled": labeled, "history": history}

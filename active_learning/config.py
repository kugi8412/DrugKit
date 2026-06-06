# -*- coding: utf-8 -*-

from typing import Any, Dict

from docking_common.config_utils import merge_section

CONFIG_PATH = "config.yaml"

DEFAULT_CONFIG: Dict[str, Any] = {
    "pool_file": "data/pool.csv",
    "seed_file": "data/seed_ligands.csv",
    "grids_file": "docking_grids.json",
    "on_targets": ["HIVPRO_1HSG"],
    "off_targets": [],
    "rounds": 5,
    "seed_size": 20,
    "acquisition_batch": 10,
    "mc_samples": 30,
    "epochs": 40,
    "batch_size": 32,
    "learning_rate": 0.0004,
    "hidden_dim": 128,
    "dropout": 0.3,
    "elite_count": 10,
    "elite_penalty": 5.0,
    "val_target_ratio": 0.15,
    "seed": 42,
    "smina_exe": "smina",
    "exhaustiveness": 8,
    "num_modes": 1,
    "n_cpu": 2,
    "default_baseline": -7.0,
    "output_dir": "output/active_learning",
}


def merge_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    return merge_section(raw, "active_learning", DEFAULT_CONFIG)

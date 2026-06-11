#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Any, Dict

from docking_common.config_utils import merge_section

CONFIG_PATH = "config.yaml"

DEFAULT_CONFIG: Dict[str, Any] = {
    "known_compounds": "data/known_compounds.csv",
    "candidates_file": "data/candidates.csv",
    "grids_file": "docking_grids.json",
    "results_file": "output/docking_smina_results.csv",
    "smina_exe": "smina",
    "exhaustiveness": 16,
    "num_modes": 1,
    "n_cpu": 4,
    "default_baseline": -7.0,
}


def merge_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    return merge_section(raw, "docking_smina", DEFAULT_CONFIG)

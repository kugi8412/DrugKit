# -*- coding: utf-8 -*-

from typing import Any, Dict

from docking_common.config_utils import merge_section

CONFIG_PATH = "config.yaml"

DEFAULT_CONFIG: Dict[str, Any] = {
    "known_compounds": "data/known_compounds.csv",
    "candidates_file": "data/candidates.csv",
    "grids_file": "docking_grids.json",
    "results_file": "output/docking_results.csv",
    "exhaustiveness": 8,
    "n_poses": 1,
}


def merge_config(raw: Dict[str, Any]) -> Dict[str, Any]:
    return merge_section(raw, "docking", DEFAULT_CONFIG)

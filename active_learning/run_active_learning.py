#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from pocket_logic.logging_utils import setup_logging
from docking_common.config_utils import read_yaml
from active_learning.config import CONFIG_PATH, merge_config
from active_learning.loop import run_active_learning


def main() -> None:
    raw = read_yaml(CONFIG_PATH)
    cfg = merge_config(raw)["active_learning"]

    logger = setup_logging(log_dir="logs", log_file="active_learning.log")
    logger.info("--- START Active Learning Loop ---")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    result = run_active_learning(cfg, logger=logger, dock_fn=None, device=device)
    logger.info(f"Final labeled set size: {len(result['labeled'])}")
    logger.info(f"History:\n{result['history']}")


if __name__ == "__main__":
    main()

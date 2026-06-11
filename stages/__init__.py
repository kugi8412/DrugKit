#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DrugKit stages run each pipeline stage independently.

Each stage is a standalone module with its own CLI:

    python -m stages.featurize --input data/pool.csv --output output/graphs.pt
    python -m stages.dock --grids docking_grids.json --engine smina
    python -m stages.train --labeled output/labeled.csv --epochs 40
    python -m stages.predict --model output/model.pth --input library.csv
    python -m stages.active_learn --config config.yaml
"""

from stages.featurize import run_featurize
from stages.dock import run_dock
from stages.train import run_train
from stages.predict import run_predict
from stages.active_learn import run_active_learn
from stages.select import run_select
from stages.expand import run_expand


__all__ = [
    "run_featurize",
    "run_dock",
    "run_train",
    "run_predict",
    "run_active_learn",
    "run_select",
    "run_expand",
]

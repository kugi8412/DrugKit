#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import pandas as pd

from active_learning.config import DEFAULT_CONFIG
from active_learning.loop import run_active_learning


def _make_pool(path, n):
    smiles = ["CCO", "CCN", "CCCl", "c1ccccc1", "CCCC", "CCOCC", "CCC=O",
              "CC(=O)O", "CCBr", "CCS", "CCCO", "CCCN"]
    pd.DataFrame({"Name": [f"p{i}" for i in range(n)],
                  "SMILES": smiles[:n]}).to_csv(path, index=False)


def _make_seed(path):
    pd.DataFrame({"Name": ["s0", "s1", "s2", "s3"],
                  "SMILES": ["CC", "CCC", "CCCC", "CCCCC"],
                  "Target": ["HIVPRO_1HSG"] * 4}).to_csv(path, index=False)


def _fake_dock(records):
    rows = []
    for r in records:
        n = len(r["SMILES"])
        rows.append({"Name": r["Name"], "SMILES": r["SMILES"],
                     "Target": "HIVPRO_1HSG", "Pocket_ID": "p1", "Energy": -7.0 - n * 0.1})
        rows.append({"Name": r["Name"], "SMILES": r["SMILES"],
                     "Target": "RENIN_2V0Z", "Pocket_ID": "q1", "Energy": -6.0 - n * 0.05})
    return pd.DataFrame(rows)


def test_loop_grows_labeled_set(tmp_path):
    pool = tmp_path / "pool.csv"
    seed = tmp_path / "seed.csv"
    _make_pool(pool, 12)
    _make_seed(seed)

    cfg = dict(DEFAULT_CONFIG)
    cfg.update({
        "pool_file": str(pool), "seed_file": str(seed),
        "on_targets": ["HIVPRO_1HSG"], "off_targets": ["RENIN_2V0Z"],
        "rounds": 2, "seed_size": 4, "acquisition_batch": 3, "mc_samples": 5,
        "epochs": 2, "hidden_dim": 16, "batch_size": 8,
        "output_dir": str(tmp_path / "out"),
    })

    result = run_active_learning(cfg, logger=None, dock_fn=_fake_dock, device="cpu")
    labeled = result["labeled"]
    assert len(labeled) >= 4
    assert len(labeled) <= 4 + 3 * 2
    assert {"Name", "SMILES", "selectivity", "Cluster_ID"}.issubset(labeled.columns)
    assert len(result["history"]) == 2
    assert os.path.exists(os.path.join(cfg["output_dir"], "labeled.csv"))
    assert os.path.exists(os.path.join(cfg["output_dir"], "al_history.csv"))


def test_loop_requires_off_targets(tmp_path):
    pool = tmp_path / "pool.csv"
    seed = tmp_path / "seed.csv"
    _make_pool(pool, 6)
    _make_seed(seed)
    cfg = dict(DEFAULT_CONFIG)
    cfg.update({"pool_file": str(pool), "seed_file": str(seed),
                "off_targets": [], "output_dir": str(tmp_path / "out")})
    try:
        run_active_learning(cfg, logger=None, dock_fn=_fake_dock, device="cpu")
        assert False, "expected ValueError"
    except ValueError as e:
        assert "off_target" in str(e).lower()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd

from active_learning.selectivity import compute_selectivity


def _results():
    return pd.DataFrame([
        {"Name": "A", "SMILES": "CCO", "Target": "HIVPRO_1HSG", "Energy": -9.0},
        {"Name": "A", "SMILES": "CCO", "Target": "HIVPRO_1HSG", "Energy": -8.0},
        {"Name": "A", "SMILES": "CCO", "Target": "RENIN_2V0Z", "Energy": -5.0},
        {"Name": "B", "SMILES": "CCN", "Target": "HIVPRO_1HSG", "Energy": -6.0},
        {"Name": "B", "SMILES": "CCN", "Target": "RENIN_2V0Z", "Energy": -8.5},
    ])


def test_selectivity_difference_and_min_over_pockets():
    out = compute_selectivity(_results(), on_targets=["HIVPRO_1HSG"],
                              off_targets=["RENIN_2V0Z"])
    row_a = out[out["Name"] == "A"].iloc[0]
    assert row_a["target_best_E"] == -9.0
    assert row_a["offtarget_best_E"] == -5.0
    assert row_a["selectivity"] == -4.0
    row_b = out[out["Name"] == "B"].iloc[0]
    assert row_b["selectivity"] == 2.5


def test_missing_target_dropped():
    df = pd.DataFrame([
        {"Name": "C", "SMILES": "CC", "Target": "HIVPRO_1HSG", "Energy": -7.0},
    ])
    out = compute_selectivity(df, on_targets=["HIVPRO_1HSG"],
                              off_targets=["RENIN_2V0Z"])
    assert out.empty


def test_custom_selectivity_fn():
    out = compute_selectivity(_results(), on_targets=["HIVPRO_1HSG"],
                              off_targets=["RENIN_2V0Z"],
                              selectivity_fn=lambda on, off: on)
    assert out[out["Name"] == "A"].iloc[0]["selectivity"] == -9.0

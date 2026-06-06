# -*- coding: utf-8 -*-
"""Compute per-compound selectivity from per-pocket docking energies."""

from typing import Callable, List, Optional

import pandas as pd


def default_selectivity(on_target_best: float, offtarget_best: float) -> float:
    """More negative = binds target better than off-targets = more selective."""
    return on_target_best - offtarget_best


def compute_selectivity(
    results: pd.DataFrame,
    on_targets: List[str],
    off_targets: List[str],
    name_col: str = "Name",
    smiles_col: str = "SMILES",
    target_col: str = "Target",
    energy_col: str = "Energy",
    selectivity_fn: Optional[Callable[[float, float], float]] = None,
) -> pd.DataFrame:
    """Return a DataFrame with Name, SMILES, target_best_E, offtarget_best_E, selectivity.

    Compounds missing any on-target or off-target energy are dropped.
    """
    if selectivity_fn is None:
        selectivity_fn = default_selectivity
    if results is None or results.empty:
        return pd.DataFrame(
            columns=[name_col, smiles_col, "target_best_E", "offtarget_best_E", "selectivity"])

    on_set, off_set = set(on_targets), set(off_targets)
    rows = []
    for (name, smiles), grp in results.groupby([name_col, smiles_col]):
        on_vals = grp[grp[target_col].isin(on_set)][energy_col].dropna()
        off_vals = grp[grp[target_col].isin(off_set)][energy_col].dropna()
        if on_vals.empty or off_vals.empty:
            continue
        on_best = float(on_vals.min())
        off_best = float(off_vals.min())
        rows.append({
            name_col: name,
            smiles_col: smiles,
            "target_best_E": on_best,
            "offtarget_best_E": off_best,
            "selectivity": float(selectivity_fn(on_best, off_best)),
        })
    return pd.DataFrame(rows)

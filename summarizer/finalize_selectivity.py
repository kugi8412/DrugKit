# -*- coding: utf-8 -*-

import os
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd

from ipz_core.config_loader import ConfigLoader
from ipz_core.logging_utils import setup_logging


REQUIRED_COLS = [
    "Name", "SMILES", "Target", "Pocket_ID", "Energy",
    "Beat_Baseline", "Baseline_Value"
]


# ----------------------------
# config helpers
# ----------------------------

def parse_structures(loader: ConfigLoader) -> Tuple[Dict[str, str], Dict[str, Set[str]]]:
    """
    Returns:
      target_to_protein: conformation_key -> protein_key
      protein_to_targets: protein_key -> set(conformation_key)
    """
    structures = loader.structures()

    target_to_protein: Dict[str, str] = {}
    protein_to_targets: Dict[str, Set[str]] = {}

    for _, scfg in structures.items():
        prot = str(scfg.protein_key)
        tgt = str(scfg.conformation_key)
        target_to_protein[tgt] = prot
        protein_to_targets.setdefault(prot, set()).add(tgt)

    return target_to_protein, protein_to_targets


def resolve_paths(loader: ConfigLoader) -> Tuple[str, str, str, str]:
    """
    Returns:
      energies_csv, wide_csv, final_results_csv, focus_protein
    """
    raw = loader.raw or {}

    project = loader.project()
    prj_raw = raw.get("project", {}) or {}
    dock_raw = raw.get("docking", {}) or {}

    energies_csv = dock_raw.get("results_energies_file", "output/docking_energies_etc.csv")
    wide_csv = dock_raw.get("results_etc_wide_file", "output/docking_energies_etc_wide.csv")

    # final_results path: project.final_results
    final_results_csv = prj_raw.get("final_results")
    if not final_results_csv:
        if project.output_dir:
            final_results_csv = os.path.join(project.output_dir, "final_results.csv")
        else:
            final_results_csv = "output/final_results.csv"

    # focus protein: docking.focus_protein (optional), default SLC6A20
    focus_protein = dock_raw.get("focus_protein", "SLC6A20")

    return energies_csv, wide_csv, final_results_csv, focus_protein


# ----------------------------
# dataframe helpers
# ----------------------------

def validate_input_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input CSV: {missing}")

    df["Target"] = df["Target"].astype(str)
    df["Energy"] = pd.to_numeric(df["Energy"], errors="coerce")
    df["Baseline_Value"] = pd.to_numeric(df["Baseline_Value"], errors="coerce")
    df["Beat_Baseline"] = df["Beat_Baseline"].astype(str).str.lower().map(
        {"true": True, "false": False, "1": True, "0": False}
    )
    return df


def choose_best_row_per_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    If there are duplicates for (Name, SMILES, Target), we take the minimum energy.
    """
    key_cols = ["Name", "SMILES", "Target"]
    df2 = df.dropna(subset=["Energy"]).copy()
    df2 = df2.sort_values(by=key_cols + ["Energy"], ascending=[True, True, True, True])
    df2 = df2.groupby(key_cols, as_index=False).first()
    return df2


def pivot_targets_to_columns(df_best: pd.DataFrame) -> pd.DataFrame:
    """
    Wide per (Name, SMILES):
      <Target>_Pocket_ID, <Target>_Energy, <Target>_Baseline_Value, <Target>_Beat
    """
    idx = ["Name", "SMILES"]

    pocket = df_best.pivot(index=idx, columns="Target", values="Pocket_ID")
    energy = df_best.pivot(index=idx, columns="Target", values="Energy")
    baseline = df_best.pivot(index=idx, columns="Target", values="Baseline_Value")
    beat = df_best.pivot(index=idx, columns="Target", values="Beat_Baseline")

    pocket.columns = [f"{c}_Pocket_ID" for c in pocket.columns.astype(str)]
    energy.columns = [f"{c}_Energy" for c in energy.columns.astype(str)]
    baseline.columns = [f"{c}_Baseline_Value" for c in baseline.columns.astype(str)]
    beat.columns = [f"{c}_Beat" for c in beat.columns.astype(str)]

    wide = pd.concat([pocket, energy, baseline, beat], axis=1).reset_index()

    fixed = ["Name", "SMILES"]
    rest = sorted([c for c in wide.columns if c not in fixed])
    return wide[fixed + rest]


def compute_selectivity(
    df_best: pd.DataFrame,
    target_to_protein: Dict[str, str],
    protein_to_targets: Dict[str, Set[str]],
    focus_protein: str,
    strict_missing_focus: bool = True,
) -> pd.DataFrame:
    """
      Selectivity = min(E_others) - max(E_focus)

    strict_missing_focus=True:
      if result is missing for any focus conformation from config.yaml -> Selectivity = NaN
    """
    key = ["Name", "SMILES"]
    focus_targets = set(protein_to_targets.get(focus_protein, set()))

    df = df_best.copy()
    df["Protein"] = df["Target"].map(target_to_protein)

    def agg_group(g: pd.DataFrame) -> float:
        # check missing focus conformations
        present_focus = set(g.loc[g["Protein"] == focus_protein, "Target"].astype(str).tolist())
        missing_focus = focus_targets - present_focus
        if strict_missing_focus and len(missing_focus) > 0:
            return float("nan")

        focus_rows = g[g["Protein"] == focus_protein]
        if focus_rows.empty or focus_rows["Energy"].isna().all():
            return float("nan")

        a_worst = float(focus_rows["Energy"].max())  # worst = highest

        other_rows = g[g["Protein"].notna() & (g["Protein"] != focus_protein)]
        if other_rows.empty or other_rows["Energy"].isna().all():
            return 0.0  # no competitors

        b_best = float(other_rows["Energy"].min())  # best competitor = lowest
        return float(b_best - a_worst)

    sel = (
        df.groupby(key, dropna=False)
          .apply(agg_group, include_groups=False)
          .reset_index(name="Selectivity")
    )
    return sel


# ----------------------------
# main
# ----------------------------

def main() -> None:
    cfg_path = "config.yaml"
    if not os.path.exists(cfg_path) and os.path.exists(f"../{cfg_path}"):
        cfg_path = f"../{cfg_path}"

    logger = setup_logging(log_dir="logs", log_file="finalize_selectivity.log")

    loader = ConfigLoader(cfg_path)
    energies_csv, wide_csv, final_results_csv, focus_protein = resolve_paths(loader)

    logger.info(f"Input energies: {energies_csv}")
    logger.info(f"Wide output: {wide_csv}")
    logger.info(f"Final results (top10): {final_results_csv}")
    logger.info(f"Focus protein: {focus_protein}")

    if not os.path.exists(energies_csv):
        logger.error(f"Missing energies file: {energies_csv}")
        return

    target_to_protein, protein_to_targets = parse_structures(loader)
    if focus_protein not in protein_to_targets:
        logger.error(f"Focus protein '{focus_protein}' not found in config.structures")
        return

    df = pd.read_csv(energies_csv)
    df = validate_input_df(df)
    df_best = choose_best_row_per_target(df)

    wide = pivot_targets_to_columns(df_best)
    sel = compute_selectivity(
        df_best=df_best,
        target_to_protein=target_to_protein,
        protein_to_targets=protein_to_targets,
        focus_protein=focus_protein,
        strict_missing_focus=True,
    )

    out = wide.merge(sel, on=["Name", "SMILES"], how="left")

    # Save full wide
    os.makedirs(os.path.dirname(wide_csv) or ".", exist_ok=True)
    out.to_csv(wide_csv, index=False)
    logger.info(f"Saved wide+selectivity: {wide_csv}")

    # Save top 10
    top10 = out.dropna(subset=["Selectivity"]).sort_values("Selectivity", ascending=False).head(10)

    os.makedirs(os.path.dirname(final_results_csv) or ".", exist_ok=True)
    top10.to_csv(final_results_csv, index=False)
    logger.info(f"Saved top10 final_results: {final_results_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 6: Selectivity Analysis compute selectivity scores between targets.

Usage:
    python -m stages.select --docking-results results/ --on-targets CDK2 --off-targets CDK4 CDK6
    python -m stages.select --input docking_all.csv --on-targets CDK2 --off-targets CDK4
    python -m stages.select --input results.csv --method ratio --threshold 2.0

Parameters:
    --input             CSV with docking results (multi-target columns)
    --docking-results   Directory with per-target result files
    --output            Output CSV with selectivity scores
    --on-targets        Target names to optimize FOR (required)
    --off-targets       Target names to optimize AGAINST (required)
    --method            Selectivity method: ratio, difference, pareto (default: ratio)
    --threshold         Selectivity threshold (default: 2.0 for ratio)
    --score-col-prefix  Prefix for score columns (default: "score_")
    --top-k             Output only top K selective compounds
    --ascending         Lower scores are better (default: True for docking)
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def run_select(
    input_file: Optional[str] = None,
    docking_results_dir: Optional[str] = None,
    output_file: str = "output/selectivity_results.csv",
    on_targets: Optional[List[str]] = None,
    off_targets: Optional[List[str]] = None,
    method: str = "ratio",
    threshold: float = 2.0,
    score_col_prefix: str = "score_",
    top_k: Optional[int] = None,
    ascending: bool = True,
) -> str:
    """Compute selectivity scores.

    Args:
        input_file: CSV with multi-target scores (columns: score_CDK2, score_CDK4, ...).
        docking_results_dir: Directory with separate per-target CSVs.
        output_file: Output file path.
        on_targets: Target names to optimize for.
        off_targets: Target names to penalize.
        method: "ratio" (on/off), "difference" (on - off), "pareto" (multi-objective).
        threshold: Filter threshold.
        score_col_prefix: Prefix for score columns in the DataFrame.
        top_k: Keep only top K compounds.
        ascending: If True, lower score = better (typical for docking).

    Returns:
        Path to output CSV.
    """
    if not on_targets or not off_targets:
        raise ValueError("Both --on-targets and --off-targets are required.")

    # Load data
    if input_file:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input not found: {input_file}")
        df = pd.read_csv(input_file)
    elif docking_results_dir:
        if not os.path.isdir(docking_results_dir):
            raise FileNotFoundError(f"Directory not found: {docking_results_dir}")
        df = _load_from_directory(docking_results_dir, score_col_prefix)
    else:
        raise ValueError("Provide either --input or --docking-results.")

    # Resolve score columns
    all_targets = on_targets + off_targets
    score_cols = {}
    for target in all_targets:
        col = f"{score_col_prefix}{target}"
        if col not in df.columns:
            # Try without prefix
            if target in df.columns:
                col = target
            else:
                raise ValueError(
                    f"Score column for target '{target}' not found. "
                    f"Tried '{score_col_prefix}{target}' and '{target}'. "
                    f"Available: {list(df.columns)}"
                )
        score_cols[target] = col

    # Compute selectivity
    print(f"Computing selectivity ({method}):")
    print(f"  ON-targets:  {on_targets}")
    print(f"  OFF-targets: {off_targets}")
    print(f"  Compounds:   {len(df)}")

    if method == "ratio":
        df["selectivity"] = _selectivity_ratio(
            df, score_cols, on_targets, off_targets, ascending
        )
    elif method == "difference":
        df["selectivity"] = _selectivity_difference(
            df, score_cols, on_targets, off_targets, ascending
        )
    elif method == "pareto":
        df["selectivity"] = _selectivity_pareto(
            df, score_cols, on_targets, off_targets, ascending
        )
    else:
        raise ValueError(f"Unknown method: '{method}'. Use ratio, difference, or pareto.")

    # Filter by threshold
    if method == "ratio":
        selective = df[df["selectivity"] >= threshold]
    else:
        selective = df[df["selectivity"] >= threshold]

    print(f"  Selective (threshold {threshold}): {len(selective)}/{len(df)}")

    # Sort: higher selectivity = better
    df = df.sort_values("selectivity", ascending=False)

    if top_k:
        df = df.head(top_k)
        print(f"  Top {top_k} kept")

    # Save
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Saved to {output_file}")
    return output_file


def _selectivity_ratio(df, score_cols, on_targets, off_targets, ascending):
    """Ratio of off-target to on-target scores."""
    on_scores = df[[score_cols[t] for t in on_targets]].mean(axis=1)
    off_scores = df[[score_cols[t] for t in off_targets]].mean(axis=1)

    if ascending:
        # Lower = better â†’ selective means on < off â†’ ratio = off/on
        ratio = off_scores / on_scores.replace(0, np.nan)
    else:
        # Higher = better â†’ ratio = on/off
        ratio = on_scores / off_scores.replace(0, np.nan)
    return ratio


def _selectivity_difference(df, score_cols, on_targets, off_targets, ascending):
    """Difference between off-target and on-target scores."""
    on_scores = df[[score_cols[t] for t in on_targets]].mean(axis=1)
    off_scores = df[[score_cols[t] for t in off_targets]].mean(axis=1)

    if ascending:
        # Lower = better â†’ selective means off - on > 0
        return off_scores - on_scores
    else:
        return on_scores - off_scores


def _selectivity_pareto(df, score_cols, on_targets, off_targets, ascending):
    """Pareto-based selectivity: combined rank across objectives."""
    on_scores = df[[score_cols[t] for t in on_targets]].mean(axis=1)
    off_scores = df[[score_cols[t] for t in off_targets]].mean(axis=1)

    # Rank-based: good on-target rank + bad off-target rank
    if ascending:
        on_rank = on_scores.rank(ascending=True)   # low score = good = low rank
        off_rank = off_scores.rank(ascending=False)  # high score = bad at off-target = good
    else:
        on_rank = on_scores.rank(ascending=False)
        off_rank = off_scores.rank(ascending=True)

    # Combined rank score (higher = more selective)
    n = len(df)
    return (n - on_rank + off_rank) / n


def _load_from_directory(results_dir, score_col_prefix):
    """Load separate per-target CSV files and merge them."""
    result_files = sorted(Path(results_dir).glob("*.csv"))
    if not result_files:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")

    merged = None
    for f in result_files:
        target_name = f.stem  # filename without extension as target name
        target_df = pd.read_csv(f)

        # Identify score column
        for col in ("score", "affinity", "docking_score"):
            if col in target_df.columns:
                target_df = target_df.rename(columns={col: f"{score_col_prefix}{target_name}"})
                break

        if merged is None:
            merged = target_df
        else:
            # Merge on SMILES
            smiles_col = next(
                (c for c in ("SMILES", "smiles", "Smiles") if c in target_df.columns),
                target_df.columns[0]
            )
            merged = merged.merge(target_df, on=smiles_col, how="outer",
                                  suffixes=("", f"_{target_name}"))

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="DrugKit Stage 6: Selectivity Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m stages.select --input multi_dock.csv --on-targets CDK2 --off-targets CDK4 CDK6
  python -m stages.select --docking-results results/ --on-targets JAK2 --off-targets JAK1 JAK3
  python -m stages.select --input results.csv --on-targets EGFR --off-targets ERBB2 \\
      --method pareto --top-k 100
        """,
    )
    parser.add_argument("--input", "-i", help="CSV with multi-target docking scores")
    parser.add_argument("--docking-results", help="Directory with per-target CSV files")
    parser.add_argument("--output", "-o", default="output/selectivity_results.csv",
                        help="Output CSV (default: output/selectivity_results.csv)")
    parser.add_argument("--on-targets", nargs="+", required=True,
                        help="Target(s) to optimize FOR")
    parser.add_argument("--off-targets", nargs="+", required=True,
                        help="Target(s) to penalize")
    parser.add_argument("--method", choices=["ratio", "difference", "pareto"],
                        default="ratio", help="Selectivity method (default: ratio)")
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="Selectivity threshold (default: 2.0)")
    parser.add_argument("--score-col-prefix", default="score_",
                        help="Score column prefix (default: score_)")
    parser.add_argument("--top-k", type=int, help="Keep only top K compounds")
    parser.add_argument("--no-ascending", action="store_true",
                        help="Higher score = better (inverts default)")
    args = parser.parse_args()

    run_select(
        input_file=args.input,
        docking_results_dir=args.docking_results,
        output_file=args.output,
        on_targets=args.on_targets,
        off_targets=args.off_targets,
        method=args.method,
        threshold=args.threshold,
        score_col_prefix=args.score_col_prefix,
        top_k=args.top_k,
        ascending=not args.no_ascending,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import logging
from typing import List, Tuple, Set, Optional, Any

import numpy as np
import pandas as pd

from pocket_logic.p2rank_stage import validate_quality
from pocket_logic.geometry_utils import get_residue_coords, calculate_centered_box


def load_expert_df(pockets_csv: str) -> pd.DataFrame:
    if os.path.exists(pockets_csv):
        return pd.read_csv(pockets_csv)
    return pd.DataFrame()


def extract_expert_groups(expert_df: pd.DataFrame,
                          protein_key: Optional[str],
                          conformation: Optional[str]) -> List[Tuple[Any, pd.DataFrame]]:
    """
    Filter expert CSV by (Protein AND Conformation).
    Keep groups even if some columns are missing/NaN (e.g., Ligand, Hbond).
    """
    if expert_df.empty:
        return []

    if "Conformation" not in expert_df.columns or "Protein" not in expert_df.columns:
        return []

    if not conformation or not protein_key:
        return []

    subset = expert_df[
        (expert_df["Conformation"] == conformation) &
        (expert_df["Protein"] == protein_key)
    ]
    if subset.empty:
        return []

    # We want grouping keys like before, but do not drop NaN groups
    cols = [c for c in ["Site", "Ligand", "Conformation"] if c in subset.columns]

    groups: List[Tuple[Any, pd.DataFrame]] = []
    for keys, group in subset.groupby(cols, dropna=False):
        groups.append((keys, group))
    return groups


def parse_residue_ids_from_group(group: pd.DataFrame) -> Set[int]:
    res_ids: Set[int] = set()
    for r in group["Residue"].dropna():
        nums = re.findall(r"\d+", str(r))
        for n in nums:
            res_ids.add(int(n))
    return res_ids


def _safe_token(x) -> str:
    # placeholder for None/NaN/empty strings
    if x is None:
        return "_"
    try:
        # pandas NaN
        if pd.isna(x):
            return "_"
    except Exception:
        pass
    s = str(x).strip()
    return s if s else "_"


def add_expert_pockets(chain_pockets: List[dict],
                       structure,
                       chain_id: str,
                       expert_df: pd.DataFrame,
                        protein_key: Optional[str],
                       conformation: Optional[str],
                       buffer_size: float,
                       p2rank_global: List[dict],
                       logger: logging.Logger) -> int:
    added_expert = 0

    for keys, group in extract_expert_groups(expert_df, protein_key, conformation):
        res_ids = parse_residue_ids_from_group(group)
        coords = get_residue_coords(structure, chain_id, res_ids)

        if coords.size > 0:
            center, size = calculate_centered_box(coords, buffer_size)
            val = validate_quality(center, p2rank_global, chain_id)
            key_list = list(keys) if isinstance(keys, tuple) else [keys]
            p_name = "_".join(_safe_token(k) for k in key_list)
            # p_name = "_".join([str(k) for k in (keys if isinstance(keys, tuple) else [keys])])

            chain_pockets.append({
                "id": f"{chain_id}_expert_{p_name}",
                "center": [round(x, 3) for x in center],
                "size": [round(x, 3) for x in size],
                "source": "Expert",
                "validation": val
            })
            added_expert += 1

    logger.info(f"    + Added {added_expert} Expert pockets.")
    if added_expert == 0:
        logger.debug(
            f"    (Possible reason: no matching conformation='{conformation}' or no residues for chain {chain_id})"
        )
    return added_expert

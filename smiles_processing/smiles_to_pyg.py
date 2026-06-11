#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# smiles_processing/smiles_to_pyg.py

"""Shared preprocessing entry point: SMILES → torch_geometric.data.Data.

This module is the **single canonical conversion function** that should be
used by all DrugKit pipelines:

* siamese_GNN/improved_train.py   (training)
* final_docking/virtual_screeing.py (virtual screening / inference)
* Any future batch processing pipeline

It replaces the ad-hoc smiles_to_graph_gine functions that currently
live inside those scripts and depend on RDKit.

Public API
----------
smiles_to_pyg(smiles, *, y=None, is_elite=False, strict=True) -> Data | None

The returned Data object carries the same attributes expected by the
existing GINEConv model code:

  data.x          -> FloatTensor (N, 42)   atom features
  data.edge_index -> LongTensor  (2, 2E)   bidirectional COO
  data.edge_attr  -> FloatTensor (2E, 11)  bond features (duplicated)
  data.smiles     -> str                   original SMILES
  data.y          -> FloatTensor (1,) | None
  data.is_elite   -> FloatTensor (1,)      0.0 or 1.0

where N = atom count, E = bond count (each bond stored twice for
undirected message passing).
"""

from __future__ import annotations

import logging
from typing import Optional

import torch
from torch_geometric.data import Data

from smiles_processing.smiles_parser import parse_smiles
from smiles_processing.smiles_features import extract_features
from smiles_processing.feature_encoding import (
    encode_atom,
    encode_bond,
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
)

logger = logging.getLogger(__name__)

_EMPTY_EDGE_INDEX: torch.Tensor = torch.empty((2, 0), dtype=torch.long)
_EMPTY_EDGE_ATTR: torch.Tensor = torch.empty((0, BOND_FEATURE_DIM), dtype=torch.float)


def smiles_to_pyg(
    smiles: str,
    *,
    y: Optional[float] = None,
    is_elite: bool = False,
    strict: bool = True,
) -> Optional[Data]:
    """Convert a SMILES string to a torch_geometric.data.Data object.

    This is the **shared preprocessing entry point** for all DrugKit
    pipelines.  It chains:

    1. :func:`~smiles_processing.smiles_parser.parse_smiles`  -> graph extraction
    2. :func:`~smiles_processing.smiles_features.extract_features` -> feature enrichment
    3. :func:`~smiles_processing.feature_encoding.encode_atom` /
       :func:`~smiles_processing.feature_encoding.encode_bond` -> tensor encoding

    The resulting Data object is directly compatible with the
    GINEConv-based SiameseRankNet model.

    Args:
        smiles: Input SMILES string.
        y: Optional float target label (e.g. selectivity score).
            If provided, stored as data.y = torch.tensor([y]).
        is_elite: If True, data.is_elite is set to 1.0
            (used by the elite-penalty loss in training).
        strict: Forwarded to the parser.  Use strict=False for
            large-scale preprocessing where robustness > correctness.

    Returns:
        A torch_geometric.data.Data instance, or None if the
        SMILES could not be parsed (only possible when strict=False).

    Examples:
        >>> data = smiles_to_pyg("CCO")
        >>> data.x.shape
        torch.Size([3, 42])
        >>> data.edge_attr.shape
        torch.Size([4, 11])
        >>> data.smiles
        'CCO'

        >>> data = smiles_to_pyg("CC(=O)O", y=-7.5, is_elite=True)
        >>> data.y.item()
        -7.5
        >>> data.is_elite.item()
        1.0
    """
    try:
        graph = parse_smiles(smiles, strict=strict)
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise
        logger.warning("smiles_to_pyg: parse failed for %r -> %s", smiles, exc)
        return None

    # Enrich atoms with hybridization (modifies graph in-place)
    try:
        featured = extract_features(graph)
    except Exception as exc:  # noqa: BLE001
        if strict:
            raise
        logger.warning("smiles_to_pyg: feature extraction failed for %r -> %s", smiles, exc)
        return None

    atom_feat_list = [encode_atom(a, graph) for a in featured["atoms"]]
    x = torch.tensor(atom_feat_list, dtype=torch.float)

    if not featured["bonds"]:
        edge_index = _EMPTY_EDGE_INDEX
        edge_attr = _EMPTY_EDGE_ATTR
    else:
        rows: list[int] = []
        cols: list[int] = []
        edge_feats: list[list[float]] = []

        for bond in featured["bonds"]:
            start, end = bond["start"], bond["end"]
            feat = encode_bond(bond)
            # Both directions for undirected message passing
            rows += [start, end]
            cols += [end, start]
            edge_feats += [feat, feat]

        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)

    y_tensor = torch.tensor([y], dtype=torch.float) if y is not None else None
    elite_tensor = torch.tensor([1.0 if is_elite else 0.0], dtype=torch.float)

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        smiles=smiles,
        y=y_tensor,
        is_elite=elite_tensor,
    )


def batch_smiles_to_pyg(
    smiles_list: list[str],
    *,
    labels: Optional[list[Optional[float]]] = None,
    elite_set: Optional[set[str]] = None,
    strict: bool = False,
) -> tuple[list[Data], list[str]]:
    """Convert a list of SMILES strings to Data objects in bulk.

    Failed conversions are silently skipped (strict=False by default
    because this function is intended for large-scale preprocessing).

    Args:
        smiles_list: Input SMILES strings.
        labels: Optional list of float labels aligned with *smiles_list*.
            None entries are allowed.
        elite_set: Optional set of SMILES strings that should be flagged
            as elite (is_elite=1.0).
        strict: Forwarded to :func:`smiles_to_pyg`.  Defaults to False
            for batch mode.

    Returns:
        A tuple of:

        * data_list -> successfully converted Data objects.
        * failed -> SMILES strings that could not be converted.

    Example:
        >>> data_list, failed = batch_smiles_to_pyg(["CCO", "C1CCCCC1", "INVALID???"])
        >>> len(data_list)
        2
        >>> len(failed)
        1
    """
    if labels is None:
        labels = [None] * len(smiles_list)
    if elite_set is None:
        elite_set = set()

    data_list: list[Data] = []
    failed: list[str] = []

    for smi, label in zip(smiles_list, labels):
        data = smiles_to_pyg(
            smi,
            y=label,
            is_elite=(smi in elite_set),
            strict=strict,
        )
        if data is None:
            failed.append(smi)
        else:
            data_list.append(data)

    if failed:
        logger.info(
            "batch_smiles_to_pyg: %d/%d SMILES failed conversion",
            len(failed), len(smiles_list),
        )

    return data_list, failed

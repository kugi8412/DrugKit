#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# smiles_processing/__init__.py

"""
DrugKit SMILES preprocessing package.

Public surface
--------------
The recommended entry points are:

* :func:`smiles_to_pyg` — convert a single SMILES to a PyG Data object.
* :func:`batch_smiles_to_pyg` — convert a list of SMILES in bulk.
* :func:`parse_smiles` — raw graph extraction (returns a plain dict).
* :func:`extract_features` — enrich a raw graph with atom/bond features.
* :data:`ATOM_FEATURE_DIM` / :data:`BOND_FEATURE_DIM` — tensor dimensions (42, 11).
"""

from smiles_processing.smiles_errors import (
    SMILESError,
    SMILESTokenizationError,
    SMILESParseError,
    SMILESValidationError,
    UnsupportedSMILESFeatureError,
)
from smiles_processing.smiles_tokenizer import tokenize_smiles, parse_bracket_atom
from smiles_processing.smiles_parser import parse_smiles
from smiles_processing.smiles_features import extract_features, extract_atom_features, extract_bond_features
from smiles_processing.feature_encoding import (
    encode_atom,
    encode_bond,
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
)
from smiles_processing.smiles_to_pyg import smiles_to_pyg, batch_smiles_to_pyg

__all__ = [
    # Errors
    "SMILESError",
    "SMILESTokenizationError",
    "SMILESParseError",
    "SMILESValidationError",
    "UnsupportedSMILESFeatureError",
    # Tokenizer
    "tokenize_smiles",
    "parse_bracket_atom",
    # Parser
    "parse_smiles",
    # Feature extraction
    "extract_features",
    "extract_atom_features",
    "extract_bond_features",
    # Encoding
    "encode_atom",
    "encode_bond",
    "ATOM_FEATURE_DIM",
    "BOND_FEATURE_DIM",
    # PyG conversion
    "smiles_to_pyg",
    "batch_smiles_to_pyg",
]

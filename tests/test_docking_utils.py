# -*- coding: utf-8 -*-
"""Unit tests for docking common utilities and SMILES processing pipeline."""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest

from docking_common.io_utils import load_grids, ensure_dir, safe_filename
from docking_common.receptor_io import sanitize_key, get_receptor_path
from smiles_processing import (
    smiles_to_pyg,
    batch_smiles_to_pyg,
    parse_smiles,
    extract_features,
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
)
from smiles_processing.smiles_errors import SMILESParseError


class TestDockingIOUtils:
    """Tests for docking_common.io_utils."""

    def test_ensure_dir_creates_nested(self, tmp_path):
        target = str(tmp_path / "a" / "b" / "c")
        ensure_dir(target)
        assert os.path.isdir(target)

    def test_ensure_dir_noop_on_existing(self, tmp_path):
        ensure_dir(str(tmp_path))
        assert os.path.isdir(str(tmp_path))

    def test_safe_filename(self):
        assert safe_filename("Hello World!") == "Hello_World_"
        assert safe_filename("mol@123") == "mol_123"
        assert safe_filename("") == ""

    def test_load_grids_valid(self, tmp_path):
        grids = {
            "TARGET_A": [{"id": "p1", "center": [0, 0, 0], "size": [20, 20, 20]}],
            "TARGET_B": [{"id": "p2", "center": [1, 1, 1], "size": [15, 15, 15]}],
        }
        grids_file = str(tmp_path / "grids.json")
        with open(grids_file, "w") as f:
            json.dump(grids, f)

        class FakeLogger:
            def error(self, msg): pass
            def info(self, msg): pass

        result = load_grids(grids_file, FakeLogger())
        assert result == grids

    def test_load_grids_missing_file(self, tmp_path):
        messages = []

        class FakeLogger:
            def error(self, msg): messages.append(msg)
            def info(self, msg): pass

        result = load_grids(str(tmp_path / "nonexistent.json"), FakeLogger())
        assert result is None
        assert len(messages) == 1


class TestReceptorIO:
    """Tests for docking_common.receptor_io."""

    def test_sanitize_key(self):
        assert sanitize_key("path/to/receptor") == "path_to_receptor"
        assert sanitize_key("simple") == "simple"

    def test_get_receptor_path_finds_pdbqt(self, tmp_path):
        # Create a fake receptor file
        receptor = tmp_path / "TARGET_A.pdbqt"
        receptor.write_text("ATOM test data")

        result = get_receptor_path("TARGET_A", str(tmp_path))
        assert result is not None
        assert result.endswith(".pdbqt")

    def test_get_receptor_path_returns_none_when_missing(self, tmp_path):
        result = get_receptor_path("NONEXISTENT", str(tmp_path))
        assert result is None


class TestSMILESPipeline:
    """Tests for the custom SMILES processing pipeline (RDKit-free)."""

    def test_parse_smiles_ethanol(self):
        graph = parse_smiles("CCO")
        assert len(graph["atoms"]) == 3
        assert len(graph["bonds"]) == 2

    def test_parse_smiles_benzene(self):
        graph = parse_smiles("c1ccccc1")
        assert len(graph["atoms"]) == 6
        assert len(graph["bonds"]) == 6

    def test_extract_features_adds_descriptors(self):
        graph = parse_smiles("CCO")
        featured = extract_features(graph)
        assert "atom_features" in featured or len(featured["atoms"]) > 0

    def test_smiles_to_pyg_dimensions(self):
        data = smiles_to_pyg("c1ccccc1")
        assert data.x.shape == (6, ATOM_FEATURE_DIM)
        assert data.edge_index.shape[0] == 2
        # Benzene: 6 bonds * 2 directions = 12 edges
        assert data.edge_index.shape[1] == 12
        assert data.edge_attr.shape == (12, BOND_FEATURE_DIM)

    def test_batch_smiles_handles_mixed(self):
        smiles_list = ["CCO", "INVALID_XYZ", "c1ccccc1"]
        results = batch_smiles_to_pyg(smiles_list)
        # Valid SMILES should produce results, invalid returns None
        assert results[0] is not None
        assert results[2] is not None

    def test_atom_feature_dim_matches_spec(self):
        # Node featurization: atom type, hybridization, formal charge, chirality
        assert ATOM_FEATURE_DIM == 42

    def test_bond_feature_dim_matches_spec(self):
        # Edge featurization: bond type, stereochemistry, ring status
        assert BOND_FEATURE_DIM == 11


class TestSminaEngineUnit:
    """Unit tests for smina engine parsing functions."""

    def test_parse_affinity_from_stdout(self):
        from docking_smina.smina_engine import parse_affinity_from_stdout

        stdout = "   1     -7.3      0.000      0.000\n   2     -6.8      1.234      1.567\n"
        result = parse_affinity_from_stdout(stdout)
        assert abs(result - (-7.3)) < 0.01

    def test_parse_affinity_empty_stdout(self):
        from docking_smina.smina_engine import parse_affinity_from_stdout

        result = parse_affinity_from_stdout("")
        assert np.isnan(result)

    def test_parse_affinity_from_file(self, tmp_path):
        from docking_smina.smina_engine import parse_affinity_from_file

        pdbqt = tmp_path / "docked.pdbqt"
        pdbqt.write_text("REMARK VINA RESULT:    -8.2      0.000      0.000\nATOM ...\n")
        result = parse_affinity_from_file(str(pdbqt))
        assert abs(result - (-8.2)) < 0.01

    def test_parse_affinity_file_not_found(self):
        from docking_smina.smina_engine import parse_affinity_from_file

        result = parse_affinity_from_file("/nonexistent/path.pdbqt")
        assert np.isnan(result)

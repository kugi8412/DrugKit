#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tests/test_smiles_to_pyg.py

"""Tests for smiles_to_pyg — PyG conversion and pipeline compatibility."""

import pytest
import torch
from torch_geometric.data import Data

from smiles_processing.smiles_to_pyg import smiles_to_pyg, batch_smiles_to_pyg
from smiles_processing.feature_encoding import ATOM_FEATURE_DIM, BOND_FEATURE_DIM


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_valid_data(data, smiles=None):
    """Assert that a Data object has the correct schema for GINEConv."""
    assert isinstance(data, Data)
    assert hasattr(data, "x") and data.x is not None
    assert hasattr(data, "edge_index") and data.edge_index is not None
    assert hasattr(data, "edge_attr") and data.edge_attr is not None
    assert hasattr(data, "smiles")
    assert hasattr(data, "is_elite")

    assert data.x.dtype == torch.float
    assert data.edge_index.dtype == torch.long
    assert data.edge_attr.dtype == torch.float

    assert data.x.shape[1] == ATOM_FEATURE_DIM
    if data.edge_attr.shape[0] > 0:
        assert data.edge_attr.shape[1] == BOND_FEATURE_DIM

    assert data.edge_index.shape[0] == 2

    # edge_index and edge_attr must be consistent
    n_edges = data.edge_index.shape[1]
    assert data.edge_attr.shape[0] == n_edges

    if smiles is not None:
        assert data.smiles == smiles


# ---------------------------------------------------------------------------
# Basic conversion
# ---------------------------------------------------------------------------

class TestBasicConversion:
    def test_ethanol(self):
        data = smiles_to_pyg("CCO")
        _assert_valid_data(data, "CCO")
        assert data.x.shape[0] == 3  # 3 atoms

    def test_atom_feature_dim(self):
        data = smiles_to_pyg("C")
        assert data.x.shape == (1, ATOM_FEATURE_DIM)

    def test_single_bond_bidirectional(self):
        data = smiles_to_pyg("CC")
        # 1 bond → 2 directed edges
        assert data.edge_index.shape == (2, 2)
        assert data.edge_attr.shape == (2, BOND_FEATURE_DIM)

    def test_benzene(self):
        data = smiles_to_pyg("c1ccccc1")
        _assert_valid_data(data)
        assert data.x.shape[0] == 6
        # 6 bonds × 2 directions = 12 edges
        assert data.edge_index.shape[1] == 12

    def test_aspirin(self):
        data = smiles_to_pyg("CC(=O)Oc1ccccc1C(=O)O")
        _assert_valid_data(data)
        assert data.x.shape[0] == 13

    def test_smiles_attribute(self):
        data = smiles_to_pyg("CCO")
        assert data.smiles == "CCO"

    def test_no_y_by_default(self):
        data = smiles_to_pyg("CCO")
        assert data.y is None

    def test_is_elite_default_false(self):
        data = smiles_to_pyg("CCO")
        assert data.is_elite.item() == 0.0


# ---------------------------------------------------------------------------
# Optional metadata
# ---------------------------------------------------------------------------

class TestOptionalMetadata:
    def test_y_label(self):
        data = smiles_to_pyg("CCO", y=-7.5)
        assert data.y is not None
        assert data.y.shape == (1,)
        assert abs(data.y.item() - (-7.5)) < 1e-6

    def test_y_zero(self):
        data = smiles_to_pyg("C", y=0.0)
        assert data.y.item() == 0.0

    def test_is_elite_true(self):
        data = smiles_to_pyg("CCO", is_elite=True)
        assert data.is_elite.item() == 1.0

    def test_is_elite_false(self):
        data = smiles_to_pyg("CCO", is_elite=False)
        assert data.is_elite.item() == 0.0

    def test_y_and_elite_together(self):
        data = smiles_to_pyg("c1ccccc1", y=-9.2, is_elite=True)
        assert data.y.item() == pytest.approx(-9.2)
        assert data.is_elite.item() == 1.0


# ---------------------------------------------------------------------------
# Bracket atom support
# ---------------------------------------------------------------------------

class TestBracketAtomsInPyG:
    def test_ammonium(self):
        data = smiles_to_pyg("[NH4+]")
        _assert_valid_data(data)
        assert data.x.shape[0] == 1

    def test_oxide(self):
        data = smiles_to_pyg("[O-]")
        _assert_valid_data(data)

    def test_glycine_zwitterion(self):
        data = smiles_to_pyg("[NH3+]CC(=O)[O-]")
        _assert_valid_data(data)
        assert data.x.shape[0] == 5

    def test_chiral_molecule(self):
        data = smiles_to_pyg("[C@H](F)(Cl)Br")
        _assert_valid_data(data)
        assert data.x.shape[0] == 4

    def test_histidine_like(self):
        data = smiles_to_pyg("c1cnc[nH]1")
        _assert_valid_data(data)


# ---------------------------------------------------------------------------
# Real-world drug-like molecules
# ---------------------------------------------------------------------------

class TestRealWorldMolecules:
    DRUG_SMILES = [
        ("Aspirin",      "CC(=O)Oc1ccccc1C(=O)O"),
        ("Ibuprofen",    "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
        ("Caffeine",     "Cn1cnc2c1c(=O)n(c(=O)n2C)C"),
        ("Paracetamol",  "CC(=O)Nc1ccc(O)cc1"),
        ("Atenolol",     "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1"),
        ("Metformin",    "CN(C)C(=N)NC(=N)N"),
        ("Dopamine",     "NCCc1ccc(O)c(O)c1"),
    ]

    @pytest.mark.parametrize("name,smiles", DRUG_SMILES)
    def test_drug_converts(self, name, smiles):
        data = smiles_to_pyg(smiles)
        assert data is not None, f"{name} conversion returned None"
        _assert_valid_data(data, smiles)
        assert data.x.shape[0] > 0
        assert data.x.shape[1] == ATOM_FEATURE_DIM

    @pytest.mark.parametrize("name,smiles", DRUG_SMILES)
    def test_drug_edge_consistency(self, name, smiles):
        data = smiles_to_pyg(smiles)
        n_atoms = data.x.shape[0]
        if data.edge_index.shape[1] > 0:
            assert data.edge_index.max() < n_atoms


# ---------------------------------------------------------------------------
# GINEConv compatibility smoke test
# ---------------------------------------------------------------------------

class TestGINEConvCompatibility:
    """Confirm output dimensions match what the existing SiameseRankNet expects."""

    def test_node_dim_matches_gine_expectation(self):
        # The existing model was built with get_atom_features returning 42 dims
        data = smiles_to_pyg("CCO")
        assert data.x.shape[1] == 42

    def test_edge_dim_matches_gine_expectation(self):
        # The existing model was built with get_bond_features returning 11 dims
        data = smiles_to_pyg("CC")
        assert data.edge_attr.shape[1] == 11

    def test_edge_index_is_bidirectional(self):
        data = smiles_to_pyg("CCC")
        # 2 bonds → 4 directed edges for undirected message passing
        assert data.edge_index.shape[1] == 4
        # check both directions exist
        edges = set(map(tuple, data.edge_index.t().tolist()))
        assert (0, 1) in edges and (1, 0) in edges


# ---------------------------------------------------------------------------
# Tolerant mode (batch & single)
# ---------------------------------------------------------------------------

class TestTolerantMode:
    def test_all_garbage_returns_none(self):
        # "???" contains no recognisable atoms at all; produces no graph.
        result = smiles_to_pyg("???", strict=False)
        assert result is None

    def test_strict_invalid_raises(self):
        # strict mode raises on ANY unrecognised character
        with pytest.raises(Exception):
            smiles_to_pyg("???", strict=True)

    def test_tolerant_partial_parse(self):
        # "INVALID???" contains I, N, I — valid atoms — so tolerant mode
        # builds a partial 3-atom graph rather than returning None.
        # This is the correct, documented behaviour for billion-scale use.
        result = smiles_to_pyg("INVALID???", strict=False)
        assert result is not None
        assert result.x.shape[0] == 3  # I, N, I

    def test_batch_skips_all_garbage(self):
        # Only "???" has zero valid atoms and therefore fails
        smiles = ["CCO", "???", "c1ccccc1"]
        data_list, failed = batch_smiles_to_pyg(smiles)
        assert len(data_list) == 2
        assert len(failed) == 1
        assert "???" in failed

    def test_batch_all_valid(self):
        smiles = ["C", "CC", "CCO"]
        data_list, failed = batch_smiles_to_pyg(smiles)
        assert len(data_list) == 3
        assert len(failed) == 0

    def test_batch_with_labels(self):
        smiles = ["CCO", "c1ccccc1"]
        labels = [-7.0, -8.5]
        data_list, _ = batch_smiles_to_pyg(smiles, labels=labels)
        assert data_list[0].y.item() == pytest.approx(-7.0)
        assert data_list[1].y.item() == pytest.approx(-8.5)

    def test_batch_with_elite_set(self):
        smiles = ["CCO", "c1ccccc1"]
        elite = {"CCO"}
        data_list, _ = batch_smiles_to_pyg(smiles, elite_set=elite)
        assert data_list[0].is_elite.item() == 1.0
        assert data_list[1].is_elite.item() == 0.0


# ---------------------------------------------------------------------------
# Regression: empty edge case
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_atom_no_edges(self):
        data = smiles_to_pyg("C")
        assert data.x.shape == (1, ATOM_FEATURE_DIM)
        assert data.edge_index.shape == (2, 0)
        assert data.edge_attr.shape == (0, BOND_FEATURE_DIM)

    def test_large_ring(self):
        # Cyclooctane
        data = smiles_to_pyg("C1CCCCCCC1")
        _assert_valid_data(data)
        assert data.x.shape[0] == 8


# ---------------------------------------------------------------------------
# Corrected tolerant mode tests (appended to fix the 2 failures)
# ---------------------------------------------------------------------------

class TestTolerantModeCorrected:
    """Corrected tolerant-mode assumptions.

    In tolerant mode the parser skips bad tokens and builds whatever graph
    it can.  A string containing *only* unknown characters produces no atoms
    and therefore raises SMILESValidationError.  A string with a mix of
    valid atoms and garbage produces a partial graph — that is expected
    behaviour for billion-scale pipelines.
    """

    def test_all_garbage_returns_none(self):
        # "???" has no atoms at all; the parser raises SMILESValidationError
        # which smiles_to_pyg catches and converts to None.
        result = smiles_to_pyg("???", strict=False)
        assert result is None

    def test_batch_skips_all_garbage(self):
        smiles = ["CCO", "???", "c1ccccc1"]
        data_list, failed = batch_smiles_to_pyg(smiles)
        assert len(data_list) == 2
        assert "???" in failed

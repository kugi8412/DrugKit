#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tests/test_regression.py

"""Regression and corpus tests — real-world SMILES, encoding consistency.

These tests lock in the exact atom/bond counts and feature vectors for a
curated set of well-known molecules so that future refactors cannot silently
change the graph topology or encoding layout.
"""

import pytest
from smiles_processing.smiles_parser import parse_smiles
from smiles_processing.smiles_features import extract_features
from smiles_processing.feature_encoding import (
    encode_atom, encode_bond,
    ATOM_FEATURE_DIM, BOND_FEATURE_DIM,
)
from smiles_processing.smiles_to_pyg import smiles_to_pyg, batch_smiles_to_pyg


# ---------------------------------------------------------------------------
# Topology regression table
# ---------------------------------------------------------------------------
# (name, smiles, expected_atoms, expected_bonds)
TOPOLOGY_CORPUS = [
    ("methane",      "C",                         1,  0),
    ("ethane",       "CC",                         2,  1),
    ("ethanol",      "CCO",                        3,  2),
    ("acetone",      "CC(=O)C",                    4,  3),
    ("acetic_acid",  "CC(=O)O",                    4,  3),
    ("benzene",      "c1ccccc1",                   6,  6),
    ("pyridine",     "c1ccncc1",                   6,  6),
    ("toluene",      "Cc1ccccc1",                  7,  7),
    ("naphthalene",  "c1ccc2ccccc2c1",            10, 11),
    ("cyclohexane",  "C1CCCCC1",                   6,  6),
    ("aniline",      "Nc1ccccc1",                  7,  7),
    ("phenol",       "Oc1ccccc1",                  7,  7),
    ("aspirin",      "CC(=O)Oc1ccccc1C(=O)O",     13, 13),
    ("paracetamol",  "CC(=O)Nc1ccc(O)cc1",        11, 11),
    ("caffeine",     "Cn1cnc2c1c(=O)n(c(=O)n2C)C", 14, 15),
]


class TestTopologyRegression:
    @pytest.mark.parametrize("name,smiles,n_atoms,n_bonds", TOPOLOGY_CORPUS)
    def test_atom_count(self, name, smiles, n_atoms, n_bonds):
        g = parse_smiles(smiles)
        assert len(g["atoms"]) == n_atoms, (
            f"{name}: expected {n_atoms} atoms, got {len(g['atoms'])}"
        )

    @pytest.mark.parametrize("name,smiles,n_atoms,n_bonds", TOPOLOGY_CORPUS)
    def test_bond_count(self, name, smiles, n_atoms, n_bonds):
        g = parse_smiles(smiles)
        assert len(g["bonds"]) == n_bonds, (
            f"{name}: expected {n_bonds} bonds, got {len(g['bonds'])}"
        )

    @pytest.mark.parametrize("name,smiles,n_atoms,n_bonds", TOPOLOGY_CORPUS)
    def test_pyg_node_count(self, name, smiles, n_atoms, n_bonds):
        data = smiles_to_pyg(smiles)
        assert data.x.shape[0] == n_atoms

    @pytest.mark.parametrize("name,smiles,n_atoms,n_bonds", TOPOLOGY_CORPUS)
    def test_pyg_edge_count(self, name, smiles, n_atoms, n_bonds):
        data = smiles_to_pyg(smiles)
        # bidirectional: 2 * n_bonds
        assert data.edge_index.shape[1] == 2 * n_bonds


# ---------------------------------------------------------------------------
# Encoding stability — spot-check specific feature vector positions
# ---------------------------------------------------------------------------

class TestEncodingStability:
    """Lock in known feature vector values for simple molecules."""

    def test_methane_carbon_symbol(self):
        g = parse_smiles("C")
        extract_features(g)
        feats = encode_atom(g["atoms"][0], g)
        # C is at index 0 in PERMITTED_ATOMS
        assert feats[0] == 1.0
        assert feats[1] == 0.0  # N should be 0

    def test_ethene_double_bond_flag(self):
        g = parse_smiles("C=C")
        extract_features(g)
        feats = encode_bond(g["bonds"][0])
        assert feats[0] == 0.0  # SINGLE
        assert feats[1] == 1.0  # DOUBLE

    def test_benzene_aromatic_bond_flag(self):
        g = parse_smiles("c1ccccc1")
        extract_features(g)
        for bond in g["bonds"]:
            feats = encode_bond(bond)
            assert feats[3] == 1.0  # AROMATIC

    def test_cyclohexane_in_ring_flag(self):
        g = parse_smiles("C1CCCCC1")
        extract_features(g)
        for bond in g["bonds"]:
            feats = encode_bond(bond)
            assert feats[5] == 1.0  # in_ring

    def test_encoding_sum_is_reasonable(self):
        """Feature vectors should have at least a few non-zero entries."""
        g = parse_smiles("CCO")
        extract_features(g)
        for atom in g["atoms"]:
            feats = encode_atom(atom, g)
            nonzero = sum(1 for v in feats if v != 0.0)
            assert nonzero >= 4, f"Atom encoding suspiciously sparse: {feats}"


# ---------------------------------------------------------------------------
# Batch corpus — real-world SMILES from ZINC/ChEMBL style
# ---------------------------------------------------------------------------

ZINC_SAMPLE = [
    "CC(C)NCC(O)COc1ccc(CC(N)=O)cc1",       # atenolol
    "COc1ccc2[nH]cc(CCNC(C)=O)c2c1",        # melatonin
    "O=C(O)c1ccccc1OC(C)=O",                # aspirin alt
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # testosterone-like
    "Cn1c(=O)c2c(ncn2C)n(c1=O)C",           # caffeine alt
    "CC(=O)Nc1ccc(O)cc1",                    # paracetamol
    "c1ccc(NCc2ccccn2)cc1",                  # phenylpyridine amine
    "O=C([O-])c1ccccc1",                     # benzoate anion
    "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", # glucose
    "[NH3+]CC(=O)[O-]",                      # glycine zwitterion
]


class TestZincCorpus:
    def test_all_convert(self):
        data_list, failed = batch_smiles_to_pyg(ZINC_SAMPLE, strict=False)
        # Allow up to 2 failures for complex stereochemistry
        assert len(failed) <= 2, f"Too many failures: {failed}"
        assert len(data_list) >= len(ZINC_SAMPLE) - 2

    def test_all_correct_dims(self):
        data_list, _ = batch_smiles_to_pyg(ZINC_SAMPLE, strict=False)
        for data in data_list:
            assert data.x.shape[1] == ATOM_FEATURE_DIM
            if data.edge_attr.shape[0] > 0:
                assert data.edge_attr.shape[1] == BOND_FEATURE_DIM

    def test_edge_index_valid(self):
        data_list, _ = batch_smiles_to_pyg(ZINC_SAMPLE, strict=False)
        for data in data_list:
            n_atoms = data.x.shape[0]
            if data.edge_index.shape[1] > 0:
                assert data.edge_index.max() < n_atoms
                assert data.edge_index.min() >= 0


# ---------------------------------------------------------------------------
# Idempotency — parsing twice gives identical tensors
# ---------------------------------------------------------------------------

class TestIdempotency:
    MOLECULES = [
        "CCO", "c1ccccc1", "CC(=O)O",
        "[NH4+]", "[C@H](F)(Cl)Br", "[NH3+]CC(=O)[O-]",
    ]

    @pytest.mark.parametrize("smiles", MOLECULES)
    def test_pyg_idempotent(self, smiles):
        d1 = smiles_to_pyg(smiles)
        d2 = smiles_to_pyg(smiles)
        assert d1.x.tolist() == d2.x.tolist()
        assert d1.edge_index.tolist() == d2.edge_index.tolist()
        assert d1.edge_attr.tolist() == d2.edge_attr.tolist()

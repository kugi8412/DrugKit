#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# tests/test_feature_encoding.py
"""Tests for feature_encoding — dimension consistency and RDKit parity."""

import pytest
from smiles_processing.smiles_parser import parse_smiles
from smiles_processing.smiles_features import extract_features
from smiles_processing.feature_encoding import (
    encode_atom,
    encode_bond,
    ATOM_FEATURE_DIM,
    BOND_FEATURE_DIM,
    PERMITTED_ATOMS,
    FORMAL_CHARGES,
    HYBRIDIZATIONS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _graph(smiles):
    g = parse_smiles(smiles)
    extract_features(g)
    return g


# ---------------------------------------------------------------------------
# Dimension tests
# ---------------------------------------------------------------------------

class TestDimensions:
    """All encoding vectors must have fixed, correct lengths."""

    def test_atom_dim_constant(self):
        assert ATOM_FEATURE_DIM == 42

    def test_bond_dim_constant(self):
        assert BOND_FEATURE_DIM == 11

    @pytest.mark.parametrize("smiles", [
        "C", "CC", "C=C", "C#N", "CCO", "c1ccccc1",
        "CC(=O)O", "C1CCCCC1", "[NH4+]", "[O-]", "c1cc[nH]cc1",
        "[C@H](F)(Cl)Br", "[Fe+2]", "[NH3+]CC(=O)[O-]",
    ])
    def test_atom_encoding_length(self, smiles):
        g = _graph(smiles)
        for atom in g["atoms"]:
            feats = encode_atom(atom, g)
            assert len(feats) == ATOM_FEATURE_DIM, (
                f"Atom dim mismatch for {smiles}: got {len(feats)}"
            )

    @pytest.mark.parametrize("smiles", [
        "CC", "C=C", "C#N", "c1ccccc1", "CC(=O)O",
        "C1CCCCC1", "F/C=C/F",
    ])
    def test_bond_encoding_length(self, smiles):
        g = _graph(smiles)
        for bond in g["bonds"]:
            feats = encode_bond(bond)
            assert len(feats) == BOND_FEATURE_DIM, (
                f"Bond dim mismatch for {smiles}: got {len(feats)}"
            )

    def test_atom_floats(self):
        g = _graph("CCO")
        for atom in g["atoms"]:
            feats = encode_atom(atom, g)
            assert all(isinstance(v, float) for v in feats)

    def test_bond_floats(self):
        g = _graph("C=C")
        feats = encode_bond(g["bonds"][0])
        assert all(isinstance(v, float) for v in feats)


# ---------------------------------------------------------------------------
# Atom one-hot correctness
# ---------------------------------------------------------------------------

class TestAtomEncoding:
    def test_carbon_symbol_onehot(self):
        g = _graph("C")
        feats = encode_atom(g["atoms"][0], g)
        # C is index 0 in PERMITTED_ATOMS
        assert feats[0] == 1.0
        # all other symbol positions should be 0
        n_symbols = len(PERMITTED_ATOMS) + 1
        for i in range(1, n_symbols):
            assert feats[i] == 0.0

    def test_nitrogen_symbol_onehot(self):
        g = _graph("N")
        feats = encode_atom(g["atoms"][0], g)
        n_idx = PERMITTED_ATOMS.index("N")
        assert feats[n_idx] == 1.0

    def test_unknown_element_uses_unknown_bucket(self):
        # Fe is not in PERMITTED_ATOMS — should activate the unknown bucket
        g = _graph("[Fe+2]")
        feats = encode_atom(g["atoms"][0], g)
        n_symbols = len(PERMITTED_ATOMS) + 1
        # unknown bucket is last = index n_symbols - 1
        assert feats[n_symbols - 1] == 1.0

    def test_aromatic_flag(self):
        g = _graph("c1ccccc1")
        for atom in g["atoms"]:
            feats = encode_atom(atom, g)
            # aromatic flag is at index 37
            assert feats[37] == 1.0

    def test_non_aromatic_flag(self):
        g = _graph("CC")
        for atom in g["atoms"]:
            feats = encode_atom(atom, g)
            assert feats[37] == 0.0

    def test_formal_charge_minus1(self):
        g = _graph("[O-]")
        feats = encode_atom(g["atoms"][0], g)
        # formal charge section starts at index 13+6+6=25
        charge_start = len(PERMITTED_ATOMS) + 1 + 6 + 6
        fc_idx = FORMAL_CHARGES.index(-1)
        assert feats[charge_start + fc_idx] == 1.0

    def test_formal_charge_plus1(self):
        g = _graph("[NH4+]")
        feats = encode_atom(g["atoms"][0], g)
        charge_start = len(PERMITTED_ATOMS) + 1 + 6 + 6
        fc_idx = FORMAL_CHARGES.index(1)
        assert feats[charge_start + fc_idx] == 1.0

    def test_mass_normalised(self):
        g = _graph("C")
        feats = encode_atom(g["atoms"][0], g)
        # mass is at index 38 (after aromatic flag)
        mass_val = feats[38]
        # carbon mass ≈ 12.011 * 0.01 ≈ 0.12
        assert 0.10 < mass_val < 0.15

    def test_chirality_cw(self):
        g = _graph("[C@@H](F)(Cl)Br")
        extract_features(g)
        feats = encode_atom(g["atoms"][0], g)
        # chirality section is at [39:42]
        chirality_vec = feats[39:42]
        # CHI_TETRAHEDRAL_CW maps from "@@"
        assert chirality_vec[0] == 1.0  # CW bucket

    def test_chirality_ccw(self):
        g = _graph("[C@H](F)(Cl)Br")
        extract_features(g)
        feats = encode_atom(g["atoms"][0], g)
        chirality_vec = feats[39:42]
        assert chirality_vec[1] == 1.0  # CCW bucket

    def test_no_chirality_uses_unknown_bucket(self):
        g = _graph("C")
        feats = encode_atom(g["atoms"][0], g)
        assert feats[41] == 1.0  # unknown/none chirality bucket


# ---------------------------------------------------------------------------
# Bond encoding correctness
# ---------------------------------------------------------------------------

class TestBondEncoding:
    def test_single_bond(self):
        g = _graph("CC")
        feats = encode_bond(g["bonds"][0])
        assert feats[0] == 1.0  # SINGLE
        assert feats[1] == 0.0  # DOUBLE
        assert feats[2] == 0.0  # TRIPLE
        assert feats[3] == 0.0  # AROMATIC

    def test_double_bond(self):
        g = _graph("C=C")
        feats = encode_bond(g["bonds"][0])
        assert feats[1] == 1.0  # DOUBLE

    def test_triple_bond(self):
        g = _graph("C#C")
        feats = encode_bond(g["bonds"][0])
        assert feats[2] == 1.0  # TRIPLE

    def test_aromatic_bond(self):
        g = _graph("c1ccccc1")
        feats = encode_bond(g["bonds"][0])
        assert feats[3] == 1.0  # AROMATIC

    def test_conjugated_flag(self):
        g = _graph("C=C")
        feats = encode_bond(g["bonds"][0])
        assert feats[4] == 1.0  # conjugated

    def test_in_ring_flag(self):
        g = _graph("C1CCCCC1")
        for bond in g["bonds"]:
            feats = encode_bond(bond)
            assert feats[5] == 1.0  # in_ring

    def test_not_in_ring(self):
        g = _graph("CC")
        feats = encode_bond(g["bonds"][0])
        assert feats[5] == 0.0

    def test_stereo_none_uses_unknown_bucket(self):
        g = _graph("CC")
        feats = encode_bond(g["bonds"][0])
        # stereo section [6:11], last (index 10) = unknown/none
        assert feats[10] == 1.0

    def test_stereo_bond_encoded(self):
        g = _graph("F/C=C/F")
        stereo_bonds = [b for b in g["bonds"] if b["stereochemistry"] is not None]
        for bond in stereo_bonds:
            feats = encode_bond(bond)
            # at least one stereo bucket should be active
            assert sum(feats[6:10]) == 1.0


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    """Same SMILES must produce exactly the same vectors every time."""

    @pytest.mark.parametrize("smiles", [
        "CCO", "c1ccccc1", "CC(=O)O", "[NH4+]", "[C@H](F)(Cl)Br",
    ])
    def test_atom_encoding_deterministic(self, smiles):
        g1 = _graph(smiles)
        g2 = _graph(smiles)
        for a1, a2 in zip(g1["atoms"], g2["atoms"]):
            assert encode_atom(a1, g1) == encode_atom(a2, g2)

    @pytest.mark.parametrize("smiles", ["CC", "C=C", "c1ccccc1"])
    def test_bond_encoding_deterministic(self, smiles):
        g1 = _graph(smiles)
        g2 = _graph(smiles)
        for b1, b2 in zip(g1["bonds"], g2["bonds"]):
            assert encode_bond(b1) == encode_bond(b2)

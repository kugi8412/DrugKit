#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for smiles_tokenizer — including bracket atom support and tolerant mode.
"""

import pytest
from smiles_processing.smiles_tokenizer import tokenize_smiles, parse_bracket_atom
from smiles_processing.smiles_errors import (
    SMILESTokenizationError,
    SMILESValidationError,
    UnsupportedSMILESFeatureError,
)


class TestBasicTokenization:
    def test_simple_chain(self):
        assert tokenize_smiles("CC") == ["C", "C"]

    def test_with_branch_and_double_bond(self):
        assert tokenize_smiles("CC(=O)O") == ["C", "C", "(", "=", "O", ")", "O"]

    def test_aromatic_ring(self):
        assert tokenize_smiles("c1ccccc1") == ["c", "1", "c", "c", "c", "c", "c", "1"]

    def test_two_char_atoms(self):
        toks = tokenize_smiles("ClBrSi")
        assert toks == ["Cl", "Br", "Si"]

    def test_triple_bond(self):
        assert tokenize_smiles("C#N") == ["C", "#", "N"]

    def test_stereo_bonds(self):
        toks = tokenize_smiles("F/C=C/F")
        assert "/" in toks

    def test_chirality_markers(self):
        # bare @ outside brackets
        toks = tokenize_smiles("N[C@@H](C)C(=O)O")
        bracket_chirals = [t["chirality"] for t in toks if isinstance(t, dict) and t.get("chirality")]
        assert "@@" in bracket_chirals or "@" in bracket_chirals or "@@" in toks or "@" in toks

    def test_ring_digits(self):
        toks = tokenize_smiles("C1CC1")
        assert "1" in toks


class TestBracketAtomTokenization:
    def test_ammonium(self):
        toks = tokenize_smiles("[NH4+]")
        assert len(toks) == 1
        t = toks[0]
        assert isinstance(t, dict)
        assert t["symbol"] == "N"
        assert t["formal_charge"] == 1
        assert t["explicit_h"] == 4

    def test_oxide_anion(self):
        toks = tokenize_smiles("[O-]")
        t = toks[0]
        assert t["symbol"] == "O"
        assert t["formal_charge"] == -1
        assert t["explicit_h"] == 0

    def test_aromatic_nh(self):
        toks = tokenize_smiles("c1cc[nH]cc1")
        bracket = next(t for t in toks if isinstance(t, dict))
        assert bracket["symbol"] == "N"
        assert bracket["aromatic"] is True
        assert bracket["explicit_h"] == 1

    def test_chiral_carbon_with_h(self):
        toks = tokenize_smiles("[C@H](F)(Cl)Br")
        t = toks[0]
        assert t["chirality"] == "@"
        assert t["explicit_h"] == 1

    def test_chiral_carbon_double_at(self):
        toks = tokenize_smiles("[C@@H](O)CC")
        t = toks[0]
        assert t["chirality"] == "@@"

    def test_iron_plus2(self):
        toks = tokenize_smiles("[Fe+2]")
        t = toks[0]
        assert t["symbol"] == "FE"
        assert t["formal_charge"] == 2

    def test_calcium_plus2(self):
        toks = tokenize_smiles("[Ca+2]")
        t = toks[0]
        assert t["formal_charge"] == 2

    def test_isotope_ignored(self):
        # Isotope is parsed but semantics ignored downstream
        toks = tokenize_smiles("[13C]")
        t = toks[0]
        assert t["symbol"] == "C"
        assert t["isotope"] == 13

    def test_charge_double_plus(self):
        toks = tokenize_smiles("[Mg++]")
        t = toks[0]
        assert t["formal_charge"] == 2

    def test_charge_double_minus(self):
        toks = tokenize_smiles("[S--]")
        t = toks[0]
        assert t["formal_charge"] == -2

    def test_bracket_atom_in_chain(self):
        # Mixed bracket and bare atoms
        toks = tokenize_smiles("C[NH3+]")
        assert toks[0] == "C"
        assert isinstance(toks[1], dict)
        assert toks[1]["symbol"] == "N"

    def test_multiple_bracket_atoms(self):
        toks = tokenize_smiles("[Na+].[Cl-]", strict=False)
        bracket_toks = [t for t in toks if isinstance(t, dict)]
        assert len(bracket_toks) == 2


class TestParseBracketAtom:
    def test_nh3_plus(self):
        r = parse_bracket_atom("[NH3+]")
        assert r["symbol"] == "N"
        assert r["explicit_h"] == 3
        assert r["formal_charge"] == 1

    def test_o_minus(self):
        r = parse_bracket_atom("[O-]")
        assert r["formal_charge"] == -1

    def test_no_h_no_charge(self):
        r = parse_bracket_atom("[C]")
        assert r["explicit_h"] == 0
        assert r["formal_charge"] == 0

    def test_h_count_multi(self):
        r = parse_bracket_atom("[CH2]")
        assert r["explicit_h"] == 2

    def test_invalid_bracket_raises(self):
        with pytest.raises(SMILESTokenizationError):
            parse_bracket_atom("[Xx99]")


class TestTolerantMode:
    def test_tolerant_skips_unsupported_char(self):
        # '?' is not a valid SMILES char; strict raises, tolerant skips
        with pytest.raises((SMILESTokenizationError, UnsupportedSMILESFeatureError)):
            tokenize_smiles("CC?CC", strict=True)
        toks = tokenize_smiles("CC?CC", strict=False)
        assert "C" in toks

    def test_tolerant_skips_zero_ring_digit(self):
        with pytest.raises(SMILESTokenizationError):
            tokenize_smiles("C0C", strict=True)
        # tolerant should not crash
        toks = tokenize_smiles("C0C", strict=False)
        assert "C" in toks

    def test_tolerant_returns_partial_for_unclosed_bracket(self):
        # "[NH4+" has no closing bracket
        result = tokenize_smiles("[NH4+", strict=False)
        # Should return empty or partial list without crashing
        assert isinstance(result, list)


class TestStrictErrors:
    def test_empty_string(self):
        with pytest.raises(SMILESValidationError):
            tokenize_smiles("")

    def test_whitespace_only(self):
        with pytest.raises(SMILESValidationError):
            tokenize_smiles("   ")

    def test_disconnected_raises(self):
        with pytest.raises(UnsupportedSMILESFeatureError):
            tokenize_smiles("C.C")

    def test_reaction_raises(self):
        with pytest.raises(UnsupportedSMILESFeatureError):
            tokenize_smiles("C>C>C")

    def test_wildcard_raises(self):
        with pytest.raises(UnsupportedSMILESFeatureError):
            tokenize_smiles("C*C")

    def test_unknown_atom_raises(self):
        with pytest.raises(SMILESTokenizationError):
            tokenize_smiles("CXC")

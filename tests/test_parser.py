# tests/test_parser.py
"""Tests for smiles_parser — graph topology correctness."""

import pytest
from smiles_processing.smiles_parser import parse_smiles
from smiles_processing.smiles_errors import SMILESParseError, SMILESValidationError


# ---------------------------------------------------------------------------
# Graph schema validation helper
# ---------------------------------------------------------------------------

def _validate_graph(g):
    """Assert that a graph has valid schema and consistent atom ids."""
    assert "atoms" in g and "bonds" in g
    for i, atom in enumerate(g["atoms"]):
        assert atom["id"] == i
        assert isinstance(atom["symbol"], str) and len(atom["symbol"]) >= 1
        assert isinstance(atom["aromatic"], bool)
        assert isinstance(atom["formal_charge"], int)
        assert atom["chirality"] in (None, "@", "@@")
    atom_ids = {a["id"] for a in g["atoms"]}
    for bond in g["bonds"]:
        assert bond["start"] in atom_ids
        assert bond["end"] in atom_ids
        assert bond["bond_type"] in {"SINGLE", "DOUBLE", "TRIPLE", "AROMATIC"}
        assert bond["start"] != bond["end"]


# ---------------------------------------------------------------------------
# Basic topology
# ---------------------------------------------------------------------------

class TestBasicTopology:
    def test_ethane(self):
        g = parse_smiles("CC")
        assert len(g["atoms"]) == 2
        assert len(g["bonds"]) == 1
        assert g["bonds"][0]["bond_type"] == "SINGLE"

    def test_ethene(self):
        g = parse_smiles("C=C")
        assert g["bonds"][0]["bond_type"] == "DOUBLE"

    def test_ethyne(self):
        g = parse_smiles("C#C")
        assert g["bonds"][0]["bond_type"] == "TRIPLE"

    def test_schema_valid(self):
        for smi in ["CC", "C=C", "C#N", "c1ccccc1", "CC(=O)O"]:
            _validate_graph(parse_smiles(smi))

    def test_single_atom(self):
        g = parse_smiles("C")
        assert len(g["atoms"]) == 1
        assert len(g["bonds"]) == 0

    def test_atom_count_propane(self):
        g = parse_smiles("CCC")
        assert len(g["atoms"]) == 3
        assert len(g["bonds"]) == 2

    def test_branch(self):
        g = parse_smiles("CC(C)C")
        assert len(g["atoms"]) == 4
        assert len(g["bonds"]) == 3


# ---------------------------------------------------------------------------
# Aromatic systems
# ---------------------------------------------------------------------------

class TestAromaticSystems:
    def test_benzene_bond_types(self):
        g = parse_smiles("c1ccccc1")
        for bond in g["bonds"]:
            assert bond["bond_type"] == "AROMATIC"

    def test_benzene_ring_flag(self):
        g = parse_smiles("c1ccccc1")
        assert all(b["in_ring"] for b in g["bonds"])

    def test_aromatic_atom_flag(self):
        g = parse_smiles("c1ccccc1")
        assert all(a["aromatic"] for a in g["atoms"])

    def test_pyridine(self):
        g = parse_smiles("c1ccncc1")
        n_atoms = [a for a in g["atoms"] if a["symbol"] == "N"]
        assert len(n_atoms) == 1
        assert n_atoms[0]["aromatic"] is True

    def test_naphthalene(self):
        g = parse_smiles("c1ccc2ccccc2c1")
        assert len(g["atoms"]) == 10


# ---------------------------------------------------------------------------
# Ring closures
# ---------------------------------------------------------------------------

class TestRingClosures:
    def test_cyclohexane_all_in_ring(self):
        g = parse_smiles("C1CCCCC1")
        assert all(b["in_ring"] for b in g["bonds"])

    def test_cyclopentane_count(self):
        g = parse_smiles("C1CCCC1")
        assert len(g["atoms"]) == 5
        assert len(g["bonds"]) == 5

    def test_multiple_ring_closures(self):
        # Bicyclo[2.2.1]heptane (norbornane) simplified
        g = parse_smiles("C1CC2CCC1C2")
        assert len(g["atoms"]) == 7

    def test_spiro(self):
        g = parse_smiles("C1CCC2(CC1)CCCC2")
        _validate_graph(g)


# ---------------------------------------------------------------------------
# Bracket atoms
# ---------------------------------------------------------------------------

class TestBracketAtoms:
    def test_nh4_plus_parsed(self):
        g = parse_smiles("[NH4+]")
        assert len(g["atoms"]) == 1
        atom = g["atoms"][0]
        assert atom["symbol"] == "N"
        assert atom["formal_charge"] == 1
        assert atom["explicit_h"] == 4

    def test_oxide_anion(self):
        g = parse_smiles("[O-]")
        assert g["atoms"][0]["formal_charge"] == -1

    def test_nh_aromatic(self):
        g = parse_smiles("c1cc[nH]cc1")
        n_atom = next(a for a in g["atoms"] if a["symbol"] == "N")
        assert n_atom["aromatic"] is True
        assert n_atom["explicit_h"] == 1

    def test_chiral_bracket_at(self):
        g = parse_smiles("[C@H](F)(Cl)Br")
        c_atom = g["atoms"][0]
        assert c_atom["chirality"] == "@"

    def test_chiral_bracket_atat(self):
        g = parse_smiles("[C@@H](O)CC")
        c_atom = g["atoms"][0]
        assert c_atom["chirality"] == "@@"

    def test_iron_center(self):
        g = parse_smiles("[Fe+2]")
        assert g["atoms"][0]["symbol"] == "FE"
        assert g["atoms"][0]["formal_charge"] == 2

    def test_bracket_in_chain(self):
        g = parse_smiles("C[NH3+]")
        assert len(g["atoms"]) == 2
        n_atom = next(a for a in g["atoms"] if a["symbol"] == "N")
        assert n_atom["formal_charge"] == 1

    def test_aspirin_with_brackets(self):
        # Aspirin with explicit O charge variant (CC(=O)Oc1ccccc1C(=O)[O-])
        g = parse_smiles("CC(=O)Oc1ccccc1C(=O)[O-]")
        _validate_graph(g)
        charged = [a for a in g["atoms"] if a["formal_charge"] != 0]
        assert len(charged) == 1
        assert charged[0]["formal_charge"] == -1

    def test_drug_like_molecule(self):
        # Ibuprofen-like: CC(C)Cc1ccc(cc1)C(C)C(=O)O
        g = parse_smiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O")
        _validate_graph(g)

    def test_charged_drug_glycine(self):
        # Glycine zwitterion: [NH3+]CC(=O)[O-]
        g = parse_smiles("[NH3+]CC(=O)[O-]")
        charges = {a["formal_charge"] for a in g["atoms"]}
        assert 1 in charges
        assert -1 in charges


# ---------------------------------------------------------------------------
# Conjugation and ring flags
# ---------------------------------------------------------------------------

class TestPostProcessing:
    def test_ethene_conjugated(self):
        g = parse_smiles("C=C")
        assert g["bonds"][0]["conjugated"] is True

    def test_butadiene_all_conjugated(self):
        g = parse_smiles("C=CC=C")
        # All 3 bonds should be conjugated (two double + the single bridging)
        assert all(b["conjugated"] for b in g["bonds"])

    def test_ethane_not_conjugated(self):
        g = parse_smiles("CC")
        assert g["bonds"][0]["conjugated"] is False

    def test_ring_bond_false_for_chain(self):
        g = parse_smiles("CCC")
        assert all(not b["in_ring"] for b in g["bonds"])


# ---------------------------------------------------------------------------
# Tolerant parsing mode
# ---------------------------------------------------------------------------

class TestTolerantParsing:
    def test_tolerant_returns_partial_graph(self):
        # Contains '?' which is invalid; tolerant should still return a graph
        g = parse_smiles("CC?CC", strict=False)
        assert len(g["atoms"]) >= 2  # at least the Cs parsed

    def test_tolerant_unmatched_paren(self):
        # Unmatched ')' — tolerant should not crash
        g = parse_smiles("CC)C", strict=False)
        assert g is not None
        assert len(g["atoms"]) >= 2

    def test_strict_unmatched_paren_raises(self):
        with pytest.raises(SMILESParseError):
            parse_smiles("CC)C", strict=True)

    def test_strict_unclosed_paren_raises(self):
        with pytest.raises(SMILESParseError):
            parse_smiles("CC(C", strict=True)

    def test_strict_unclosed_ring_raises(self):
        with pytest.raises(SMILESParseError):
            parse_smiles("C1CC", strict=True)

    def test_empty_raises_always(self):
        with pytest.raises((SMILESValidationError, Exception)):
            parse_smiles("", strict=False)


# ---------------------------------------------------------------------------
# Stereo bond tokens
# ---------------------------------------------------------------------------

class TestStereoBonds:
    def test_stereo_recorded(self):
        g = parse_smiles("F/C=C/F")
        stereo_bonds = [b for b in g["bonds"] if b["stereochemistry"] is not None]
        assert len(stereo_bonds) >= 1

    def test_stereo_bond_type_still_single(self):
        g = parse_smiles("F/C=C/F")
        stereo = next(b for b in g["bonds"] if b["stereochemistry"] == "/")
        assert stereo["bond_type"] == "SINGLE"

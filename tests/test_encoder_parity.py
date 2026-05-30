# tests/test_encoder_parity.py
"""
RDKit parity tests: compare our smiles_processing encoder against the
verbatim RDKit-based encoder extracted from the target scripts
(siamese_GNN/improved_train.py and final_docking/virtual_screeing.py).

Dataset:  ESOL (Delaney, 1128 molecules) downloaded from
          deepchem/datasets/delaney-processed.csv on GitHub.
          Filtered to molecules parseable by both encoders with
          matching atom/bond counts (all 1128 pass).

Test strategy
─────────────
1. STRUCTURAL tests — both encoders produce graphs with the same
   atom count, bond count, and node/edge tensor dimensions.
   These MUST pass with zero tolerance.

2. EXACT-MATCH tests — features that must be bit-for-bit identical
   after the two bugs we fixed (symbol case, mass precision):
     • atom symbol one-hot  (pos 0–12)
     • atom degree one-hot  (pos 13–18)
     • atom mass            (pos 38, within 1e-4)
     • atom chirality       (pos 39–41)
     • bond stereo          (pos 6–10, always unknown bucket on both sides)

3. KNOWN-APPROXIMATION tests — features that differ due to documented
   limitations of our heuristic pipeline vs RDKit's full chemistry engine.
   Each test asserts that:
     (a) the divergence rate is below a documented threshold, and
     (b) no previously-unknown feature positions are diverging.
   These tests DOCUMENT the gap and will FAIL if a regression makes it worse.

Known approximations (see smiles_processing_comparison/compare_encoders.py for full explanation):
  Atom: total_h, formal_charge, hybridization, aromatic_flag
  Bond: bond_type, conjugated, in_ring, stereo (E/Z on double bonds)

4. CORPUS-LEVEL tests — aggregate statistics over all 1128 molecules.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

HERE      = Path(__file__).parent
ROOT      = HERE.parent
DATA_DIR  = ROOT / "testing_data"
DATASET   = DATA_DIR / "esol_filtered.csv"

sys.path.insert(0, str(ROOT))

from smiles_processing_comparison.rdkit_encoder import (
    get_raw_atom_features,
    get_raw_bond_features,
    smiles_to_graph_rdkit,
)
from smiles_processing_comparison.compare_encoders import (
    compare_smiles,
    run_corpus,
    ATOM_SLICES,
    BOND_SLICES,
    FLOAT_ATOL,
)
from smiles_processing.smiles_parser import parse_smiles
from smiles_processing.smiles_features import extract_features
from smiles_processing.feature_encoding import encode_atom, encode_bond
from smiles_processing.smiles_to_pyg import smiles_to_pyg


# ---------------------------------------------------------------------------
# Dataset fixture
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def esol_smiles() -> list[str]:
    """Load all 1128 filtered ESOL SMILES once per session."""
    if not DATASET.exists():
        pytest.skip(
            f"ESOL dataset not found at {DATASET}. "
            "Run: python comparison/prepare_dataset.py"
        )
    with open(DATASET) as f:
        smiles = [row["smiles"].strip() for row in csv.DictReader(f)]
    assert len(smiles) >= 1000, f"Expected ≥1000 SMILES, got {len(smiles)}"
    return smiles


@pytest.fixture(scope="session")
def corpus_stats(esol_smiles):
    """Run full corpus comparison once and cache the stats."""
    return run_corpus(DATASET)


# ---------------------------------------------------------------------------
# Spot-check molecules used in per-molecule parametrised tests
# ---------------------------------------------------------------------------

SPOT_CHECK = [
    # (name, smiles)
    ("ethanol",          "CCO"),
    ("acetic_acid",      "CC(=O)O"),
    ("benzene",          "c1ccccc1"),
    ("pyridine",         "c1ccncc1"),
    ("aspirin",          "CC(=O)Oc1ccccc1C(=O)O"),
    ("naphthalene",      "c1ccc2ccccc2c1"),
    ("cyclohexane",      "C1CCCCC1"),
    ("aniline",          "Nc1ccccc1"),
    ("phenol",           "Oc1ccccc1"),
    ("paracetamol",      "CC(=O)Nc1ccc(O)cc1"),
    ("chlorobenzene",    "Clc1ccccc1"),
    ("bromobenzene",     "Brc1ccccc1"),
    ("iodobenzene",      "Ic1ccccc1"),
    ("furan",            "c1ccoc1"),
    ("thiophene",        "c1ccsc1"),
    ("chiral_mol",       "[C@H](F)(Cl)Br"),
    ("glycine_zw",       "[NH3+]CC(=O)[O-]"),
    ("caffeine",         "Cn1cnc2c1c(=O)n(c(=O)n2C)C"),
    ("ibuprofen",        "CC(C)Cc1ccc(cc1)C(C)C(=O)O"),
    ("stereo_E",         "F/C=C/F"),
]


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _our_feats(smiles: str):
    """Return (atom_feats, bond_feats) from our encoder."""
    g = parse_smiles(smiles, strict=True)
    extract_features(g)
    atoms = [encode_atom(a, g) for a in g["atoms"]]
    bonds = [encode_bond(b) for b in g["bonds"]]
    return atoms, bonds


def _slice(vec, sl):
    return vec[sl.start:sl.stop]


def _vecs_close(a, b, atol=FLOAT_ATOL):
    return len(a) == len(b) and all(abs(float(x) - float(y)) <= atol for x, y in zip(a, b))


# ===========================================================================
# 1. STRUCTURAL TESTS — atom/bond counts and tensor dimensions must match
# ===========================================================================

class TestStructuralParity:
    """Both encoders must produce identical graph topology."""

    @pytest.mark.parametrize("name,smiles", SPOT_CHECK)
    def test_atom_count_matches(self, name, smiles):
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        g = parse_smiles(smiles)
        assert mol.GetNumAtoms() == len(g["atoms"]), (
            f"{name}: atom count mismatch rdkit={mol.GetNumAtoms()} ours={len(g['atoms'])}"
        )

    @pytest.mark.parametrize("name,smiles", SPOT_CHECK)
    def test_bond_count_matches(self, name, smiles):
        from rdkit import Chem
        mol = Chem.MolFromSmiles(smiles)
        g = parse_smiles(smiles)
        assert mol.GetNumBonds() == len(g["bonds"]), (
            f"{name}: bond count mismatch rdkit={mol.GetNumBonds()} ours={len(g['bonds'])}"
        )

    @pytest.mark.parametrize("name,smiles", SPOT_CHECK)
    def test_pyg_node_dim_matches(self, name, smiles):
        from rdkit import Chem
        rdkit_data = smiles_to_graph_rdkit(smiles)
        our_data   = smiles_to_pyg(smiles)
        assert rdkit_data.x.shape == our_data.x.shape, (
            f"{name}: x shape mismatch rdkit={rdkit_data.x.shape} ours={our_data.x.shape}"
        )

    @pytest.mark.parametrize("name,smiles", SPOT_CHECK)
    def test_pyg_edge_dim_matches(self, name, smiles):
        rdkit_data = smiles_to_graph_rdkit(smiles)
        our_data   = smiles_to_pyg(smiles)
        assert rdkit_data.edge_index.shape == our_data.edge_index.shape
        assert rdkit_data.edge_attr.shape  == our_data.edge_attr.shape

    def test_corpus_all_topology_matches(self, esol_smiles):
        """All 1128 ESOL molecules must have matching atom/bond counts."""
        from rdkit import Chem
        mismatches = []
        for smi in esol_smiles:
            mol = Chem.MolFromSmiles(smi)
            g = parse_smiles(smi)
            if mol.GetNumAtoms() != len(g["atoms"]) or mol.GetNumBonds() != len(g["bonds"]):
                mismatches.append(smi)
        assert mismatches == [], f"Topology mismatches: {mismatches}"


# ===========================================================================
# 2. EXACT-MATCH TESTS — features that must be bit-for-bit identical
# ===========================================================================

class TestExactMatchFeatures:
    """
    After fixing the two bugs (symbol case, mass precision),
    these feature sections must be identical between encoders.
    """

    @pytest.mark.parametrize("name,smiles", SPOT_CHECK)
    def test_symbol_onehot_exact(self, name, smiles):
        rdkit_a = get_raw_atom_features(smiles)
        our_a, _ = _our_feats(smiles)
        sl = ATOM_SLICES["symbol"]
        for i, (rf, of) in enumerate(zip(rdkit_a, our_a)):
            assert _slice(rf, sl) == [int(v) for v in _slice(of, sl)], (
                f"{name} atom[{i}] symbol mismatch: rdkit={_slice(rf,sl)} ours={_slice(of,sl)}"
            )

    @pytest.mark.parametrize("name,smiles", SPOT_CHECK)
    def test_degree_onehot_exact(self, name, smiles):
        rdkit_a = get_raw_atom_features(smiles)
        our_a, _ = _our_feats(smiles)
        sl = ATOM_SLICES["degree"]
        for i, (rf, of) in enumerate(zip(rdkit_a, our_a)):
            assert _slice(rf, sl) == [int(v) for v in _slice(of, sl)], (
                f"{name} atom[{i}] degree mismatch: rdkit={_slice(rf,sl)} ours={_slice(of,sl)}"
            )

    @pytest.mark.parametrize("name,smiles", SPOT_CHECK)
    def test_mass_within_tolerance(self, name, smiles):
        rdkit_a = get_raw_atom_features(smiles)
        our_a, _ = _our_feats(smiles)
        sl = ATOM_SLICES["mass"]
        for i, (rf, of) in enumerate(zip(rdkit_a, our_a)):
            rdkit_mass = rf[sl.start]
            our_mass   = of[sl.start]
            assert abs(rdkit_mass - our_mass) <= FLOAT_ATOL, (
                f"{name} atom[{i}] mass diff too large: "
                f"rdkit={rdkit_mass:.6f} ours={our_mass:.6f}"
            )

    @pytest.mark.parametrize("name,smiles", SPOT_CHECK)
    def test_chirality_exact(self, name, smiles):
        rdkit_a = get_raw_atom_features(smiles)
        our_a, _ = _our_feats(smiles)
        sl = ATOM_SLICES["chirality"]
        for i, (rf, of) in enumerate(zip(rdkit_a, our_a)):
            assert _slice(rf, sl) == [int(v) for v in _slice(of, sl)], (
                f"{name} atom[{i}] chirality mismatch: rdkit={_slice(rf,sl)} ours={_slice(of,sl)}"
            )

    @pytest.mark.parametrize("name,smiles", SPOT_CHECK)
    def test_bond_stereo_is_unknown_bucket_both_sides(self, name, smiles):
        """
        Both encoders should agree on stereo for the flanking single bonds
        (always unknown bucket), even if the double bond encoding diverges.
        """
        from rdkit import Chem
        rdkit_b = get_raw_bond_features(smiles)
        _, our_b = _our_feats(smiles)
        mol = Chem.MolFromSmiles(smiles)
        sl = BOND_SLICES["stereo"]
        for i, (bond, rf, of) in enumerate(zip(mol.GetBonds(), rdkit_b, our_b)):
            from rdkit import Chem as C
            bt = bond.GetBondType()
            if bt == C.rdchem.BondType.SINGLE:
                # Single bonds adjacent to stereo double bond:
                # RDKit gives STEREONONE → unknown bucket [0,0,0,0,1]
                # We also give unknown bucket → must match
                rdkit_stereo = _slice(rf, sl)
                our_stereo   = [int(v) for v in _slice(of, sl)]
                assert rdkit_stereo == our_stereo, (
                    f"{name} bond[{i}] (SINGLE) stereo mismatch: "
                    f"rdkit={rdkit_stereo} ours={our_stereo}"
                )


# ===========================================================================
# 3. KNOWN-APPROXIMATION TESTS — divergence rates within documented bounds
# ===========================================================================

class TestKnownApproximations:
    """
    Features that diverge due to heuristic vs rigorous chemistry.
    Tests assert:
      (a) the feature positions involved are exactly those documented, and
      (b) divergence rates do not exceed the measured baseline.
    """

    def test_only_documented_atom_features_diverge(self, corpus_stats):
        """Only the four documented atom approximations should diverge."""
        documented = {
            "atom.total_h", "atom.formal_charge",
            "atom.hybridization", "atom.aromatic",
        }
        actual = set(corpus_stats.atom_diffs_by_feature.keys())
        unexpected = actual - documented
        assert unexpected == set(), (
            f"Unexpected atom feature divergences: {unexpected}"
        )

    def test_only_documented_bond_features_diverge(self, corpus_stats):
        """Only the four documented bond approximations should diverge."""
        documented = {
            "bond.bond_type", "bond.conjugated",
            "bond.in_ring", "bond.stereo",
        }
        actual = set(corpus_stats.bond_diffs_by_feature.keys())
        unexpected = actual - documented
        assert unexpected == set(), (
            f"Unexpected bond feature divergences: {unexpected}"
        )

    def test_no_unexpected_full_molecule_diffs(self, corpus_stats):
        """Zero molecules should have diffs outside the documented approximations."""
        assert corpus_stats.unexpected_diff == 0, (
            f"{corpus_stats.unexpected_diff} molecule(s) have unexpected divergences:\n"
            + "\n".join(d.smiles for d in corpus_stats.unexpected_cases)
        )

    # ── Divergence rate regression guards ─────────────────────────────────
    # These catch regressions where a code change makes the approximation worse.
    # Thresholds are set to 110% of the measured baseline to allow minor float
    # variation while catching meaningful regressions.

    def test_total_h_divergence_rate(self, corpus_stats):
        n = corpus_stats.atom_diffs_by_feature.get("atom.total_h", 0)
        # Measured baseline: 2077 atom-level diffs across 14991 atoms
        assert n <= 2300, f"atom.total_h diffs regressed: {n} > threshold 2300"

    def test_formal_charge_divergence_rate(self, corpus_stats):
        n = corpus_stats.atom_diffs_by_feature.get("atom.formal_charge", 0)
        # Measured baseline: 158
        assert n <= 180, f"atom.formal_charge diffs regressed: {n} > threshold 180"

    def test_hybridization_divergence_rate(self, corpus_stats):
        n = corpus_stats.atom_diffs_by_feature.get("atom.hybridization", 0)
        # Measured baseline: 1039
        assert n <= 1150, f"atom.hybridization diffs regressed: {n} > threshold 1150"

    def test_aromatic_divergence_rate(self, corpus_stats):
        n = corpus_stats.atom_diffs_by_feature.get("atom.aromatic", 0)
        # Measured baseline: 16
        assert n <= 25, f"atom.aromatic diffs regressed: {n} > threshold 25"

    def test_bond_type_divergence_rate(self, corpus_stats):
        n = corpus_stats.bond_diffs_by_feature.get("bond.bond_type", 0)
        # Measured baseline: 1803
        assert n <= 2000, f"bond.bond_type diffs regressed: {n} > threshold 2000"

    def test_conjugated_divergence_rate(self, corpus_stats):
        n = corpus_stats.bond_diffs_by_feature.get("bond.conjugated", 0)
        # Measured baseline: 2559
        assert n <= 2820, f"bond.conjugated diffs regressed: {n} > threshold 2820"

    def test_in_ring_divergence_rate(self, corpus_stats):
        n = corpus_stats.bond_diffs_by_feature.get("bond.in_ring", 0)
        # Measured baseline: 3474
        assert n <= 3825, f"bond.in_ring diffs regressed: {n} > threshold 3825"

    def test_stereo_divergence_rate(self, corpus_stats):
        n = corpus_stats.bond_diffs_by_feature.get("bond.stereo", 0)
        # Measured baseline: 6 (only double-bond stereo without context)
        assert n <= 10, f"bond.stereo diffs regressed: {n} > threshold 10"


# ===========================================================================
# 4. CORPUS-LEVEL TESTS — aggregate statistics
# ===========================================================================

class TestCorpusStatistics:
    """High-level assertions about the overall comparison."""

    def test_all_molecules_processed(self, esol_smiles, corpus_stats):
        """Every molecule in the dataset must reach the comparison stage."""
        assert corpus_stats.total_molecules == len(esol_smiles)

    def test_exact_match_rate_above_floor(self, corpus_stats):
        """At least 17% of molecules should be exact matches (all features identical)."""
        assert corpus_stats.exact_match_pct >= 17.0, (
            f"Exact match rate {corpus_stats.exact_match_pct:.1f}% < 17% floor"
        )

    def test_no_unexpected_differences(self, corpus_stats):
        """No molecule should have a difference outside the documented approximations."""
        assert corpus_stats.unexpected_diff == 0

    def test_dimension_invariant_across_corpus(self, esol_smiles):
        """Every molecule must produce 42-dim atom vectors and 11-dim bond vectors."""
        from smiles_processing.feature_encoding import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
        for smi in esol_smiles:
            rdkit_a = get_raw_atom_features(smi)
            our_a, our_b = _our_feats(smi)
            assert all(len(f) == ATOM_FEATURE_DIM for f in our_a), smi
            assert all(len(f) == BOND_FEATURE_DIM for f in our_b), smi

    def test_symbol_exact_across_corpus(self, esol_smiles):
        """Symbol one-hot must be bit-for-bit identical for ALL 1128 molecules."""
        sl = ATOM_SLICES["symbol"]
        failures = []
        for smi in esol_smiles:
            rdkit_a = get_raw_atom_features(smi)
            our_a, _ = _our_feats(smi)
            for i, (rf, of) in enumerate(zip(rdkit_a, our_a)):
                if _slice(rf, sl) != [int(v) for v in _slice(of, sl)]:
                    failures.append((smi, i))
        assert failures == [], f"Symbol mismatches: {failures[:5]}"

    def test_degree_exact_across_corpus(self, esol_smiles):
        """Degree one-hot must be bit-for-bit identical for ALL 1128 molecules."""
        sl = ATOM_SLICES["degree"]
        failures = []
        for smi in esol_smiles:
            rdkit_a = get_raw_atom_features(smi)
            our_a, _ = _our_feats(smi)
            for i, (rf, of) in enumerate(zip(rdkit_a, our_a)):
                if _slice(rf, sl) != [int(v) for v in _slice(of, sl)]:
                    failures.append((smi, i, _slice(rf,sl), [int(v) for v in _slice(of,sl)]))
        assert failures == [], f"Degree mismatches: {failures[:5]}"

    def test_mass_within_tolerance_across_corpus(self, esol_smiles):
        """Mass (pos 38) must agree within 1e-4 for ALL 1128 molecules."""
        sl = ATOM_SLICES["mass"]
        failures = []
        for smi in esol_smiles:
            rdkit_a = get_raw_atom_features(smi)
            our_a, _ = _our_feats(smi)
            for i, (rf, of) in enumerate(zip(rdkit_a, our_a)):
                diff = abs(rf[sl.start] - of[sl.start])
                if diff > FLOAT_ATOL:
                    failures.append((smi, i, rf[sl.start], of[sl.start], diff))
        assert failures == [], f"Mass out-of-tolerance: {failures[:5]}"

    def test_chirality_exact_across_corpus(self, esol_smiles):
        """Chirality one-hot must be bit-for-bit identical for ALL 1128 molecules."""
        sl = ATOM_SLICES["chirality"]
        failures = []
        for smi in esol_smiles:
            rdkit_a = get_raw_atom_features(smi)
            our_a, _ = _our_feats(smi)
            for i, (rf, of) in enumerate(zip(rdkit_a, our_a)):
                if _slice(rf, sl) != [int(v) for v in _slice(of, sl)]:
                    failures.append((smi, i))
        assert failures == [], f"Chirality mismatches: {failures[:5]}"

    def test_pyg_shapes_match_across_corpus(self, esol_smiles):
        """PyG Data shapes must match for all 1128 molecules."""
        from smiles_processing.feature_encoding import ATOM_FEATURE_DIM, BOND_FEATURE_DIM
        failures = []
        for smi in esol_smiles:
            rdkit_d = smiles_to_graph_rdkit(smi)
            our_d   = smiles_to_pyg(smi)
            if rdkit_d is None or our_d is None:
                failures.append((smi, "parse_failed"))
                continue
            if rdkit_d.x.shape != our_d.x.shape:
                failures.append((smi, "x_shape", rdkit_d.x.shape, our_d.x.shape))
            if rdkit_d.edge_attr.shape != our_d.edge_attr.shape:
                failures.append((smi, "edge_attr_shape", rdkit_d.edge_attr.shape, our_d.edge_attr.shape))
        assert failures == [], f"PyG shape mismatches: {failures[:5]}"

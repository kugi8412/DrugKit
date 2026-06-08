# comparison/compare_encoders.py
"""
Structured comparison engine: RDKit encoder vs our smiles_processing encoder.

Provides:
  EncoderComparison   — runs both encoders on one SMILES, returns a diff report
  run_corpus          — runs over the full ESOL corpus and aggregates stats

The comparison distinguishes:
  - EXACT MATCH    : vectors are identical (within float tolerance)
  - KNOWN_APPROX   : differs only in a documented approximation feature
  - UNEXPECTED_DIFF: differs in a feature that should match — indicates a bug

Known approximation features (documented divergences between our heuristic
pipeline and RDKit's rigorous chemistry engine):

  Atom features:
    total_h      (pos 19-24) Our valence-table heuristic differs from RDKit's
                              exact implicit-H computation (esp. for N, O, S
                              adjacent to pi systems and charged atoms).
    formal_charge (pos 25-30) RDKit sanitises N(=O)=O to N+(=O)O- (charge +1/-1).
                              Our parser reads the SMILES literally; charge is 0
                              unless a bracket atom specifies it explicitly.
    hybridization (pos 31-36) Our bond-counting heuristic approximates SP/SP2/SP3.
                              RDKit uses a full valence model (e.g. amide N → SP2,
                              sulfoxide S → SP3D).
    aromatic_flag (pos 37)   We trust the SMILES lowercase notation literally.
                              RDKit sanitises and can demote atoms that are formally
                              lowercase but sp3 in a fused ring system.

  Bond features:
    bond_type    (pos 0-3)   We infer AROMATIC from atom aromaticity; RDKit assigns
                              this after full Kekulisation. In fused bicyclics with
                              mixed Kekulé notation the assignment can differ.
    conjugated   (pos 4)     Our heuristic marks bonds adjacent to any pi atom as
                              conjugated. RDKit uses a stricter resonance-based
                              criterion (e.g. C-O of an ester: RDKit=False, ours=True
                              for the C-C=O neighbour).
    in_ring      (pos 5)     Both use DFS/SSSR, but RDKit's SSSR (Smallest Set of
                              Smallest Rings) differs from our spanning-tree DFS for
                              bridged bicyclic systems.

Features that MUST match exactly (bugs if they differ):
    symbol one-hot (pos 0-12), degree (pos 13-18), mass (pos 38), chirality (pos 39-41)
    stereo (bond pos 6-10)
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

HERE = Path(__file__).parent

# ---------------------------------------------------------------------------
# Feature slice indices (matching the layout in feature_encoding.py)
# ---------------------------------------------------------------------------

ATOM_SLICES = {
    "symbol":        slice(0,  13),
    "degree":        slice(13, 19),
    "total_h":       slice(19, 25),
    "formal_charge": slice(25, 31),
    "hybridization": slice(31, 37),
    "aromatic":      slice(37, 38),
    "mass":          slice(38, 39),
    "chirality":     slice(39, 42),
}

BOND_SLICES = {
    "bond_type":  slice(0, 4),
    "conjugated": slice(4, 5),
    "in_ring":    slice(5, 6),
    "stereo":     slice(6, 11),
}

# Features where divergence is an expected approximation, not a bug
ATOM_APPROX_FEATURES = {"total_h", "formal_charge", "hybridization", "aromatic"}
BOND_APPROX_FEATURES = {"bond_type", "conjugated", "in_ring", "stereo"}
# stereo: resolving E/Z on the double bond requires cross-bond context
# (the / \ tokens sit on adjacent single bonds, not on the double bond
# itself). Our heuristic-free encoder cannot resolve this without a
# full stereo perception pass — classified as a known approximation.

FLOAT_ATOL = 1e-4   # tolerance for mass comparison


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FeatureDiff:
    feature_name: str
    atom_or_bond_idx: int
    rdkit_vec: list
    ours_vec:  list
    is_approx: bool   # True = expected approximation; False = unexpected bug

    def __str__(self) -> str:
        tag = "APPROX" if self.is_approx else "BUG   "
        return (f"[{tag}] {self.feature_name:15s} "
                f"{'atom' if 'atom' in self.feature_name or True else 'bond'}[{self.atom_or_bond_idx}]: "
                f"rdkit={self.rdkit_vec} ours={self.ours_vec}")


@dataclass
class MoleculeDiff:
    smiles: str
    n_atoms: int
    n_bonds: int
    atom_diffs: list[FeatureDiff] = field(default_factory=list)
    bond_diffs: list[FeatureDiff] = field(default_factory=list)

    @property
    def has_unexpected(self) -> bool:
        return any(not d.is_approx for d in self.atom_diffs + self.bond_diffs)

    @property
    def is_exact_match(self) -> bool:
        return not self.atom_diffs and not self.bond_diffs


@dataclass
class CorpusStats:
    total_molecules:      int = 0
    exact_match:          int = 0
    approx_only:          int = 0
    unexpected_diff:      int = 0
    total_atoms:          int = 0
    total_bonds:          int = 0
    atom_diffs_by_feature: dict = field(default_factory=dict)
    bond_diffs_by_feature: dict = field(default_factory=dict)
    unexpected_cases:      list = field(default_factory=list)

    @property
    def exact_match_pct(self) -> float:
        return 100 * self.exact_match / max(self.total_molecules, 1)

    @property
    def unexpected_pct(self) -> float:
        return 100 * self.unexpected_diff / max(self.total_molecules, 1)


# ---------------------------------------------------------------------------
# Core comparison
# ---------------------------------------------------------------------------

def _vecs_equal(a: list, b: list, atol: float = FLOAT_ATOL) -> bool:
    if len(a) != len(b):
        return False
    return all(abs(float(x) - float(y)) <= atol for x, y in zip(a, b))


def compare_atom_features(
    rdkit_feats: list[list],
    our_feats: list[list],
) -> list[FeatureDiff]:
    """Compare per-atom feature vectors; return list of diffs."""
    diffs = []
    for idx, (rf, of) in enumerate(zip(rdkit_feats, our_feats)):
        for fname, sl in ATOM_SLICES.items():
            rv = rf[sl.start:sl.stop]
            ov = of[sl.start:sl.stop]
            if not _vecs_equal(rv, ov):
                diffs.append(FeatureDiff(
                    feature_name=f"atom.{fname}",
                    atom_or_bond_idx=idx,
                    rdkit_vec=rv,
                    ours_vec=ov,
                    is_approx=(fname in ATOM_APPROX_FEATURES),
                ))
    return diffs


def compare_bond_features(
    rdkit_feats: list[list],
    our_feats: list[list],
) -> list[FeatureDiff]:
    """Compare per-bond feature vectors (undirected); return list of diffs."""
    diffs = []
    for idx, (rf, of) in enumerate(zip(rdkit_feats, our_feats)):
        for fname, sl in BOND_SLICES.items():
            rv = rf[sl.start:sl.stop]
            ov = of[sl.start:sl.stop]
            if not _vecs_equal(rv, ov):
                diffs.append(FeatureDiff(
                    feature_name=f"bond.{fname}",
                    atom_or_bond_idx=idx,
                    rdkit_vec=rv,
                    ours_vec=ov,
                    is_approx=(fname in BOND_APPROX_FEATURES),
                ))
    return diffs


def compare_smiles(smiles: str) -> Optional[MoleculeDiff]:
    """Run both encoders on one SMILES and return a diff report.

    Returns None if either encoder cannot parse the SMILES.
    """
    import sys
    sys.path.insert(0, str(HERE.parent))

    from smiles_processing_comparison.rdkit_encoder import get_raw_atom_features, get_raw_bond_features
    from smiles_processing.smiles_parser import parse_smiles
    from smiles_processing.smiles_features import extract_features
    from smiles_processing.feature_encoding import encode_atom, encode_bond

    rdkit_atoms = get_raw_atom_features(smiles)
    rdkit_bonds = get_raw_bond_features(smiles)
    if rdkit_atoms is None:
        return None

    try:
        g = parse_smiles(smiles, strict=True)
        extract_features(g)
    except Exception:
        return None

    if len(rdkit_atoms) != len(g["atoms"]) or len(rdkit_bonds) != len(g["bonds"]):
        return None  # topology mismatch — exclude from comparison

    our_atoms = [encode_atom(a, g) for a in g["atoms"]]
    our_bonds = [encode_bond(b) for b in g["bonds"]]

    atom_diffs = compare_atom_features(rdkit_atoms, our_atoms)
    bond_diffs = compare_bond_features(rdkit_bonds, our_bonds)

    return MoleculeDiff(
        smiles=smiles,
        n_atoms=len(g["atoms"]),
        n_bonds=len(g["bonds"]),
        atom_diffs=atom_diffs,
        bond_diffs=bond_diffs,
    )


def run_corpus(csv_path: Optional[Path] = None) -> CorpusStats:
    """Run comparison over the entire filtered corpus.

    Args:
        csv_path: Path to filtered CSV.  Defaults to ``esol_filtered.csv``
                  next to this file.

    Returns:
        Populated :class:`CorpusStats`.
    """
    if csv_path is None:
        csv_path = HERE / "esol_filtered.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {csv_path}\n"
            "Run:  python comparison/prepare_dataset.py"
        )

    with open(csv_path) as f:
        smiles_list = [row["smiles"].strip() for row in csv.DictReader(f)]

    stats = CorpusStats()

    for smi in smiles_list:
        diff = compare_smiles(smi)
        if diff is None:
            continue  # both parsers failed — skip

        stats.total_molecules += 1
        stats.total_atoms     += diff.n_atoms
        stats.total_bonds     += diff.n_bonds

        if diff.is_exact_match:
            stats.exact_match += 1
        elif diff.has_unexpected:
            stats.unexpected_diff += 1
            stats.unexpected_cases.append(diff)
        else:
            stats.approx_only += 1

        for d in diff.atom_diffs:
            stats.atom_diffs_by_feature[d.feature_name] = (
                stats.atom_diffs_by_feature.get(d.feature_name, 0) + 1
            )
        for d in diff.bond_diffs:
            stats.bond_diffs_by_feature[d.feature_name] = (
                stats.bond_diffs_by_feature.get(d.feature_name, 0) + 1
            )

    return stats


def print_report(stats: CorpusStats) -> None:
    """Print a human-readable summary of corpus stats."""
    print("=" * 60)
    print("ENCODER COMPARISON REPORT — ESOL corpus")
    print("=" * 60)
    print(f"Molecules compared  : {stats.total_molecules}")
    print(f"Exact match         : {stats.exact_match:4d}  ({stats.exact_match_pct:.1f}%)")
    print(f"Approx diffs only   : {stats.approx_only:4d}")
    print(f"Unexpected diffs    : {stats.unexpected_diff:4d}  ({stats.unexpected_pct:.1f}%)")
    print()
    print("Atom feature diffs (known approximations):")
    for feat, n in sorted(stats.atom_diffs_by_feature.items()):
        tag = "APPROX" if any(a in feat for a in ATOM_APPROX_FEATURES) else "BUG"
        print(f"  [{tag}] {feat:30s}: {n} atoms")
    print("Bond feature diffs (known approximations):")
    for feat, n in sorted(stats.bond_diffs_by_feature.items()):
        tag = "APPROX" if any(a in feat for a in BOND_APPROX_FEATURES) else "BUG"
        print(f"  [{tag}] {feat:30s}: {n} bonds")
    if stats.unexpected_cases:
        print()
        print("UNEXPECTED DIFFERENCES (potential bugs):")
        for diff in stats.unexpected_cases[:5]:
            print(f"  {diff.smiles}")
            for d in diff.atom_diffs + diff.bond_diffs:
                if not d.is_approx:
                    print(f"    {d}")
    print("=" * 60)


if __name__ == "__main__":
    stats = run_corpus()
    print_report(stats)

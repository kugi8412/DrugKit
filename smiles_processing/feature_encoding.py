#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# smiles_processing/feature_encoding.py

"""Deterministic fixed-size feature encoding for GNN preprocessing.

Produces atom and bond feature vectors that are **dimensionally and
semantically compatible** with the RDKit-based encodings used in
``siamese_GNN/improved_train.py`` and ``final_docking/virtual_screeing.py``.

The RDKit layout (derived from the existing codebase) is:

Atom features (42 dimensions total):
  [0:13]   atom symbol one-hot  (12 permitted atoms + 1 unknown)
  [13:19]  degree one-hot       (0,1,2,3,4 + unknown)
  [19:25]  total H count        (0,1,2,3,4 + unknown)
  [25:31]  formal charge        (-1,-2,+1,+2,0 + unknown)
  [31:37]  hybridization        (SP,SP2,SP3,SP3D,SP3D2 + unknown)
  [37]     is_aromatic flag
  [38]     mass * 0.01          (normalised)
  [39:42]  chirality one-hot    (CW, CCW + unknown/none)

Bond features (11 dimensions total):
  [0]   is_SINGLE
  [1]   is_DOUBLE
  [2]   is_TRIPLE
  [3]   is_AROMATIC
  [4]   is_conjugated
  [5]   is_in_ring
  [6:11] stereo one-hot (STEREOZ,STEREOE,STEREOCIS,STEREOTRANS + unknown/none)

All vectors are plain Python ``list[float]`` for zero-dependency portability.
Call :func:`atom_features_tensor` / :func:`bond_features_tensor` when you
need ``torch.Tensor`` output instead.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Vocabulary constants  (must stay in sync with the existing GNN code)
# ---------------------------------------------------------------------------

PERMITTED_ATOMS: Final[list[str]] = [
    "C", "N", "O", "S", "F", "Si", "P", "Cl", "Br", "I", "B", "H"
]
ATOM_DEGREES: Final[list[int]] = [0, 1, 2, 3, 4]
TOTAL_HS: Final[list[int]] = [0, 1, 2, 3, 4]
FORMAL_CHARGES: Final[list[int]] = [-1, -2, 1, 2, 0]
HYBRIDIZATIONS: Final[list[str]] = ["SP", "SP2", "SP3", "SP3D", "SP3D2"]
CHIRALITY_TAGS: Final[list[str]] = ["CHI_TETRAHEDRAL_CW", "CHI_TETRAHEDRAL_CCW"]
STEREO_TYPES: Final[list[str]] = ["STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS"]

# Approximate atomic masses (for the 12 permitted atoms + fallback)
_ATOMIC_MASS: Final[dict[str, float]] = {
    "C": 12.011,  "N": 14.007,  "O": 15.999,  "S": 32.06,
    "F": 18.998,  "SI": 28.086, "P": 30.974,  "CL": 35.45,
    "BR": 79.904, "I": 126.90,  "B": 10.811,  "H": 1.008,
    # metals that may appear in bracket atoms
    "FE": 55.845, "CU": 63.546, "ZN": 65.38,  "MG": 24.305,
    "CA": 40.078, "NA": 22.990, "K": 39.098,  "LI": 6.941,
}
_DEFAULT_MASS: Final[float] = 12.0  # fallback

# Map our parser's chirality strings to RDKit-style tag names
_CHIRALITY_MAP: Final[dict[str, str]] = {
    "@":  "CHI_TETRAHEDRAL_CCW",
    "@@": "CHI_TETRAHEDRAL_CW",
}

# Map our parser's stereo bond strings to RDKit-style names
# Note: without full stereo resolution we can only signal E/Z directionally;
# we map "/" → STEREOE and "\" → STEREOZ as a stable heuristic.
_STEREO_MAP: Final[dict[str, str]] = {
    "/":  "STEREOE",
    "\\": "STEREOZ",
}

# Computed once
ATOM_FEATURE_DIM: Final[int] = (
    len(PERMITTED_ATOMS) + 1       # symbol
    + len(ATOM_DEGREES) + 1        # degree
    + len(TOTAL_HS) + 1            # total_h
    + len(FORMAL_CHARGES) + 1      # formal_charge
    + len(HYBRIDIZATIONS) + 1      # hybridization
    + 1                            # aromatic flag
    + 1                            # mass
    + len(CHIRALITY_TAGS) + 1      # chirality
)  # = 42

BOND_FEATURE_DIM: Final[int] = (
    4          # bond type one-hot
    + 1        # conjugated
    + 1        # in_ring
    + len(STEREO_TYPES) + 1   # stereo = 5
)  # = 11


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _one_hot(value: object, choices: list) -> list[float]:
    """One-hot encode *value* against *choices*, with an unknown/other bucket.

    The unknown bucket is the **last** element (index -1), matching the
    convention in the existing GNN codebase.

    Args:
        value: The value to encode.
        choices: Ordered list of known values.

    Returns:
        Float list of length ``len(choices) + 1``.
    """
    encoding = [0.0] * (len(choices) + 1)
    try:
        idx = choices.index(value)
    except ValueError:
        idx = -1  # unknown bucket
    encoding[idx] = 1.0
    return encoding


def _infer_degree(atom: dict, graph: dict) -> int:
    """Count the number of bonds incident to *atom*.

    This is a graph-degree count, not a valence count, so it closely
    mirrors ``atom.GetDegree()`` from RDKit for non-hydrogen atoms.

    Args:
        atom: Atom dict from the graph.
        graph: Full molecular graph (needed to scan bonds).

    Returns:
        Integer degree (number of explicit bonds).
    """
    atom_id = atom["id"]
    return sum(
        1 for b in graph["bonds"]
        if b["start"] == atom_id or b["end"] == atom_id
    )


def _infer_total_h(atom: dict, graph: dict) -> int:
    """Estimate total implicit + explicit H count for *atom*.

    For bracket atoms the ``explicit_h`` field is used directly.
    For bare organic atoms we apply a simple valence-table heuristic
    (standard valence minus degree) that gives correct results for
    the common organic subset.

    Args:
        atom: Atom dict.
        graph: Full molecular graph.

    Returns:
        Integer H count (clamped to 0–4 for encoding purposes).
    """
    if atom.get("explicit_h", 0) > 0:
        return min(atom["explicit_h"], 4)

    degree = _infer_degree(atom, graph)
    # Standard valences for the supported organic subset
    _VALENCE: dict[str, int] = {
        "C": 4, "N": 3, "O": 2, "S": 2, "P": 3,
        "F": 1, "CL": 1, "BR": 1, "I": 1, "B": 3,
        "H": 1, "SI": 4,
    }
    valence = _VALENCE.get(atom["symbol"].upper(), 0)
    if atom["aromatic"] and atom["symbol"].upper() in {"C", "N"}:
        # aromatic carbons typically have 1 implicit H, nitrogens 0 or 1
        valence -= 1  # rough correction
    return max(0, min(valence - degree, 4))


# ---------------------------------------------------------------------------
# Public feature extraction functions
# ---------------------------------------------------------------------------

def encode_atom(atom: dict, graph: dict) -> list[float]:
    """Return a fixed-size float list encoding for a single atom.

    The encoding is deterministic and compatible with the existing
    ``get_atom_features`` function used in ``improved_train.py``.

    Args:
        atom: Atom dict from ``graph["atoms"]``.
        graph: Full molecular graph (used to infer degree and H count).

    Returns:
        Float list of length :data:`ATOM_FEATURE_DIM` (42).

    Example:
        >>> from smiles_processing.smiles_parser import parse_smiles
        >>> from smiles_processing.smiles_features import extract_features
        >>> g = parse_smiles("CC")
        >>> fg = extract_features(g)
        >>> feats = encode_atom(fg["atoms"][0], g)
        >>> len(feats)
        42
    """
    symbol = atom["symbol"].upper()
    degree = _infer_degree(atom, graph)
    total_h = _infer_total_h(atom, graph)
    formal_charge = atom.get("formal_charge", 0)
    hyb = atom.get("hybridization") or "SP3"  # default if not yet computed
    aromatic = atom.get("aromatic", False)
    mass = _ATOMIC_MASS.get(symbol, _DEFAULT_MASS) * 0.01

    # Chirality: map parser strings to RDKit-style tag names
    raw_chiral = atom.get("chirality")
    chiral_tag = _CHIRALITY_MAP.get(raw_chiral, "") if raw_chiral else ""

    features: list[float] = []
    features += _one_hot(symbol, PERMITTED_ATOMS)
    features += _one_hot(degree, ATOM_DEGREES)
    features += _one_hot(total_h, TOTAL_HS)
    features += _one_hot(formal_charge, FORMAL_CHARGES)
    features += _one_hot(hyb, HYBRIDIZATIONS)
    features += [1.0 if aromatic else 0.0]
    features += [mass]
    features += _one_hot(chiral_tag, CHIRALITY_TAGS)

    assert len(features) == ATOM_FEATURE_DIM, (
        f"BUG: atom feature dim mismatch: got {len(features)}, expected {ATOM_FEATURE_DIM}"
    )
    return features


def encode_bond(bond: dict) -> list[float]:
    """Return a fixed-size float list encoding for a single bond.

    The encoding is deterministic and compatible with the existing
    ``get_bond_features`` function used in ``improved_train.py``.

    Args:
        bond: Bond dict from ``graph["bonds"]``.

    Returns:
        Float list of length :data:`BOND_FEATURE_DIM` (11).

    Example:
        >>> from smiles_processing.smiles_parser import parse_smiles
        >>> g = parse_smiles("C=C")
        >>> feats = encode_bond(g["bonds"][0])
        >>> feats[1]  # DOUBLE flag
        1.0
        >>> len(feats)
        11
    """
    bt = bond["bond_type"]
    features: list[float] = [
        1.0 if bt == "SINGLE"   else 0.0,
        1.0 if bt == "DOUBLE"   else 0.0,
        1.0 if bt == "TRIPLE"   else 0.0,
        1.0 if bt == "AROMATIC" else 0.0,
        1.0 if bond.get("conjugated", False) else 0.0,
        1.0 if bond.get("in_ring", False) else 0.0,
    ]

    # Stereo: map parser "/" "\" to RDKit-style names, or None
    raw_stereo = bond.get("stereochemistry")
    stereo_tag = _STEREO_MAP.get(raw_stereo, "") if raw_stereo else ""
    features += _one_hot(stereo_tag, STEREO_TYPES)

    assert len(features) == BOND_FEATURE_DIM, (
        f"BUG: bond feature dim mismatch: got {len(features)}, expected {BOND_FEATURE_DIM}"
    )
    return features


# ---------------------------------------------------------------------------
# Tensor wrappers (optional — requires torch)
# ---------------------------------------------------------------------------

def atom_features_tensor(atom: dict, graph: dict):  # -> torch.Tensor
    """Return atom features as a ``torch.Tensor`` of shape ``(42,)``.

    Requires PyTorch to be installed.

    Args:
        atom: Atom dict from the graph.
        graph: Full molecular graph.

    Returns:
        ``torch.FloatTensor`` of shape ``(ATOM_FEATURE_DIM,)``.
    """
    import torch  # noqa: PLC0415
    return torch.tensor(encode_atom(atom, graph), dtype=torch.float)


def bond_features_tensor(bond: dict):  # -> torch.Tensor
    """Return bond features as a ``torch.Tensor`` of shape ``(11,)``.

    Requires PyTorch to be installed.

    Args:
        bond: Bond dict from the graph.

    Returns:
        ``torch.FloatTensor`` of shape ``(BOND_FEATURE_DIM,)``.
    """
    import torch  # noqa: PLC0415
    return torch.tensor(encode_bond(bond), dtype=torch.float)

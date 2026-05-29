"""Atom and bond feature extraction for GNN preprocessing.

Takes the raw molecular graph produced by :func:`smiles_parser.parse_smiles`
and enriches each atom with a heuristic ``hybridization`` value, then
returns structured feature dictionaries ready for downstream tensor encoding.

No chemistry library is used.  Hybridization is determined from the local
bond environment of each atom, which is accurate for the common organic
molecules targeted by this parser but is not a rigorous quantum-chemical
calculation.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Hybridization string values (kept as plain strings for easy JSON export).
SP: Final[str] = "SP"
SP2: Final[str] = "SP2"
SP3: Final[str] = "SP3"

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_adjacency(graph: dict) -> dict[int, list[dict]]:
    """Build a mapping from atom index to its incident bonds.

    Args:
        graph: Molecular graph with ``"atoms"`` and ``"bonds"`` lists.

    Returns:
        Dict mapping atom id -> list of bond dicts incident to that atom.
    """
    adj: dict[int, list[dict]] = {atom["id"]: [] for atom in graph["atoms"]}
    for bond in graph["bonds"]:
        adj[bond["start"]].append(bond)
        adj[bond["end"]].append(bond)
    return adj


def _infer_hybridization(atom: dict, incident_bonds: list[dict]) -> str:
    """Infer hybridization state from an atom's bond environment.

    Rules applied in priority order:

    1. **SP** — atom has a triple bond, or two or more double bonds.
    2. **SP2** — atom is aromatic, or has exactly one double bond.
    3. **SP3** — all bonds are single.

    These heuristics cover the vast majority of common drug-like molecules.

    Args:
        atom: Atom dictionary from the graph.
        incident_bonds: All bonds connected to this atom.

    Returns:
        One of ``"SP"``, ``"SP2"``, or ``"SP3"``.
    """
    if atom["aromatic"]:
        return SP2

    bond_types = [b["bond_type"] for b in incident_bonds]
    n_triple = bond_types.count("TRIPLE")
    n_double = bond_types.count("DOUBLE")

    if n_triple >= 1 or n_double >= 2:
        return SP
    if n_double == 1:
        return SP2
    return SP3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_atom_features(atom: dict, graph: dict) -> dict:
    """Extract and return a feature dictionary for a single atom.

    This function also **updates** the atom's ``hybridization`` field
    in-place inside *graph* so that subsequent calls to
    :func:`extract_bond_features` can rely on it.

    Args:
        atom: An atom dict from ``graph["atoms"]``.
        graph: The full molecular graph (used to look up incident bonds).

    Returns:
        A feature dictionary with keys:

        * ``id`` (int) — atom index.
        * ``symbol`` (str) — uppercase atom symbol.
        * ``formal_charge`` (int) — always 0 in this parser subset.
        * ``hybridization`` (str) — one of ``"SP"``, ``"SP2"``, ``"SP3"``.
        * ``chirality`` (str | None) — ``"@"``, ``"@@"``, or ``None``.
        * ``aromatic`` (bool) — ``True`` if the atom was lowercase in SMILES.

    Example:
        >>> from smiles_parser import parse_smiles
        >>> g = parse_smiles("C=C")
        >>> extract_atom_features(g["atoms"][0], g)
        {'id': 0, 'symbol': 'C', 'formal_charge': 0, 'hybridization': 'SP2', 'chirality': None, 'aromatic': False}
    """
    adj = _build_adjacency(graph)
    incident = adj[atom["id"]]
    hybridization = _infer_hybridization(atom, incident)

    # Update in-place so the graph stays consistent.
    atom["hybridization"] = hybridization

    return {
        "id": atom["id"],
        "symbol": atom["symbol"],
        "formal_charge": atom["formal_charge"],
        "hybridization": hybridization,
        "chirality": atom["chirality"],
        "aromatic": atom["aromatic"],
    }


def extract_bond_features(bond: dict, graph: dict) -> dict:  # noqa: ARG001
    """Extract and return a feature dictionary for a single bond.

    Args:
        bond: A bond dict from ``graph["bonds"]``.
        graph: The full molecular graph (accepted for API consistency; not
            currently used but may be needed for future features).

    Returns:
        A feature dictionary with keys:

        * ``start`` (int) — index of the first atom.
        * ``end`` (int) — index of the second atom.
        * ``bond_type`` (str) — ``"SINGLE"``, ``"DOUBLE"``, ``"TRIPLE"``,
          or ``"AROMATIC"``.
        * ``stereochemistry`` (str | None) — ``"/"``, ``"\\"`` or ``None``.
        * ``in_ring`` (bool) — ``True`` if the bond is part of a ring.
        * ``conjugated`` (bool) — ``True`` if the bond is in a conjugated
          system.

    Example:
        >>> from smiles_parser import parse_smiles
        >>> g = parse_smiles("C=C")
        >>> extract_bond_features(g["bonds"][0], g)
        {'start': 0, 'end': 1, 'bond_type': 'DOUBLE', 'stereochemistry': None, 'in_ring': False, 'conjugated': True}
    """
    return {
        "start": bond["start"],
        "end": bond["end"],
        "bond_type": bond["bond_type"],
        "stereochemistry": bond["stereochemistry"],
        "in_ring": bond["in_ring"],
        "conjugated": bond["conjugated"],
    }


def extract_features(graph: dict) -> dict:
    """Enrich an entire molecular graph with atom and bond features.

    Convenience wrapper that calls :func:`extract_atom_features` and
    :func:`extract_bond_features` for every node/edge and returns an
    updated graph dict.  The input *graph* is modified in-place (the
    ``hybridization`` fields of atoms are set), and a new dict with
    ``"atoms"`` and ``"bonds"`` feature lists is returned.

    Args:
        graph: Raw molecular graph from :func:`smiles_parser.parse_smiles`.

    Returns:
        Dict with:

        * ``"atoms"`` — list of atom feature dicts.
        * ``"bonds"`` — list of bond feature dicts.

    Example:
        >>> from smiles_parser import parse_smiles
        >>> g = parse_smiles("c1ccccc1")
        >>> result = extract_features(g)
        >>> result["atoms"][0]["hybridization"]
        'SP2'
        >>> result["bonds"][0]["in_ring"]
        True
    """
    atom_features = [extract_atom_features(atom, graph) for atom in graph["atoms"]]
    bond_features = [extract_bond_features(bond, graph) for bond in graph["bonds"]]
    return {"atoms": atom_features, "bonds": bond_features}

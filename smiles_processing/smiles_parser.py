#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# smiles_processing/smiles_parser.py

"""Graph construction from a tokenized SMILES string.

Converts the flat token list produced by :mod:`smiles_tokenizer` into a
molecular graph represented as a plain Python dictionary with ``"atoms"``
and ``"bonds"`` lists.  Bracket atoms (``[NH4+]``, ``[O-]``, ``[nH]``,
``[C@H]``, ``[Fe+2]``, …) are fully supported.

The public entry point is :func:`parse_smiles`, which accepts an optional
``strict`` keyword argument forwarded to the tokenizer.
"""

from __future__ import annotations

import logging
from typing import Final

from smiles_processing.smiles_errors import SMILESParseError, SMILESValidationError
from smiles_processing.smiles_tokenizer import tokenize_smiles

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Lowercase aromatic atom symbols (bare, i.e. outside brackets).
AROMATIC_SYMBOLS: Final[frozenset[str]] = frozenset({"c", "n", "o", "s"})

#: Map from SMILES bond token to canonical bond-type name.
BOND_TYPE_MAP: Final[dict[str, str]] = {
    "-": "SINGLE",
    "=": "DOUBLE",
    "#": "TRIPLE",
    ":": "AROMATIC",
    "/": "SINGLE",   # stereo single
    "\\": "SINGLE",  # stereo single
}

#: Stereo-carrying bond tokens.
STEREO_BOND_TOKENS: Final[frozenset[str]] = frozenset({"/", "\\"})

#: Default bond when no explicit token appears between two atoms.
DEFAULT_BOND: Final[str] = "SINGLE"

# ---------------------------------------------------------------------------
# Internal atom / bond constructors
# ---------------------------------------------------------------------------

def _make_atom(
    atom_id: int,
    symbol: str,
    chirality: str | None,
    aromatic: bool = False,
    formal_charge: int = 0,
    explicit_h: int = 0,
) -> dict:
    """Build the raw atom dictionary.

    Args:
        atom_id: Zero-based index.
        symbol: **Uppercase** element symbol (e.g. ``"C"``, ``"FE"``).
        chirality: ``"@"`` or ``"@@"`` or ``None``.
        aromatic: ``True`` if the atom originated from a lowercase or bracket
            aromatic token.
        formal_charge: Integer formal charge (0 for most atoms).
        explicit_h: Count of explicit hydrogens specified in a bracket atom.

    Returns:
        Atom dict. ``hybridization`` is ``None`` — filled by feature extractor.
    """
    return {
        "id": atom_id,
        "symbol": symbol,
        "formal_charge": formal_charge,
        "hybridization": None,    # computed later by smiles_features
        "chirality": chirality,
        "aromatic": aromatic,
        "explicit_h": explicit_h,
    }


def _make_bond(start: int, end: int, bond_token: str) -> dict:
    """Build the raw bond dictionary.

    Args:
        start: Index of the first atom.
        end: Index of the second atom.
        bond_token: The SMILES token that produced this bond.

    Returns:
        Bond dict.
    """
    stereo: str | None = bond_token if bond_token in STEREO_BOND_TOKENS else None
    bond_type = BOND_TYPE_MAP.get(bond_token, "SINGLE")
    return {
        "start": start,
        "end": end,
        "bond_type": bond_type,
        "stereochemistry": stereo,
        "in_ring": False,
        "conjugated": False,
    }


# ---------------------------------------------------------------------------
# Post-processing passes  (identical logic to the original)
# ---------------------------------------------------------------------------

def _infer_aromatic_bonds(graph: dict) -> None:
    """Upgrade implicit SINGLE bonds to AROMATIC where both atoms are aromatic."""
    aromatic_ids: set[int] = {a["id"] for a in graph["atoms"] if a["aromatic"]}
    for bond in graph["bonds"]:
        if bond["start"] in aromatic_ids and bond["end"] in aromatic_ids:
            if bond["bond_type"] == "SINGLE" and bond["stereochemistry"] is None:
                bond["bond_type"] = "AROMATIC"


def _mark_ring_bonds(graph: dict) -> None:
    """Set ``in_ring=True`` for every bond that belongs to a ring (DFS)."""
    n = len(graph["atoms"])
    if n == 0:
        return

    adj: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n)}
    for bi, bond in enumerate(graph["bonds"]):
        adj[bond["start"]].append((bond["end"], bi))
        adj[bond["end"]].append((bond["start"], bi))

    visited = [False] * n
    parent_bond: list[int] = [-1] * n
    parent_node: list[int] = [-1] * n
    ring_bond_indices: set[int] = set()

    def dfs(node: int) -> None:
        visited[node] = True
        for neighbour, bond_idx in adj[node]:
            if not visited[neighbour]:
                parent_bond[neighbour] = bond_idx
                parent_node[neighbour] = node
                dfs(neighbour)
            elif bond_idx != parent_bond[node]:
                ring_bond_indices.add(bond_idx)
                current = node
                while current != neighbour and current != -1:
                    pb = parent_bond[current]
                    if pb != -1:
                        ring_bond_indices.add(pb)
                    current = parent_node[current]

    for start in range(n):
        if not visited[start]:
            dfs(start)

    for bi in ring_bond_indices:
        graph["bonds"][bi]["in_ring"] = True


def _mark_conjugated_bonds(graph: dict) -> None:
    """Set ``conjugated=True`` for bonds in conjugated systems."""
    pi_bond_indices: set[int] = set()
    for bi, bond in enumerate(graph["bonds"]):
        if bond["bond_type"] in {"DOUBLE", "AROMATIC", "TRIPLE"}:
            pi_bond_indices.add(bi)

    if not pi_bond_indices:
        return

    pi_atoms: set[int] = set()
    for bi in pi_bond_indices:
        pi_atoms.add(graph["bonds"][bi]["start"])
        pi_atoms.add(graph["bonds"][bi]["end"])

    for bi, bond in enumerate(graph["bonds"]):
        if bi in pi_bond_indices:
            bond["conjugated"] = True
        elif bond["start"] in pi_atoms or bond["end"] in pi_atoms:
            bond["conjugated"] = True


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------

def parse_smiles(smiles: str, strict: bool = True) -> dict:
    """Parse a SMILES string and return a molecular graph dictionary.

    The returned graph has two keys:

    * ``"atoms"`` — list of atom dicts, each with keys ``id``, ``symbol``,
      ``formal_charge``, ``hybridization`` (``None`` until feature extraction),
      ``chirality``, ``aromatic``, and ``explicit_h``.
    * ``"bonds"`` — list of bond dicts, each with keys ``start``, ``end``,
      ``bond_type``, ``stereochemistry``, ``in_ring``, and ``conjugated``.

    Bracket atoms (``[NH4+]``, ``[O-]``, ``[nH]``, ``[C@H]``, ``[Fe+2]`` …)
    are parsed and their charge/H-count/chirality fields are captured.

    Args:
        smiles: A SMILES string.
        strict: Forwarded to :func:`~smiles_tokenizer.tokenize_smiles`.
            When ``False``, unsupported tokens are skipped rather than
            raising exceptions — suitable for large-scale preprocessing.

    Returns:
        Molecular graph as a plain ``dict``.

    Raises:
        SMILESValidationError: If *smiles* is empty or produces no atoms.
        SMILESTokenizationError: If an unrecognised token is found and
            ``strict=True``.
        UnsupportedSMILESFeatureError: If unsupported syntax is detected and
            ``strict=True``.
        SMILESParseError: If the token stream is structurally invalid (e.g.
            unmatched parentheses, unclosed ring closures).

    Examples:
        >>> g = parse_smiles("CC")
        >>> [(b["start"], b["end"], b["bond_type"]) for b in g["bonds"]]
        [(0, 1, 'SINGLE')]

        >>> g = parse_smiles("[NH4+]")
        >>> g["atoms"][0]["formal_charge"]
        1

        >>> g = parse_smiles("[C@H](F)(Cl)Br")
        >>> g["atoms"][0]["chirality"]
        '@'

        >>> g = parse_smiles("C1CCCCC1")
        >>> all(b["in_ring"] for b in g["bonds"])
        True
    """
    tokens = tokenize_smiles(smiles, strict=strict)

    atoms: list[dict] = []
    bonds: list[dict] = []

    branch_stack: list[int] = []
    ring_openings: dict[str, tuple[int, str]] = {}

    current_atom_idx: int = -1
    pending_bond: str = DEFAULT_BOND
    pending_bond_explicit: bool = False

    # Bare chirality marker buffer (outside brackets)
    chirality_buffer: str | None = None

    i = 0
    n_tokens = len(tokens)

    while i < n_tokens:
        tok = tokens[i]

        # ------------------------------------------------------------------ #
        # Bracket atom dict — emitted by tokenizer for [...] expressions     #
        # ------------------------------------------------------------------ #
        if isinstance(tok, dict) and tok.get("token_type") == "bracket_atom":
            new_idx = len(atoms)
            # Chirality may come from the bracket itself or from a preceding '@'
            chirality = tok["chirality"] or chirality_buffer
            chirality_buffer = None
            atoms.append(_make_atom(
                atom_id=new_idx,
                symbol=tok["symbol"],
                chirality=chirality,
                aromatic=tok["aromatic"],
                formal_charge=tok["formal_charge"],
                explicit_h=tok["explicit_h"],
            ))
            if current_atom_idx >= 0:
                bonds.append(_make_bond(current_atom_idx, new_idx, pending_bond))
            current_atom_idx = new_idx
            pending_bond = DEFAULT_BOND
            pending_bond_explicit = False
            i += 1
            continue

        # ------------------------------------------------------------------ #
        # Bare chirality marker                                               #
        # ------------------------------------------------------------------ #
        if tok in {"@", "@@"}:
            chirality_buffer = tok
            i += 1
            continue

        # ------------------------------------------------------------------ #
        # Explicit bond token                                                 #
        # ------------------------------------------------------------------ #
        if tok in BOND_TYPE_MAP:
            if pending_bond_explicit:
                if strict:
                    raise SMILESParseError(
                        f"Consecutive bond tokens at position {i}: "
                        f"'{pending_bond}' followed by '{tok}'."
                    )
                else:
                    logger.warning("Tolerant mode: ignoring consecutive bond token '%s'", tok)
                    i += 1
                    continue
            pending_bond = tok
            pending_bond_explicit = True
            i += 1
            continue

        # ------------------------------------------------------------------ #
        # Branch open                                                         #
        # ------------------------------------------------------------------ #
        if tok == "(":
            if current_atom_idx < 0:
                if strict:
                    raise SMILESParseError(
                        "Branch '(' opened before any atom has been parsed."
                    )
                else:
                    logger.warning("Tolerant mode: ignoring '(' before first atom")
                    i += 1
                    continue
            branch_stack.append(current_atom_idx)
            i += 1
            continue

        # ------------------------------------------------------------------ #
        # Branch close                                                        #
        # ------------------------------------------------------------------ #
        if tok == ")":
            if not branch_stack:
                if strict:
                    raise SMILESParseError(f"Unmatched ')' at token position {i}.")
                else:
                    logger.warning("Tolerant mode: ignoring unmatched ')' at token %d", i)
                    i += 1
                    continue
            current_atom_idx = branch_stack.pop()
            pending_bond = DEFAULT_BOND
            pending_bond_explicit = False
            i += 1
            continue

        # ------------------------------------------------------------------ #
        # Ring closure digit                                                  #
        # ------------------------------------------------------------------ #
        if isinstance(tok, str) and tok.isdigit():
            if current_atom_idx < 0:
                if strict:
                    raise SMILESParseError(
                        f"Ring closure digit '{tok}' appears before any atom."
                    )
                else:
                    logger.warning("Tolerant mode: ignoring ring digit before first atom")
                    i += 1
                    continue
            if tok not in ring_openings:
                ring_openings[tok] = (current_atom_idx, pending_bond)
                pending_bond = DEFAULT_BOND
                pending_bond_explicit = False
            else:
                open_idx, open_bond = ring_openings.pop(tok)
                if open_idx == current_atom_idx:
                    if strict:
                        raise SMILESParseError(
                            f"Ring closure '{tok}' connects an atom to itself."
                        )
                    else:
                        logger.warning("Tolerant mode: skipping self-ring closure '%s'", tok)
                        i += 1
                        continue
                ring_bond_token = pending_bond if pending_bond_explicit else open_bond
                bonds.append(_make_bond(open_idx, current_atom_idx, ring_bond_token))
                pending_bond = DEFAULT_BOND
                pending_bond_explicit = False
            i += 1
            continue

        # ------------------------------------------------------------------ #
        # Bare atom symbol (single or two-character string)                  #
        # ------------------------------------------------------------------ #
        new_idx = len(atoms)
        aromatic = tok in AROMATIC_SYMBOLS
        canonical = tok.upper()
        atoms.append(_make_atom(new_idx, canonical, chirality_buffer, aromatic=aromatic))
        chirality_buffer = None

        if current_atom_idx >= 0:
            bonds.append(_make_bond(current_atom_idx, new_idx, pending_bond))

        current_atom_idx = new_idx
        pending_bond = DEFAULT_BOND
        pending_bond_explicit = False
        i += 1

    # ---------------------------------------------------------------------- #
    # Post-parse validation                                                   #
    # ---------------------------------------------------------------------- #
    if branch_stack:
        if strict:
            raise SMILESParseError(
                f"{len(branch_stack)} unclosed branch parenthesis/parentheses."
            )
        else:
            logger.warning(
                "Tolerant mode: %d unclosed branch(es) ignored", len(branch_stack)
            )

    if ring_openings:
        unclosed = ", ".join(sorted(ring_openings.keys()))
        if strict:
            raise SMILESParseError(f"Unclosed ring closure(s): {unclosed}.")
        else:
            logger.warning("Tolerant mode: unclosed ring closure(s) %s ignored", unclosed)

    if not atoms:
        raise SMILESValidationError("SMILES produced no atoms.")

    graph = {"atoms": atoms, "bonds": bonds}

    _infer_aromatic_bonds(graph)
    _mark_ring_bonds(graph)
    _mark_conjugated_bonds(graph)

    return graph

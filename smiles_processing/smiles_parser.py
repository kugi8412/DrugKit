"""Graph construction from a tokenized SMILES string.

Converts the flat token list produced by :mod:`smiles_tokenizer` into a
molecular graph represented as a plain Python dictionary with ``"atoms"``
and ``"bonds"`` lists.  No chemistry library is used.
"""

from __future__ import annotations

from typing import Final

from smiles_errors import SMILESParseError, SMILESValidationError
from smiles_tokenizer import tokenize_smiles

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Lowercase aromatic atom symbols.
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
# Internal helpers
# ---------------------------------------------------------------------------

def _make_atom(
    atom_id: int,
    symbol: str,
    chirality: str | None,
) -> dict:
    """Build the raw atom dictionary (features filled in later by extractor).

    Args:
        atom_id: Zero-based index of the atom in the graph.
        symbol: Atom symbol as it appears in the SMILES string.
        chirality: ``"@"`` or ``"@@"`` if present, else ``None``.

    Returns:
        Atom dictionary with ``id``, ``symbol``, ``aromatic``, and
        ``chirality``.  ``formal_charge`` and ``hybridization`` are
        placeholder ``None`` values; they are filled by the feature
        extractor.
    """
    aromatic = symbol in AROMATIC_SYMBOLS
    # Normalize: store uppercase symbol for consistency.
    canonical = symbol.upper()
    return {
        "id": atom_id,
        "symbol": canonical,
        "formal_charge": 0,
        "hybridization": None,   # computed later
        "chirality": chirality,
        "aromatic": aromatic,
    }


def _make_bond(
    start: int,
    end: int,
    bond_token: str,
) -> dict:
    """Build the raw bond dictionary (ring/conjugation filled later).

    Args:
        start: Index of the first atom.
        end: Index of the second atom.
        bond_token: The SMILES token that produced this bond (``"-"``,
            ``"="``, ``"#"``, ``":"``, ``"/"``, or ``"\\"``).

    Returns:
        Bond dictionary with ``start``, ``end``, ``bond_type``,
        ``stereochemistry``, ``in_ring``, and ``conjugated``.
    """
    stereo: str | None = None
    if bond_token in STEREO_BOND_TOKENS:
        stereo = bond_token

    bond_type = BOND_TYPE_MAP.get(bond_token, "SINGLE")

    return {
        "start": start,
        "end": end,
        "bond_type": bond_type,
        "stereochemistry": stereo,
        "in_ring": False,   # updated in post-processing
        "conjugated": False,  # updated in post-processing
    }


# ---------------------------------------------------------------------------
# Post-processing passes
# ---------------------------------------------------------------------------

def _infer_aromatic_bonds(graph: dict) -> None:
    """Upgrade implicit single bonds to AROMATIC where both atoms are aromatic.

    When a SMILES string uses lowercase atom symbols (``c``, ``n``, …)
    without explicit ``:`` bond tokens, the tokenizer emits no bond token
    between them and the parser defaults to ``SINGLE``.  This pass corrects
    those bonds to ``AROMATIC``.

    Args:
        graph: Molecular graph.  Modified in-place.
    """
    aromatic_ids: set[int] = {a["id"] for a in graph["atoms"] if a["aromatic"]}
    for bond in graph["bonds"]:
        if bond["start"] in aromatic_ids and bond["end"] in aromatic_ids:
            if bond["bond_type"] == "SINGLE" and bond["stereochemistry"] is None:
                bond["bond_type"] = "AROMATIC"


def _mark_ring_bonds(graph: dict) -> None:
    """Set ``in_ring`` to ``True`` for every bond that belongs to a ring.

    Strategy: a bond (u, v) is in a ring if and only if there exists a path
    from u to v that does *not* use that bond.  We implement this by removing
    each bond in turn and checking connectivity with a simple BFS — but that
    would be O(E²).  Instead we use the standard approach:

    1. Find all back-edges with a DFS (these are necessarily in rings).
    2. For each back-edge (u, v), walk the DFS ancestor path from v up to u
       and mark every bond on that path as ``in_ring``.

    This correctly marks every bond on every simple cycle.

    Args:
        graph: Molecular graph with ``"atoms"`` and ``"bonds"`` lists.
            Modified in-place.
    """
    n = len(graph["atoms"])
    if n == 0:
        return

    # Build adjacency list: node -> list of (neighbour, bond_index)
    adj: dict[int, list[tuple[int, int]]] = {i: [] for i in range(n)}
    for bi, bond in enumerate(graph["bonds"]):
        adj[bond["start"]].append((bond["end"], bi))
        adj[bond["end"]].append((bond["start"], bi))

    visited = [False] * n
    # parent_bond[i] = bond index used to arrive at node i; -1 for roots.
    parent_bond: list[int] = [-1] * n
    # parent_node[i] = the node we came from to reach node i; -1 for roots.
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
                # Back edge found: walk the ancestor chain from node → neighbour
                # marking every bond along the way.
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
    """Set ``conjugated`` to ``True`` for bonds in conjugated systems.

    A bond is considered conjugated if it is:

    * a double or aromatic bond, **or**
    * a single bond that is adjacent to at least one double/aromatic bond.

    This is a well-known heuristic used in GNN preprocessing and is *not*
    a rigorous quantum-chemical determination.

    Args:
        graph: Molecular graph.  Modified in-place.
    """
    pi_bond_indices: set[int] = set()
    for bi, bond in enumerate(graph["bonds"]):
        if bond["bond_type"] in {"DOUBLE", "AROMATIC", "TRIPLE"}:
            pi_bond_indices.add(bi)

    if not pi_bond_indices:
        return

    # Build set of atoms involved in π bonds
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

def parse_smiles(smiles: str) -> dict:
    """Parse a SMILES string and return a molecular graph dictionary.

    The returned graph has two keys:

    * ``"atoms"`` — list of atom dicts, each with keys ``id``, ``symbol``,
      ``formal_charge``, ``hybridization``, ``chirality``, and ``aromatic``.
    * ``"bonds"`` — list of bond dicts, each with keys ``start``, ``end``,
      ``bond_type``, ``stereochemistry``, ``in_ring``, and ``conjugated``.

    Hybridization is ``None`` at this stage; call
    :func:`smiles_features.extract_atom_features` to populate it.

    Args:
        smiles: A SMILES string using the subset of syntax supported by
            the DrugKit parser (see documentation for full details).

    Returns:
        Molecular graph as a plain ``dict``.

    Raises:
        SMILESValidationError: If *smiles* is empty.
        SMILESTokenizationError: If the string contains unrecognised tokens.
        UnsupportedSMILESFeatureError: If the string uses unsupported syntax.
        SMILESParseError: If the token stream is structurally invalid (e.g.
            unmatched parentheses, unclosed ring closures).

    Examples:
        >>> g = parse_smiles("CC")
        >>> [(b["start"], b["end"], b["bond_type"]) for b in g["bonds"]]
        [(0, 1, 'SINGLE')]

        >>> g = parse_smiles("C1CCCCC1")
        >>> all(b["in_ring"] for b in g["bonds"])
        True
    """
    tokens = tokenize_smiles(smiles)

    atoms: list[dict] = []
    bonds: list[dict] = []

    # Stack of atom indices representing the current open branch heads.
    branch_stack: list[int] = []

    # Maps ring-digit string -> (atom_index, bond_token) for open closures.
    ring_openings: dict[str, tuple[int, str]] = {}

    # Index of the atom that the next atom will bond to.
    current_atom_idx: int = -1

    # Explicit bond token waiting to be consumed by the next atom.
    pending_bond: str = DEFAULT_BOND

    # Whether the pending_bond was explicitly set (vs defaulted).
    pending_bond_explicit: bool = False

    chirality_buffer: str | None = None

    i = 0
    n_tokens = len(tokens)

    while i < n_tokens:
        tok = tokens[i]

        # ------------------------------------------------------------------ #
        # Chirality — buffered until the atom token appears next             #
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
                raise SMILESParseError(
                    f"Consecutive bond tokens at position {i}: "
                    f"'{pending_bond}' followed by '{tok}'."
                )
            pending_bond = tok
            pending_bond_explicit = True
            i += 1
            continue

        # ------------------------------------------------------------------ #
        # Branch open                                                         #
        # ------------------------------------------------------------------ #
        if tok == "(":
            if current_atom_idx < 0:
                raise SMILESParseError(
                    "Branch '(' opened before any atom has been parsed."
                )
            branch_stack.append(current_atom_idx)
            i += 1
            continue

        # ------------------------------------------------------------------ #
        # Branch close                                                        #
        # ------------------------------------------------------------------ #
        if tok == ")":
            if not branch_stack:
                raise SMILESParseError(
                    f"Unmatched ')' at token position {i}."
                )
            current_atom_idx = branch_stack.pop()
            # Reset pending bond — after closing a branch the next atom
            # bonds implicitly to the branch root.
            pending_bond = DEFAULT_BOND
            pending_bond_explicit = False
            i += 1
            continue

        # ------------------------------------------------------------------ #
        # Ring closure digit                                                  #
        # ------------------------------------------------------------------ #
        if tok.isdigit():
            if current_atom_idx < 0:
                raise SMILESParseError(
                    f"Ring closure digit '{tok}' appears before any atom."
                )
            if tok not in ring_openings:
                # First encounter — record it together with any pending bond.
                ring_openings[tok] = (current_atom_idx, pending_bond)
                pending_bond = DEFAULT_BOND
                pending_bond_explicit = False
            else:
                # Second encounter — close the ring.
                open_idx, open_bond = ring_openings.pop(tok)
                if open_idx == current_atom_idx:
                    raise SMILESParseError(
                        f"Ring closure '{tok}' connects an atom to itself."
                    )
                # Prefer the bond token from whichever end specified it.
                ring_bond_token = (
                    pending_bond if pending_bond_explicit else open_bond
                )
                bonds.append(_make_bond(open_idx, current_atom_idx, ring_bond_token))
                pending_bond = DEFAULT_BOND
                pending_bond_explicit = False
            i += 1
            continue

        # ------------------------------------------------------------------ #
        # Atom symbol (single or two-character)                               #
        # ------------------------------------------------------------------ #
        # At this point the token must be an atom symbol; the tokenizer
        # already validated it so no further check is needed here.
        new_idx = len(atoms)
        atoms.append(_make_atom(new_idx, tok, chirality_buffer))
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
        raise SMILESParseError(
            f"{len(branch_stack)} unclosed branch parenthesis/parentheses."
        )

    if ring_openings:
        unclosed = ", ".join(sorted(ring_openings.keys()))
        raise SMILESParseError(
            f"Unclosed ring closure(s): {unclosed}."
        )

    if not atoms:
        raise SMILESValidationError("SMILES produced no atoms.")

    graph = {"atoms": atoms, "bonds": bonds}

    # ---------------------------------------------------------------------- #
    # Post-processing                                                         #
    # ---------------------------------------------------------------------- #
    _infer_aromatic_bonds(graph)
    _mark_ring_bonds(graph)
    _mark_conjugated_bonds(graph)

    return graph

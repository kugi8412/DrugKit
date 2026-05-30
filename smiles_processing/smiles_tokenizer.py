# smiles_processing/smiles_tokenizer.py

"""Tokenizer for a subset of the SMILES string format.

Converts a raw SMILES string into a flat list of typed tokens that the
parser can consume one-by-one.  Only the subset described in the DrugKit
specification is supported; anything outside that subset raises an explicit
exception rather than being silently ignored.
"""

import re
from typing import Final

from smiles_processing.smiles_errors import SMILESTokenizationError, SMILESValidationError, UnsupportedSMILESFeatureError

# ---------------------------------------------------------------------------
# Supported symbol sets
# ---------------------------------------------------------------------------

#: Two-character atom symbols that must be matched before single-character ones.
TWO_CHAR_ATOMS: Final[frozenset[str]] = frozenset({"Cl", "Br", "Si"})

#: All supported atom symbols (aromatic lowercase + standard uppercase).
SUPPORTED_ATOMS: Final[frozenset[str]] = frozenset({
    "C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "H", "Si",
    "c", "n", "o", "s",
})

#: Explicit bond characters.
BOND_CHARS: Final[frozenset[str]] = frozenset({"=", "#", ":", "/", "\\"})

#: Characters that open/close branches.
BRANCH_CHARS: Final[frozenset[str]] = frozenset({"(", ")"})

#: Ring-closure digits (1–9 only per spec).
RING_DIGITS: Final[frozenset[str]] = frozenset("123456789")

#: Chirality markers (longest first so '@@' is tried before '@').
CHIRALITY_MARKERS: Final[tuple[str, ...]] = ("@@", "@")

# ---------------------------------------------------------------------------
# Patterns that signal *unsupported* features so we can raise early.
# ---------------------------------------------------------------------------

_UNSUPPORTED_PATTERNS: Final[list[tuple[str, str, str]]] = [
    (r"\[.*?\]", "bracket_atom", "bracket atoms (e.g. [NH3+], [13C]) are not supported"),
    (r"%\d{2}", "extended_ring", "extended ring closures (%nn) are not supported"),
    (r"\.", "disconnected", "disconnected compounds (.) are not supported"),
    (r">", "reaction", "reaction SMILES (>) are not supported"),
    (r"\*", "wildcard", "wildcard atoms (*) are not supported"),
    (r":\d", "atom_map", "atom-mapping (:n) is not supported"),
]


def _check_unsupported(smiles: str) -> None:
    """Scan *smiles* for unsupported syntax and raise early if found.

    Args:
        smiles: Raw SMILES string before tokenization.

    Raises:
        UnsupportedSMILESFeatureError: If a known-unsupported pattern is found.
    """
    for pattern, feature, detail in _UNSUPPORTED_PATTERNS:
        if re.search(pattern, smiles):
            raise UnsupportedSMILESFeatureError(feature, detail)


# ---------------------------------------------------------------------------
# Public tokenizer
# ---------------------------------------------------------------------------

def tokenize_smiles(smiles: str) -> list[str]:
    """Tokenize a SMILES string into a list of string tokens.

    Each token is one of:

    * An atom symbol (``"C"``, ``"Cl"``, ``"c"`` …)
    * A bond character (``"="``, ``"#"``, ``":"``, ``"/"``, ``"\\"``).
      Note: implicit single bonds are **not** emitted as tokens.
    * A branch delimiter (``"("`` or ``")"``)
    * A ring-closure digit (``"1"`` … ``"9"``)
    * A chirality marker (``"@"`` or ``"@@"``)

    Args:
        smiles: Input SMILES string.

    Returns:
        Ordered list of SMILES tokens.

    Raises:
        SMILESValidationError: If *smiles* is empty.
        UnsupportedSMILESFeatureError: If the string contains syntax not
            handled by this parser (isotopes, bracket atoms, wildcards, …).
        SMILESTokenizationError: If an unrecognised character is encountered.

    Example:
        >>> tokenize_smiles("CC(=O)O")
        ['C', 'C', '(', '=', 'O', ')', 'O']
        >>> tokenize_smiles("c1ccccc1")
        ['c', '1', 'c', 'c', 'c', 'c', 'c', '1']
    """
    if not smiles or not smiles.strip():
        raise SMILESValidationError("SMILES string must not be empty.")

    _check_unsupported(smiles)

    tokens: list[str] = []
    pos = 0
    length = len(smiles)

    while pos < length:
        char = smiles[pos]

        # -- Chirality (must come before '@' being treated as unknown) ------
        if char == "@":
            if smiles[pos: pos + 2] == "@@":
                tokens.append("@@")
                pos += 2
            else:
                tokens.append("@")
                pos += 1
            continue

        # -- Two-character atom symbols -------------------------------------
        if pos + 1 < length:
            two = smiles[pos: pos + 2]
            if two in TWO_CHAR_ATOMS:
                tokens.append(two)
                pos += 2
                continue

        # -- Single-character atom symbols ----------------------------------
        if char.isalpha():
            symbol = char
            if symbol not in SUPPORTED_ATOMS:
                raise SMILESTokenizationError(
                    f"Unrecognised atom symbol '{symbol}'", position=pos
                )
            tokens.append(symbol)
            pos += 1
            continue

        # -- Bond characters ------------------------------------------------
        if char in BOND_CHARS:
            tokens.append(char)
            pos += 1
            continue

        # -- Branch delimiters ----------------------------------------------
        if char in BRANCH_CHARS:
            tokens.append(char)
            pos += 1
            continue

        # -- Ring-closure digits --------------------------------------------
        if char in RING_DIGITS:
            tokens.append(char)
            pos += 1
            continue

        # -- Digit '0' is not a valid ring index in this subset -------------
        if char == "0":
            raise SMILESTokenizationError(
                "Ring index '0' is not supported; use 1–9.", position=pos
            )

        # -- Anything else is unrecognised ----------------------------------
        raise SMILESTokenizationError(
            f"Unrecognised character '{char}'", position=pos
        )

    return tokens

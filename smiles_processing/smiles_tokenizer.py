#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# smiles_processing/smiles_tokenizer.py

"""Tokenizer for a subset of the SMILES string format.

Converts a raw SMILES string into a flat list of typed tokens that the
parser can consume one-by-one.  Bracket atoms (e.g. [NH4+], [O-], [nH],
[C@H], [Fe+2]) are now supported and emitted as a single opaque token
that the parser unpacks.

Only the subset described in the DrugKit specification is supported;
anything outside that subset raises an explicit exception rather than
being silently ignored -> unless strict=Fals  is passed, in which case
unsupported tokens are skipped with a warning.
"""

import re
import logging
from typing import Final

from smiles_processing.smiles_errors import (
    SMILESTokenizationError,
    SMILESValidationError,
    UnsupportedSMILESFeatureError,
)

logger = logging.getLogger(__name__)


TWO_CHAR_ATOMS: Final[frozenset[str]] = frozenset({"Cl", "Br", "Si"})
SUPPORTED_ATOMS: Final[frozenset[str]] = frozenset({
    "C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "H", "Si",
    "c", "n", "o", "s",
})

BRACKET_ATOM_SYMBOLS: Final[frozenset[str]] = frozenset({
    "C", "N", "O", "S", "P", "F", "Cl", "Br", "I", "B", "H", "Si",
    "c", "n", "o", "s",
    # metals / extended organic subset
    "Fe", "Cu", "Zn", "Mg", "Ca", "Na", "K", "Li", "Al", "Se",
    "As", "Ge", "Sn", "Pb", "Co", "Ni", "Mn", "Cr", "Mo", "V",
    "Ti", "Pt", "Pd", "Au", "Ag", "Hg", "Bi", "Sb", "Te",
})

#: Explicit bond characters.
BOND_CHARS: Final[frozenset[str]] = frozenset({"=", "#", ":", "/", "\\"})

#: Characters that open/close branches.
BRANCH_CHARS: Final[frozenset[str]] = frozenset({"(", ")"})

#: Ring-closure digits (1–9 only per spec).
RING_DIGITS: Final[frozenset[str]] = frozenset("123456789")

_BRACKET_RE = re.compile(
    r"^\[(?P<isotope>\d+)?"          # optional isotope mass number
    r"(?P<symbol>[A-Z][a-z]?|[a-z])" # element symbol (1–2 chars, case-sensitive)
    r"(?P<chiral>@@|@)?"              # optional chirality
    r"(?P<hcount>H\d*)?"              # optional explicit H (H, H2, H3 …)
    r"(?P<charge>[+-]\d?|[+-]{2})?"   # optional formal charge (+, -, +2, -2, ++, --)
    r"(?::\d+)?"                       # optional atom map (ignored)
    r"\]$"
)

# ---------------------------------------------------------------------------
# Patterns that signal unsupported features (checked in strict mode only)
# ---------------------------------------------------------------------------

_UNSUPPORTED_PATTERNS: Final[list[tuple[str, str, str]]] = [
    (r"%\d{2}", "extended_ring", "extended ring closures (%nn) are not supported"),
    (r"\.", "disconnected", "disconnected compounds (.) are not supported"),
    (r">", "reaction", "reaction SMILES (>) are not supported"),
    (r"\*", "wildcard", "wildcard atoms (*) are not supported"),
]


def _check_unsupported(smiles: str) -> None:
    """Scan *smiles* for unsupported syntax and raise early if found.

    Bracket atoms are now explicitly allowed and are NOT checked here.

    Args:
        smiles: Raw SMILES string before tokenization.

    Raises:
        UnsupportedSMILESFeatureError: If a known-unsupported pattern is found.
    """
    for pattern, feature, detail in _UNSUPPORTED_PATTERNS:
        if re.search(pattern, smiles):
            raise UnsupportedSMILESFeatureError(feature, detail)


# ---------------------------------------------------------------------------
# Bracket atom parsing helper
# ---------------------------------------------------------------------------

def parse_bracket_atom(bracket_content: str) -> dict:
    """Parse the contents of a [...  bracket atom into a feature dict.

    The *bracket_content* argument should be the full bracket expression
    including the square brackets (e.g. "[NH4+] ).

    Returns a dict with keys:

    * token_type  -> always "bracket_atom 
    * symbol  -> element symbol (uppercase for non-aromatic, lowercase for aromatic)
    * aromatic  -> True if the symbol was lowercase
    * chirality  -> "@  or "@@  or None 
    * explicit_h  -> integer number of explicit hydrogens (0 if absent)
    * formal_charge  -> integer formal charge (0 if absent)
    * isotop  -> integer isotope (0 if absent; semantics are ignored downstream)

    Raises:
        SMILESTokenizationError: If the bracket content cannot be parsed.
    """
    m = _BRACKET_RE.match(bracket_content)
    if m is None:
        raise SMILESTokenizationError(
            f"Cannot parse bracket atom: {bracket_content!r}"
        )

    raw_symbol = m.group("symbol")
    aromatic = raw_symbol[0].islower()
    symbol = raw_symbol.upper()

    # Validate the element is in our supported set
    # (use the raw symbol for case-sensitive lookup)
    if raw_symbol.upper() not in {s.upper() for s in BRACKET_ATOM_SYMBOLS}:
        raise SMILESTokenizationError(
            f"Unsupported element in bracket atom: {raw_symbol!r}"
        )

    # Chirality
    chirality = m.group("chiral") or None

    # Explicit hydrogens: "H" means 1, "H2" means 2, etc.
    hcount_raw = m.group("hcount")
    if hcount_raw is None:
        explicit_h = 0
    elif hcount_raw == "H":
        explicit_h = 1
    else:
        explicit_h = int(hcount_raw[1:])  # strip leading "H"

    # Formal charge: +, -, +2, -2, ++, --
    charge_raw = m.group("charge")
    if charge_raw is None:
        formal_charge = 0
    elif charge_raw == "++":
        formal_charge = 2
    elif charge_raw == "--":
        formal_charge = -2
    elif charge_raw in ("+", "-"):
        formal_charge = 1 if charge_raw == "+" else -1
    else:
        formal_charge = int(charge_raw)  # e.g. "+2" or "-1"

    # Isotope (stored but semantics are ignored)
    isotope_raw = m.group("isotope")
    isotope = int(isotope_raw) if isotope_raw else 0

    return {
        "token_type": "bracket_atom",
        "symbol": symbol,
        "aromatic": aromatic,
        "chirality": chirality,
        "explicit_h": explicit_h,
        "formal_charge": formal_charge,
        "isotope": isotope,
        "raw": bracket_content,
    }


# ---------------------------------------------------------------------------
# Public tokenizer
# ---------------------------------------------------------------------------

def tokenize_smiles(smiles: str, strict: bool = True) -> list[str | dict]:
    """Tokenize a SMILES string into a list of tokens.

    Each token is one of:

    * An atom symbol string  "C , "Cl , "c  …)
    * A bond character string  "= , "# , ": , "/ , "\\ ).
      Implicit single bonds are **not** emitted as tokens.
    * A branch delimiter string  "(  or ") )
    * A ring-closure digit string  "1  … "9 )
    * A chirality marker string  "@  or "@@ )
    * A bracket-atom **dict** (see :func:`parse_bracket_atom`) for [... 
      expressions.

    Args:
        smiles: Input SMILES string.
        strict: If Tru  (default), raise on any unsupported feature.
            If Fals , skip unrecognised characters with a warning ->
            intended for billion-scale preprocessing where crashes are
            unacceptable.

    Returns:
        Ordered list of SMILES tokens (strings or bracket-atom dicts).

    Raises:
        SMILESValidationError: If *smiles* is empty.
        UnsupportedSMILESFeatureError: If the string contains unsupported
            syntax and strict=Tru .
        SMILESTokenizationError: If an unrecognised character is encountered
            and strict=Tru .

    Examples:
        >>> tokenize_smiles("CC(=O)O")
        ['C', 'C', '(', '=', 'O', ')', 'O']
        >>> tokenize_smiles("c1ccccc1")
        ['c', '1', 'c', 'c', 'c', 'c', 'c', '1']
        >>> tok = tokenize_smiles("[NH4+]")
        >>> tok[0]['symbol'], tok[0]['formal_charge'], tok[0]['explicit_h']
        ('N', 1, 4)
    """
    if not smiles or not smiles.strip():
        raise SMILESValidationError("SMILES string must not be empty.")

    if strict:
        _check_unsupported(smiles)

    tokens: list[str | dict] = []
    pos = 0
    length = len(smiles)

    while pos < length:
        char = smiles[pos]

        if char == "[":
            end = smiles.find("]", pos)
            if end == -1:
                if strict:
                    raise SMILESTokenizationError(
                        "Unclosed bracket atom '[' has no matching ']'",
                        position=pos,
                    )
                else:
                    logger.warning(
                        "Tolerant mode: unclosed '[' at pos %d -> skipping rest", pos
                    )
                    break
            bracket_str = smiles[pos: end + 1]
            try:
                token = parse_bracket_atom(bracket_str)
            except SMILESTokenizationError as exc:
                if strict:
                    raise
                logger.warning("Tolerant mode: skipping unrecognised bracket atom %r -> %s", bracket_str, exc)
                pos = end + 1
                continue
            tokens.append(token)
            pos = end + 1
            continue

        if char == "@":
            if smiles[pos: pos + 2] == "@@":
                tokens.append("@@")
                pos += 2
            else:
                tokens.append("@")
                pos += 1
            continue

        if pos + 1 < length:
            two = smiles[pos: pos + 2]
            if two in TWO_CHAR_ATOMS:
                tokens.append(two)
                pos += 2
                continue

        if char.isalpha():
            symbol = char
            if symbol not in SUPPORTED_ATOMS:
                if strict:
                    raise SMILESTokenizationError(
                        f"Unrecognised atom symbol '{symbol}'", position=pos
                    )
                else:
                    logger.warning(
                        "Tolerant mode: skipping unrecognised atom '%s' at pos %d",
                        symbol, pos,
                    )
                    pos += 1
                    continue
            tokens.append(symbol)
            pos += 1
            continue

        if char in BOND_CHARS:
            tokens.append(char)
            pos += 1
            continue

        if char in BRANCH_CHARS:
            tokens.append(char)
            pos += 1
            continue

        if char in RING_DIGITS:
            tokens.append(char)
            pos += 1
            continue

        if char == "0":
            if strict:
                raise SMILESTokenizationError(
                    "Ring index '0' is not supported; use 1–9.", position=pos
                )
            else:
                logger.warning("Tolerant mode: skipping ring index '0' at pos %d", pos)
                pos += 1
                continue

        if strict:
            raise SMILESTokenizationError(
                f"Unrecognised character '{char}'", position=pos
            )
        else:
            logger.warning(
                "Tolerant mode: skipping unrecognised character %r at pos %d",
                char, pos,
            )
            pos += 1

    return tokens

#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""Custom exceptions for the DrugKit SMILES parser."""


class SMILESError(Exception):
    """Base exception for all SMILES parsing errors."""


class SMILESTokenizationError(SMILESError):
    """Raised when the tokenizer encounters an unrecognized or unsupported token.

    Args:
        message: Human-readable description of the error.
        position: Index in the SMILES string where the error occurred.
    """

    def __init__(self, message: str, position: int | None = None) -> None:
        self.position = position
        location = f" at position {position}" if position is not None else ""
        super().__init__(f"{message}{location}")


class SMILESParseError(SMILESError):
    """Raised when the parser encounters structurally invalid SMILES syntax.

    Examples include unmatched parentheses or unclosed ring closures.
    """


class SMILESValidationError(SMILESError):
    """Raised when a SMILES string fails semantic validation before parsing."""


class UnsupportedSMILESFeatureError(SMILESError):
    """Raised when the input uses a valid SMILES feature not supported by this parser.

    Args:
        feature: Short name of the unsupported feature (e.g. 'isotope', 'wildcard').
    """

    def __init__(self, feature: str, detail: str = "") -> None:
        self.feature = feature
        suffix = f": {detail}" if detail else ""
        super().__init__(f"Unsupported SMILES feature '{feature}'{suffix}")

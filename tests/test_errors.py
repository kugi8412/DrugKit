# tests/test_errors.py

import pytest

from smiles_processing.smiles_parser import parse_smiles


INVALID = [
    "",
    "C(",
    "C1CC",
    "C$C",
    "C..C",
    "[NH4+]",
]


def test_invalid_smiles():
    for smi in INVALID:
        print("\nTesting:", repr(smi))

        with pytest.raises(Exception):
            parse_smiles(smi)
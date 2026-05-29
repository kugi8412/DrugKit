# tests/test_tokenizer.py

from smiles_processing.smiles_tokenizer import tokenize_smiles


CASES = {
    "CCO": ["C", "C", "O"],
    "C=C": ["C", "=", "C"],
    "C#N": ["C", "#", "N"],
    "c1ccccc1": ["c", "1", "c", "c", "c", "c", "c", "1"],
    "CC(=O)O": ["C", "C", "(", "=", "O", ")", "O"],
}


def test_tokenizer():
    for smiles, expected in CASES.items():
        result = tokenize_smiles(smiles)

        print("\nSMILES:", smiles)
        print("Expected:", expected)
        print("Result:  ", result)

        assert result == expected
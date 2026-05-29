# tests/test_features.py
from smiles_processing.smiles_parser import parse_smiles
from smiles_processing.smiles_features import extract_features

TESTS = [
    "CC",
    "C=C",
    "C#N",
    "c1ccccc1",
]


def test_features():
    for smi in TESTS:
        graph = parse_smiles(smi)
        features = extract_features(graph)

        print("\n", "=" * 60)
        print("SMILES:", smi)

        print("\nATOM FEATURES")
        for atom in features["atoms"]:
            print(atom)

            assert "symbol" in atom
            assert "hybridization" in atom

        print("\nBOND FEATURES")
        for bond in features["bonds"]:
            print(bond)

            assert "bond_type" in bond
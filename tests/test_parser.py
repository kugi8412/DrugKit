# tests/test_parser.py

from smiles_processing.smiles_parser import parse_smiles

TESTS = [
    "CC",
    "C=C",
    "c1ccccc1",
    "CC(=O)O",
]


def test_parser():
    for smi in TESTS:
        graph = parse_smiles(smi)

        print("\n", "=" * 60)
        print("SMILES:", smi)

        print("Atoms:", len(graph["atoms"]))
        print("Bonds:", len(graph["bonds"]))

        assert len(graph["atoms"]) > 0
        assert len(graph["bonds"]) > 0

        for atom in graph["atoms"]:
            print(atom)

        for bond in graph["bonds"]:
            print(bond)
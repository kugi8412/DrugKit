#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from rdkit import Chem

from smiles_processing.smiles_parser import parse_smiles


TESTS = [
    "CC",
    "C=C",
    "C#N",
    "c1ccccc1",
    "CC(=O)O",
]


def test_rdkit_atom_and_bond_counts():
    for smi in TESTS:
        print("\n", "=" * 70)
        print("SMILES:", smi)

        custom = parse_smiles(smi)
        rdkit_mol = Chem.MolFromSmiles(smi)

        custom_atoms = len(custom["atoms"])
        rdkit_atoms = rdkit_mol.GetNumAtoms()

        custom_bonds = len(custom["bonds"])
        rdkit_bonds = rdkit_mol.GetNumBonds()

        print("Custom atoms:", custom_atoms)
        print("RDKit atoms:", rdkit_atoms)

        print("Custom bonds:", custom_bonds)
        print("RDKit bonds:", rdkit_bonds)

        assert custom_atoms == rdkit_atoms
        assert custom_bonds == rdkit_bonds

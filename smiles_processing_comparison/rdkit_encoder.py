#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
RDKit-based feature encoder extracted verbatim from the two target scripts
(siamese_GNN/improved_train.py and final_docking/virtual_screeing.py).

The functions here are IDENTICAL to those in the target scripts â€” no changes
whatsoever.  This is the ground-truth encoder we are comparing against.
"""

from __future__ import annotations

import torch
from rdkit import Chem
from torch_geometric.data import Data


PERMITTED_ATOMS = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B', 'H']


def one_hot_encoding(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def get_atom_features(atom):
    features = one_hot_encoding(atom.GetSymbol(), PERMITTED_ATOMS)
    features += one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4])
    features += one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    features += one_hot_encoding(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
    features += one_hot_encoding(str(atom.GetHybridization()), ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2'])
    features += [1 if atom.GetIsAromatic() else 0]
    features += [atom.GetMass() * 0.01]
    try:
        chiral = str(atom.GetChiralTag())
        features += one_hot_encoding(chiral, ['CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW'])
    except Exception:
        features += [0, 0, 1]
    return features


def get_bond_features(bond):
    bt = bond.GetBondType()
    features = [
        1 if bt == Chem.rdchem.BondType.SINGLE   else 0,
        1 if bt == Chem.rdchem.BondType.DOUBLE   else 0,
        1 if bt == Chem.rdchem.BondType.TRIPLE   else 0,
        1 if bt == Chem.rdchem.BondType.AROMATIC else 0,
    ]
    features += [1 if bond.GetIsConjugated() else 0]
    features += [1 if bond.IsInRing() else 0]
    stereo = str(bond.GetStereo())
    features += one_hot_encoding(stereo, ['STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS'])
    return features


def smiles_to_graph_rdkit(smiles: str) -> Data | None:
    """Verbatim copy of smiles_to_graph_gine from the target scripts."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        atom_feats = [get_atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(atom_feats, dtype=torch.float)
        rows, cols, edge_feats = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            b_feat = get_bond_features(bond)
            rows += [start, end]; cols += [end, start]
            edge_feats += [b_feat, b_feat]
        if not rows:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr  = torch.empty((0, 11), dtype=torch.float)
        else:
            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            edge_attr  = torch.tensor(edge_feats, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
    except Exception:
        return None


def get_raw_atom_features(smiles: str) -> list[list[int]] | None:
    """Return per-atom feature lists (not tensors) for comparison."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return [get_atom_features(a) for a in mol.GetAtoms()]


def get_raw_bond_features(smiles: str) -> list[list[int]] | None:
    """Return per-bond feature lists (undirected, one per bond) for comparison."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return [get_bond_features(b) for b in mol.GetBonds()]

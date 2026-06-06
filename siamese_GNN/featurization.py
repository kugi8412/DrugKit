# -*- coding: utf-8 -*-
"""SMILES -> PyTorch Geometric graph featurization for the GINE encoder.

Extracted verbatim (behavior-preserving) from improved_train.py so it can be
reused by both the trainer and the active-learning loop.
"""

from typing import List, Optional, Tuple

import torch
from rdkit import Chem
from torch_geometric.data import Data

PERMITTED_ATOMS = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B', 'H']


def one_hot_encoding(value, choices) -> List[int]:
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def get_atom_features(atom) -> List[float]:
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


def get_bond_features(bond) -> List[int]:
    bt = bond.GetBondType()
    features = [
        1 if bt == Chem.rdchem.BondType.SINGLE else 0,
        1 if bt == Chem.rdchem.BondType.DOUBLE else 0,
        1 if bt == Chem.rdchem.BondType.TRIPLE else 0,
        1 if bt == Chem.rdchem.BondType.AROMATIC else 0,
    ]
    features += [1 if bond.GetIsConjugated() else 0]
    features += [1 if bond.IsInRing() else 0]
    stereo = str(bond.GetStereo())
    features += one_hot_encoding(stereo, ['STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS'])
    return features


def _empty_edge_dim() -> int:
    ref = Chem.MolFromSmiles("CC").GetBondWithIdx(0)
    return len(get_bond_features(ref))


def smiles_to_graph_gine(smiles: str, selectivity: Optional[float] = None,
                         is_elite: bool = False) -> Optional[Data]:
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    atom_feats = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float)

    rows, cols, edge_feats = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        b_feat = get_bond_features(bond)
        rows += [start, end]
        cols += [end, start]
        edge_feats += [b_feat, b_feat]

    if not rows:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, _empty_edge_dim()), dtype=torch.float)
    else:
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)

    y = torch.tensor([selectivity], dtype=torch.float) if selectivity is not None else None
    elite_flag = torch.tensor([1.0 if is_elite else 0.0], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y,
                is_elite=elite_flag, smiles=smiles)


def feature_dims() -> Tuple[int, int]:
    """Return (node_feature_dim, edge_feature_dim) using a reference molecule."""
    dummy = smiles_to_graph_gine("CCO")
    return dummy.x.shape[1], dummy.edge_attr.shape[1]

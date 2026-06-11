#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from siamese_GNN.featurization import smiles_to_graph_gine, feature_dims


def test_graph_has_expected_dims():
    g = smiles_to_graph_gine("CCO", selectivity=-1.5, is_elite=True)
    assert g is not None
    node_dim, edge_dim = feature_dims()
    assert g.x.shape[1] == node_dim
    assert g.edge_attr.shape[1] == edge_dim
    assert g.x.shape[0] == 3            # C, C, O
    assert float(g.y.item()) == -1.5
    assert float(g.is_elite.item()) == 1.0


def test_invalid_smiles_returns_none():
    assert smiles_to_graph_gine("not_a_smiles") is None


def test_single_atom_has_empty_edges():
    g = smiles_to_graph_gine("[Ne]")
    assert g is not None
    assert g.edge_index.shape == (2, 0)
    _, edge_dim = feature_dims()
    assert g.edge_attr.shape == (0, edge_dim)

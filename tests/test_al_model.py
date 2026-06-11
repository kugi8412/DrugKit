#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
from torch_geometric.data import Batch

from siamese_GNN.featurization import smiles_to_graph_gine, feature_dims
from siamese_GNN.model import SiameseRankNet, WeightedRankNetLoss, enable_mc_dropout


def _batch(smiles_list):
    graphs = [smiles_to_graph_gine(s, selectivity=0.0) for s in smiles_list]
    return Batch.from_data_list(graphs)


def test_forward_one_returns_scalar_per_graph():
    node_dim, edge_dim = feature_dims()
    model = SiameseRankNet(node_dim, edge_dim, hidden_dim=32, dropout=0.3)
    model.eval()
    batch = _batch(["CCO", "c1ccccc1", "CCN"])
    out = model.forward_one(batch)
    assert out.shape == (3, 1)


def test_enable_mc_dropout_activates_only_dropout():
    node_dim, edge_dim = feature_dims()
    model = SiameseRankNet(node_dim, edge_dim, hidden_dim=32, dropout=0.5)
    enable_mc_dropout(model)
    dropouts = [m for m in model.modules() if isinstance(m, torch.nn.Dropout)]
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm1d)]
    assert dropouts and all(m.training for m in dropouts)
    assert bns and all(not m.training for m in bns)


def test_mc_dropout_makes_outputs_vary():
    torch.manual_seed(0)
    node_dim, edge_dim = feature_dims()
    model = SiameseRankNet(node_dim, edge_dim, hidden_dim=32, dropout=0.5)
    batch = _batch(["CCO", "c1ccccc1", "CCN", "CCCl"])
    enable_mc_dropout(model)
    with torch.no_grad():
        a = model.forward_one(batch)
        b = model.forward_one(batch)
    assert not torch.allclose(a, b)


def test_loss_is_weighted():
    crit = WeightedRankNetLoss()
    s1 = torch.tensor([[2.0]])
    s2 = torch.tensor([[0.0]])
    target = torch.tensor([[1.0]])
    w1 = crit(s1, s2, target, torch.tensor([[1.0]]))
    w2 = crit(s1, s2, target, torch.tensor([[5.0]]))
    assert torch.isclose(w2, w1 * 5.0)

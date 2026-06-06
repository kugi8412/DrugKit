import numpy as np
import torch

from siamese_GNN.featurization import smiles_to_graph_gine, feature_dims
from siamese_GNN.model import SiameseRankNet
from active_learning.uncertainty import mc_dropout_predict


def _graphs():
    return [smiles_to_graph_gine(s, selectivity=0.0)
            for s in ["CCO", "c1ccccc1", "CCN", "CCCl"]]


def test_returns_mean_and_std_per_graph():
    node_dim, edge_dim = feature_dims()
    model = SiameseRankNet(node_dim, edge_dim, hidden_dim=32, dropout=0.5)
    means, stds = mc_dropout_predict(model, _graphs(), mc_samples=8,
                                     device="cpu", batch_size=2, seed=0)
    assert means.shape == (4,)
    assert stds.shape == (4,)
    assert np.all(stds > 0)


def test_reproducible_under_seed():
    node_dim, edge_dim = feature_dims()
    model = SiameseRankNet(node_dim, edge_dim, hidden_dim=32, dropout=0.5)
    m1, s1 = mc_dropout_predict(model, _graphs(), 8, "cpu", 2, seed=123)
    m2, s2 = mc_dropout_predict(model, _graphs(), 8, "cpu", 2, seed=123)
    assert np.allclose(m1, m2)
    assert np.allclose(s1, s2)

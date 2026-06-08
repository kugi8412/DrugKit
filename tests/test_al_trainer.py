import logging

import pandas as pd

from siamese_GNN.featurization import feature_dims
from siamese_GNN.trainer import DEFAULT_TRAIN_CFG, build_labeled_graphs, train_ranknet

SMILES = ["CCO", "CCN", "CCCl", "c1ccccc1", "CCCC", "CCOCC", "CCC=O", "CC(=O)O"]


def _df():
    return pd.DataFrame({
        "Name": [f"c{i}" for i in range(len(SMILES))],
        "SMILES": SMILES,
        "selectivity": [-1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        "Cluster_ID": ["Unknown"] * len(SMILES),
    })


def test_build_labeled_graphs_marks_elites():
    graphs, clusters = build_labeled_graphs(
        _df(), sel_col="selectivity", cluster_col="Cluster_ID", elite_count=2)
    assert len(graphs) == len(SMILES)
    elite = [g for g in graphs if float(g.is_elite.item()) > 0.5]
    assert len(elite) == 2


def test_train_ranknet_returns_model_and_history():
    cfg = dict(DEFAULT_TRAIN_CFG)
    cfg.update(epochs=2, hidden_dim=16, batch_size=8)
    graphs, clusters = build_labeled_graphs(
        _df(), sel_col="selectivity", cluster_col="Cluster_ID",
        elite_count=cfg["elite_count"])
    node_dim, edge_dim = feature_dims()
    model, history = train_ranknet(
        graphs, clusters, node_dim, edge_dim, cfg,
        device="cpu", logger=logging.getLogger("t"))
    assert model is not None
    assert len(history["train_loss"]) == 2
    assert "val_rho" in history

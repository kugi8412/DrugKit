#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import tempfile

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch

from smiles_processing import smiles_to_pyg, batch_smiles_to_pyg, ATOM_FEATURE_DIM, BOND_FEATURE_DIM
from siamese_GNN.featurization import smiles_to_graph_gine, feature_dims
from siamese_GNN.model import SiameseRankNet, enable_mc_dropout
from siamese_GNN.trainer import build_labeled_graphs, train_ranknet
from active_learning.acquisition import select_top_uncertain
from active_learning.selectivity import compute_selectivity
from active_learning.uncertainty import mc_dropout_predict



SAMPLE_SMILES = [
    "CCO",          # ethanol
    "c1ccccc1",    # benzene
    "CC(=O)O",     # acetic acid
    "CCN",          # ethylamine
    "CC(C)O",      # isopropanol
    "c1ccc(O)cc1", # phenol
    "CCCC",        # butane
    "CC=O",        # acetaldehyde
]


@pytest.fixture
def sample_labeled_df():
    """Create a small labeled DataFrame mimicking docking output."""
    rows = []
    for i, smi in enumerate(SAMPLE_SMILES):
        rows.append({
            "Name": f"mol_{i}",
            "SMILES": smi,
            "Target": "TARGET_A",
            "Pocket_ID": "p1",
            "Energy": -6.0 - i * 0.3,
        })
        rows.append({
            "Name": f"mol_{i}",
            "SMILES": smi,
            "Target": "OFFTARGET_B",
            "Pocket_ID": "q1",
            "Energy": -5.0 - i * 0.1,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def trained_model():
    """Train a tiny model for testing."""
    node_dim, edge_dim = feature_dims()
    model = SiameseRankNet(node_dim, edge_dim, hidden_dim=32, dropout=0.3)

    # Create minimal training data
    graphs = []
    for i, smi in enumerate(SAMPLE_SMILES):
        g = smiles_to_graph_gine(smi, selectivity=float(i) * 0.5)
        if g is not None:
            g.is_elite = torch.tensor([0.0])
            graphs.append(g)

    # Simple training loop (2 epochs)
    from torch_geometric.loader import DataLoader
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model.train()
    loader = DataLoader(graphs, batch_size=4, shuffle=True)
    for _ in range(2):
        for batch in loader:
            pred = model.forward_one(batch)
            loss = pred.mean()  # dummy loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


class TestSMILESProcessing:
    """Tests for SMILES parsing and PyG conversion."""

    def test_smiles_to_pyg_basic(self):
        data = smiles_to_pyg("CCO")
        assert data is not None
        assert data.x.shape[1] == ATOM_FEATURE_DIM
        assert data.edge_attr.shape[1] == BOND_FEATURE_DIM
        assert data.x.shape[0] == 3  # C, C, O

    def test_batch_conversion(self):
        results = batch_smiles_to_pyg(SAMPLE_SMILES)
        assert len(results) == len(SAMPLE_SMILES)
        assert all(r is not None for r in results)

    def test_invalid_smiles_returns_none(self):
        result = smiles_to_pyg("INVALID_NOT_SMILES_XYZ")
        assert result is None

    def test_rdkit_featurization_parity(self):
        """Verify RDKit-based featurization produces valid graphs."""
        for smi in SAMPLE_SMILES:
            g = smiles_to_graph_gine(smi)
            assert g is not None
            assert g.x.ndim == 2
            assert g.edge_index.shape[0] == 2
            assert g.edge_attr.ndim == 2

    def test_feature_dims_consistency(self):
        node_dim, edge_dim = feature_dims()
        for smi in SAMPLE_SMILES:
            g = smiles_to_graph_gine(smi)
            assert g.x.shape[1] == node_dim
            if g.edge_attr.numel() > 0:
                assert g.edge_attr.shape[1] == edge_dim


class TestModelArchitecture:
    """Tests for the GINE Siamese RankNet."""

    def test_forward_produces_scores(self):
        node_dim, edge_dim = feature_dims()
        model = SiameseRankNet(node_dim, edge_dim, hidden_dim=64, dropout=0.2)
        model.eval()

        graphs = [smiles_to_graph_gine(s) for s in SAMPLE_SMILES[:4]]
        batch = Batch.from_data_list(graphs)

        with torch.no_grad():
            scores = model.forward_one(batch)

        assert scores.shape == (4, 1)
        assert torch.isfinite(scores).all()

    def test_siamese_pair_forward(self):
        node_dim, edge_dim = feature_dims()
        model = SiameseRankNet(node_dim, edge_dim, hidden_dim=64, dropout=0.2)
        model.eval()

        g1 = [smiles_to_graph_gine(s) for s in SAMPLE_SMILES[:3]]
        g2 = [smiles_to_graph_gine(s) for s in SAMPLE_SMILES[3:6]]
        b1 = Batch.from_data_list(g1)
        b2 = Batch.from_data_list(g2)

        with torch.no_grad():
            s1, s2 = model(b1, b2)

        assert s1.shape == (3, 1)
        assert s2.shape == (3, 1)


class TestMCDropoutUncertainty:
    """Tests for Monte Carlo dropout uncertainty estimation."""

    def test_mc_dropout_returns_mean_and_std(self, trained_model):
        graphs = [smiles_to_graph_gine(s) for s in SAMPLE_SMILES[:4]]
        means, stds = mc_dropout_predict(
            trained_model, graphs, mc_samples=10, device="cpu", seed=42
        )
        assert len(means) == 4
        assert len(stds) == 4
        assert all(np.isfinite(means))
        assert all(stds >= 0)

    def test_mc_dropout_with_empty_input(self, trained_model):
        means, stds = mc_dropout_predict(
            trained_model, [], mc_samples=5, device="cpu"
        )
        assert len(means) == 0
        assert len(stds) == 0


class TestAcquisition:
    """Tests for acquisition function."""

    def test_selects_highest_uncertainty(self):
        names = ["a", "b", "c", "d", "e"]
        uncertainties = np.array([0.1, 0.5, 0.3, 0.9, 0.2])
        selected = select_top_uncertain(names, uncertainties, k=2, exclude=set())
        assert selected == ["d", "b"]

    def test_respects_exclusion(self):
        names = ["a", "b", "c", "d"]
        uncertainties = np.array([0.9, 0.8, 0.7, 0.6])
        selected = select_top_uncertain(names, uncertainties, k=2, exclude={"a"})
        assert "a" not in selected
        assert len(selected) == 2


class TestSelectivity:
    """Tests for selectivity computation."""

    def test_compute_selectivity(self, sample_labeled_df):
        result = compute_selectivity(
            sample_labeled_df,
            on_targets=["TARGET_A"],
            off_targets=["OFFTARGET_B"],
        )
        assert "selectivity" in result.columns
        assert len(result) > 0
        # Selectivity = off_target_energy - on_target_energy
        # Higher selectivity means more selective for on-target


class TestTraining:
    """Tests for the training pipeline."""

    def test_build_labeled_graphs(self):
        df = pd.DataFrame({
            "Name": ["a", "b", "c", "d"],
            "SMILES": ["CCO", "CCN", "CCC", "CCCC"],
            "selectivity": [-1.0, -2.0, -0.5, -3.0],
            "Cluster_ID": ["c1", "c1", "c2", "c2"],
        })
        graphs, clusters = build_labeled_graphs(
            df, sel_col="selectivity", cluster_col="Cluster_ID", elite_count=2
        )
        assert len(graphs) >= 2
        assert len(clusters) == len(graphs)
        assert all(hasattr(g, "y") for g in graphs)
        assert all(hasattr(g, "is_elite") for g in graphs)

    def test_train_ranknet_converges(self):
        df = pd.DataFrame({
            "Name": [f"m{i}" for i in range(10)],
            "SMILES": ["CCO", "CCN", "CCC", "CCCC", "CC=O",
                       "c1ccccc1", "CCBr", "CCCl", "CCOCC", "CCS"],
            "selectivity": list(np.linspace(-3.0, 0.0, 10)),
            "Cluster_ID": ["c1"] * 5 + ["c2"] * 5,
        })
        graphs, clusters = build_labeled_graphs(
            df, sel_col="selectivity", cluster_col="Cluster_ID", elite_count=2
        )
        node_dim, edge_dim = feature_dims()
        cfg = {
            "elite_penalty": 5.0,
            "val_target_ratio": 0.2,
            "batch_size": 4,
            "epochs": 5,
            "learning_rate": 0.001,
            "hidden_dim": 32,
            "dropout": 0.2,
            "seed": 42,
        }
        model, history = train_ranknet(
            graphs, clusters, node_dim, edge_dim, cfg, device="cpu"
        )
        assert model is not None
        assert len(history["train_loss"]) == 5
        # Loss should not explode
        assert all(np.isfinite(h) for h in history["train_loss"])


# --- Integration Test: Full Pipeline ---

class TestFullPipeline:
    """End-to-end integration test combining all stages."""

    def test_full_active_learning_cycle(self, tmp_path):
        """Test complete cycle: featurize -> train -> predict -> acquire."""
        from active_learning.loop import run_active_learning
        from active_learning.config import DEFAULT_CONFIG

        # Create test data
        pool_path = tmp_path / "pool.csv"
        seed_path = tmp_path / "seed.csv"

        pd.DataFrame({
            "Name": [f"pool_{i}" for i in range(12)],
            "SMILES": ["CCO", "CCN", "CCC", "CCCC", "CC=O", "CCBr",
                       "c1ccccc1", "CCCl", "CCOCC", "CCS", "CCCO", "CCCN"],
        }).to_csv(pool_path, index=False)

        pd.DataFrame({
            "Name": ["seed_0", "seed_1", "seed_2", "seed_3"],
            "SMILES": ["CC", "CCC", "CCCC", "CCCCC"],
            "Target": ["T1"] * 4,
        }).to_csv(seed_path, index=False)

        # Mock docking function
        def mock_dock(records):
            rows = []
            for r in records:
                rows.append({
                    "Name": r["Name"], "SMILES": r["SMILES"],
                    "Target": "TARGET_A", "Pocket_ID": "p1",
                    "Energy": -7.0 - len(r["SMILES"]) * 0.1,
                })
                rows.append({
                    "Name": r["Name"], "SMILES": r["SMILES"],
                    "Target": "OFFTARGET_B", "Pocket_ID": "q1",
                    "Energy": -5.5 - len(r["SMILES"]) * 0.05,
                })
            return pd.DataFrame(rows)

        cfg = dict(DEFAULT_CONFIG)
        cfg.update({
            "pool_file": str(pool_path),
            "seed_file": str(seed_path),
            "on_targets": ["TARGET_A"],
            "off_targets": ["OFFTARGET_B"],
            "rounds": 2,
            "seed_size": 4,
            "acquisition_batch": 3,
            "mc_samples": 5,
            "epochs": 3,
            "hidden_dim": 32,
            "batch_size": 4,
            "output_dir": str(tmp_path / "output"),
        })

        result = run_active_learning(cfg, logger=None, dock_fn=mock_dock, device="cpu")

        # Verify outputs
        assert "labeled" in result
        assert "history" in result
        assert "model" in result
        assert len(result["labeled"]) >= 4  # At least seed compounds
        assert len(result["history"]) == 2  # 2 rounds
        assert os.path.exists(os.path.join(cfg["output_dir"], "labeled.csv"))

    def test_smiles_featurization_to_model_inference(self):
        """Test that featurized SMILES can be passed through the model."""
        node_dim, edge_dim = feature_dims()
        model = SiameseRankNet(node_dim, edge_dim, hidden_dim=64, dropout=0.2)
        model.eval()

        # Featurize
        graphs = [smiles_to_graph_gine(s) for s in SAMPLE_SMILES]
        assert all(g is not None for g in graphs)

        # Batch and predict
        batch = Batch.from_data_list(graphs)
        with torch.no_grad():
            scores = model.forward_one(batch)

        assert scores.shape == (len(SAMPLE_SMILES), 1)
        assert torch.isfinite(scores).all()

        # MC dropout uncertainty
        means, stds = mc_dropout_predict(
            model, graphs, mc_samples=5, device="cpu", seed=0
        )
        assert len(means) == len(SAMPLE_SMILES)
        assert all(np.isfinite(stds))

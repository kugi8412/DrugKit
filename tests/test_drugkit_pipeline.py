#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the unified DrugKit pipeline runner.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from drugkit_pipeline import (
    DrugKitPipeline,
    PipelineConfig,
    PipelineError,
    validate_csv,
    validate_grids,
    validate_smiles,
)

MOCK_SMILES = [
    "CCO", "CCN", "CCC", "CCCC", "CC=O", "CC(=O)O", "c1ccccc1",
    "Oc1ccccc1", "Nc1ccccc1", "CCBr", "CCCl", "CCS", "CCCO",
    "CC(C)O", "CCOCC", "CC(=O)N", "c1ccncc1", "c1ccoc1",
    "CC(C)CC", "CCCCO", "CCCCN", "CCC=O", "CC(C)=O", "CCOC",
]


def _mock_dock_fn(records):
    """Simulate docking: energy proportional to SMILES length + noise."""
    rows = []
    rng = np.random.default_rng(42)
    for r in records:
        n = len(r["SMILES"])
        for target, pocket in [("TARGET_A", "p1"), ("OFFTARGET_B", "q1")]:
            energy = -5.0 - n * 0.15 + rng.normal(0, 0.3)
            rows.append({
                "Name": r["Name"],
                "SMILES": r["SMILES"],
                "Target": target,
                "Pocket_ID": pocket,
                "Energy": round(energy, 3),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def mock_workspace(tmp_path):
    """Create a complete mock workspace."""
    # Pool
    pool_df = pd.DataFrame({
        "Name": [f"mol_{i:03d}" for i in range(len(MOCK_SMILES))],
        "SMILES": MOCK_SMILES,
    })
    pool_path = str(tmp_path / "pool.csv")
    pool_df.to_csv(pool_path, index=False)

    # Seeds
    seeds_df = pd.DataFrame({
        "Name": ["seed_0", "seed_1", "seed_2", "seed_3"],
        "SMILES": ["CC(C)Cc1ccccc1", "CC(=O)Nc1ccccc1", "c1ccc2ccccc2c1", "OC(=O)c1ccccc1"],
    })
    seed_path = str(tmp_path / "seeds.csv")
    seeds_df.to_csv(seed_path, index=False)

    # Grids
    grids = {
        "TARGET_A": [{"id": "p1", "center": [10, 20, 30], "size": [20, 20, 20]}],
        "OFFTARGET_B": [{"id": "q1", "center": [5, 15, 25], "size": [18, 18, 18]}],
    }
    grids_path = str(tmp_path / "grids.json")
    with open(grids_path, "w") as f:
        json.dump(grids, f)

    return {
        "pool_path": pool_path,
        "seed_path": seed_path,
        "grids_path": grids_path,
        "output_dir": str(tmp_path / "output"),
        "tmp_path": tmp_path,
    }


@pytest.fixture
def mock_config(mock_workspace):
    """Create a pipeline config using mock workspace."""
    return PipelineConfig(
        pool_file=mock_workspace["pool_path"],
        seed_file=mock_workspace["seed_path"],
        grids_file=mock_workspace["grids_path"],
        output_dir=mock_workspace["output_dir"],
        on_targets=["TARGET_A"],
        off_targets=["OFFTARGET_B"],
        rounds=2,
        seed_size=4,
        acquisition_batch=3,
        mc_samples=5,
        epochs=3,
        batch_size=8,
        hidden_dim=32,
        dropout=0.3,
        device="cpu",
    )


class TestPipelineConfig:
    def test_from_dict(self):
        cfg = PipelineConfig.from_dict({
            "rounds": 10,
            "epochs": 50,
            "hidden_dim": 256,
            "unknown_key": "ignored",
        })
        assert cfg.rounds == 10
        assert cfg.epochs == 50
        assert cfg.hidden_dim == 256

    def test_defaults(self):
        cfg = PipelineConfig()
        assert cfg.rounds == 5
        assert cfg.epochs == 40
        assert cfg.hidden_dim == 128
        assert cfg.dropout == 0.3

    def test_from_yaml(self, tmp_path):
        yaml_content = """
project:
  data_dir: "my_data"
active_learning:
  rounds: 7
  epochs: 25
  hidden_dim: 64
"""
        path = str(tmp_path / "test_config.yaml")
        with open(path, "w") as f:
            f.write(yaml_content)

        cfg = PipelineConfig.from_yaml(path)
        assert cfg.rounds == 7
        assert cfg.epochs == 25
        assert cfg.hidden_dim == 64
        assert cfg.data_dir == "my_data"

    def test_resolve_device_cpu(self):
        cfg = PipelineConfig(device="cpu")
        assert cfg.resolve_device() == "cpu"

    def test_to_dict(self):
        cfg = PipelineConfig(rounds=3)
        d = cfg.to_dict()
        assert d["rounds"] == 3
        assert "epochs" in d


class TestValidation:
    def test_validate_csv_valid(self, mock_workspace):
        df = validate_csv(mock_workspace["pool_path"], ["Name", "SMILES"], "Pool")
        assert len(df) == len(MOCK_SMILES)

    def test_validate_csv_missing_file(self):
        with pytest.raises(PipelineError, match="not found"):
            validate_csv("/nonexistent.csv", ["Name"], "Test")

    def test_validate_csv_missing_columns(self, tmp_path):
        path = str(tmp_path / "bad.csv")
        pd.DataFrame({"X": [1]}).to_csv(path, index=False)
        with pytest.raises(PipelineError, match="missing required columns"):
            validate_csv(path, ["Name", "SMILES"], "Test")

    def test_validate_grids_valid(self, mock_workspace):
        grids = validate_grids(mock_workspace["grids_path"])
        assert "TARGET_A" in grids
        assert len(grids["TARGET_A"]) == 1

    def test_validate_grids_missing_file(self):
        with pytest.raises(PipelineError, match="not found"):
            validate_grids("/nonexistent.json")

    def test_validate_grids_bad_format(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w") as f:
            json.dump({"T": [{"id": "p1"}]}, f)  # missing center, size
        with pytest.raises(PipelineError, match="missing required key"):
            validate_grids(path)

    def test_validate_smiles_valid(self):
        assert validate_smiles("CCO") is True
        assert validate_smiles("c1ccccc1") is True

    def test_validate_smiles_invalid(self):
        assert validate_smiles("INVALID_XYZ") is False


class TestPipeline:
    def test_validate_inputs(self, mock_config):
        pipeline = DrugKitPipeline(mock_config, dock_fn=_mock_dock_fn)
        pipeline.validate_inputs()  # Should not raise

    def test_validate_fails_bad_targets(self, mock_config):
        mock_config.on_targets = ["NONEXISTENT"]
        pipeline = DrugKitPipeline(mock_config)  # No custom dock_fn
        with pytest.raises(PipelineError, match="not found in grids"):
            pipeline.validate_inputs()

    def test_bootstrap(self, mock_config):
        pipeline = DrugKitPipeline(mock_config, dock_fn=_mock_dock_fn)
        pipeline.validate_inputs()
        labeled = pipeline.run_bootstrap()
        assert len(labeled) >= 2
        assert "selectivity" in labeled.columns
        assert "Name" in labeled.columns

    def test_bootstrap_with_pre_labeled(self, mock_config):
        pre = pd.DataFrame({
            "Name": ["a", "b", "c"],
            "SMILES": ["CCO", "CCN", "CCC"],
            "selectivity": [-2.0, -1.5, -1.0],
            "Cluster_ID": ["c1", "c1", "c2"],
        })
        pipeline = DrugKitPipeline(mock_config, dock_fn=_mock_dock_fn)
        pipeline.validate_inputs()
        labeled = pipeline.run_bootstrap(pre_labeled=pre)
        assert len(labeled) == 3

    def test_full_active_learning(self, mock_config):
        pipeline = DrugKitPipeline(mock_config, dock_fn=_mock_dock_fn)
        result = pipeline.run_active_learning()

        assert "labeled" in result
        assert "model" in result
        assert "history" in result
        assert len(result["labeled"]) >= 4
        assert len(result["history"]) == 2
        assert result["model"] is not None

        # Check outputs saved
        assert os.path.exists(os.path.join(mock_config.output_dir, "labeled.csv"))
        assert os.path.exists(os.path.join(mock_config.output_dir, "al_history.csv"))
        assert os.path.exists(os.path.join(mock_config.output_dir, "model.pth"))

    def test_inference_after_training(self, mock_config, tmp_path):
        pipeline = DrugKitPipeline(mock_config, dock_fn=_mock_dock_fn)
        pipeline.run_active_learning()

        # Create a small library to score
        import csv
        input_file = str(tmp_path / "library.csv")
        output_file = str(tmp_path / "scored.csv")
        with open(input_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["SMILES"])
            for smi in MOCK_SMILES[:10]:
                writer.writerow([smi])

        total = pipeline.run_inference(input_file, output_file)
        assert total == 10
        assert os.path.exists(output_file)

    def test_custom_selectivity_fn(self, mock_config):
        def strict_selectivity(on_best, off_best):
            return on_best - 3.0 * off_best  # Penalize off-target 3x

        pipeline = DrugKitPipeline(
            mock_config, dock_fn=_mock_dock_fn,
            selectivity_fn=strict_selectivity
        )
        result = pipeline.run_active_learning()
        assert len(result["labeled"]) >= 2

    def test_pipeline_error_no_model_inference(self, mock_config, tmp_path):
        pipeline = DrugKitPipeline(mock_config, dock_fn=_mock_dock_fn)
        with pytest.raises(PipelineError, match="No trained model"):
            pipeline.run_inference(str(tmp_path / "x.csv"), str(tmp_path / "y.csv"))

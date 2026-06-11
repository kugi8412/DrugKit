#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for the batch inference and multi-GPU prediction modules.
"""

import csv
import os

import numpy as np
import pytest
import torch

from siamese_GNN.featurization import feature_dims
from siamese_GNN.model import SiameseRankNet
from inference.batch_inference import batch_predict, batch_predict_from_file
from inference.multigpu import MultiGPUPredictor


SAMPLE_SMILES = ["CCO", "c1ccccc1", "CC(=O)O", "CCN", "INVALID_XYZ", "CCCC"]


@pytest.fixture
def model():
    node_dim, edge_dim = feature_dims()
    m = SiameseRankNet(node_dim, edge_dim, hidden_dim=32, dropout=0.2)
    m.eval()
    return m


class TestBatchPredict:
    def test_returns_correct_shapes(self, model):
        scores, uncertainties, valid_mask = batch_predict(
            model, SAMPLE_SMILES, device="cpu", batch_size=4
        )
        assert len(scores) == len(SAMPLE_SMILES)
        assert len(uncertainties) == len(SAMPLE_SMILES)
        assert len(valid_mask) == len(SAMPLE_SMILES)

    def test_invalid_smiles_marked(self, model):
        scores, _, valid_mask = batch_predict(
            model, SAMPLE_SMILES, device="cpu"
        )
        # "INVALID_XYZ" at index 4 should be invalid
        assert valid_mask[4] is False
        assert np.isnan(scores[4])

    def test_valid_smiles_get_scores(self, model):
        scores, _, valid_mask = batch_predict(
            model, SAMPLE_SMILES, device="cpu"
        )
        # "CCO" at index 0 should be valid
        assert valid_mask[0] is True
        assert np.isfinite(scores[0])

    def test_mc_dropout_produces_uncertainty(self, model):
        scores, uncertainties, valid_mask = batch_predict(
            model, SAMPLE_SMILES[:4], device="cpu",
            batch_size=4, mc_samples=10
        )
        # With mc_samples > 0, should get non-zero uncertainties
        valid_unc = uncertainties[np.array(valid_mask[:4])]
        assert all(np.isfinite(valid_unc))

    def test_empty_input(self, model):
        scores, uncertainties, valid_mask = batch_predict(
            model, [], device="cpu"
        )
        assert len(scores) == 0
        assert len(valid_mask) == 0

    def test_all_invalid(self, model):
        scores, _, valid_mask = batch_predict(
            model, ["INVALID1", "INVALID2", "!!!"], device="cpu"
        )
        assert not any(valid_mask)


class TestBatchPredictFromFile:
    def test_processes_file(self, model, tmp_path):
        input_file = str(tmp_path / "input.csv")
        output_file = str(tmp_path / "output.csv")

        with open(input_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["SMILES"])
            for smi in SAMPLE_SMILES * 3:  # 18 compounds
                writer.writerow([smi])

        total = batch_predict_from_file(
            model, input_file, output_file,
            device="cpu", batch_size=4, chunk_size=5
        )
        assert total == 18
        assert os.path.exists(output_file)

        # Check output structure
        with open(output_file, "r") as f:
            reader = csv.reader(f)
            header = next(reader)
            assert "SMILES" in header
            assert "score" in header
            rows = list(reader)
            assert len(rows) == 18


class TestMultiGPUPredictor:
    def test_cpu_fallback(self, model):
        predictor = MultiGPUPredictor(model, gpu_ids=[])
        scores, valid_mask = predictor.predict(SAMPLE_SMILES[:4], batch_size=4)
        assert len(scores) == 4
        assert all(np.isfinite(scores[np.array(valid_mask[:4])]))

    def test_predict_file(self, model, tmp_path):
        predictor = MultiGPUPredictor(model, gpu_ids=[])

        input_file = str(tmp_path / "big_input.csv")
        output_file = str(tmp_path / "big_output.csv")

        with open(input_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["SMILES"])
            for smi in SAMPLE_SMILES * 10:
                writer.writerow([smi])

        total = predictor.predict_file(
            input_file, output_file,
            batch_size=8, chunk_size=20
        )
        assert total == 60
        assert os.path.exists(output_file)

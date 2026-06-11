#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Multi-GPU inference support for billion-scale SMILES screening.

Uses torch.nn.DataParallel or manual device sharding to distribute
inference across multiple GPUs.
"""

import csv
import math
import os
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from siamese_GNN.featurization import smiles_to_graph_gine
from siamese_GNN.model import SiameseRankNet


class _SingleForwardWrapper(nn.Module):
    """Wrap forward_one as forward for DataParallel compatibility."""

    def __init__(self, model: SiameseRankNet):
        super().__init__()
        self.model = model

    def forward(self, x, edge_index, edge_attr, batch):
        """Reconstruct a pseudo-Data and call encoder + head."""
        emb = self.model.encoder(x, edge_index, edge_attr, batch)
        emb = self.model.head_drop(emb) if hasattr(self.model, "head_drop") else emb
        emb = torch.relu(self.model.fc1(emb))
        return self.model.fc2(emb)


class MultiGPUPredictor:
    """Distribute GNN inference across multiple GPUs.

    Usage:
        predictor = MultiGPUPredictor(model, gpu_ids=[0, 1, 2, 3])
        scores = predictor.predict(smiles_list, batch_size=512)
    """

    def __init__(self, model: SiameseRankNet, gpu_ids: Optional[List[int]] = None):
        """
        Args:
            model: Trained SiameseRankNet (on CPU or single GPU).
            gpu_ids: List of GPU device IDs. If None, uses all available.
        """
        if gpu_ids is None:
            gpu_ids = list(range(torch.cuda.device_count()))

        if not gpu_ids or not torch.cuda.is_available():
            self.device = torch.device("cpu")
            self.model = model.to(self.device)
            self.model.eval()
            self._parallel = False
            return

        self.device = torch.device(f"cuda:{gpu_ids[0]}")
        self.gpu_ids = gpu_ids
        self.model = model.to(self.device)
        self.model.eval()
        self._parallel = len(gpu_ids) > 1

    def predict(
        self,
        smiles_list: List[str],
        batch_size: int = 512,
    ) -> Tuple[np.ndarray, List[bool]]:
        """Predict scores for a list of SMILES using all available GPUs.

        Args:
            smiles_list: SMILES strings to score.
            batch_size: Per-GPU batch size.

        Returns:
            scores: Array of predicted scores (NaN for invalid SMILES).
            valid_mask: Boolean list indicating valid SMILES.
        """
        # Featurize
        graphs: List[Data] = []
        valid_indices: List[int] = []
        for i, smi in enumerate(smiles_list):
            g = smiles_to_graph_gine(smi)
            if g is not None:
                graphs.append(g)
                valid_indices.append(i)

        if not graphs:
            return np.full(len(smiles_list), np.nan), [False] * len(smiles_list)

        # For multi-GPU: shard the data manually
        if self._parallel:
            scores = self._predict_sharded(graphs, batch_size)
        else:
            scores = self._predict_single(graphs, batch_size)

        # Map back
        full_scores = np.full(len(smiles_list), np.nan)
        valid_mask = [False] * len(smiles_list)
        for j, idx in enumerate(valid_indices):
            full_scores[idx] = scores[j]
            valid_mask[idx] = True

        return full_scores, valid_mask

    def _predict_single(self, graphs: List[Data], batch_size: int) -> np.ndarray:
        """Run inference on a single device."""
        loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
        all_scores = []
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(self.device)
                out = self.model.forward_one(batch).cpu().numpy().flatten()
                all_scores.append(out)
        return np.concatenate(all_scores)

    def _predict_sharded(self, graphs: List[Data], batch_size: int) -> np.ndarray:
        """Shard data across GPUs and collect results."""
        n_gpus = len(self.gpu_ids)
        shard_size = math.ceil(len(graphs) / n_gpus)
        shards = [graphs[i * shard_size:(i + 1) * shard_size] for i in range(n_gpus)]

        all_scores = [None] * n_gpus

        for gpu_idx, (gpu_id, shard) in enumerate(zip(self.gpu_ids, shards)):
            if not shard:
                all_scores[gpu_idx] = np.array([])
                continue
            device = torch.device(f"cuda:{gpu_id}")
            model_copy = self.model.to(device)
            model_copy.eval()

            loader = DataLoader(shard, batch_size=batch_size, shuffle=False)
            shard_scores = []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(device)
                    out = model_copy.forward_one(batch).cpu().numpy().flatten()
                    shard_scores.append(out)
            all_scores[gpu_idx] = np.concatenate(shard_scores) if shard_scores else np.array([])

        # Move model back to primary device
        self.model.to(self.device)
        return np.concatenate([s for s in all_scores if len(s) > 0])

    def predict_file(
        self,
        input_file: str,
        output_file: str,
        batch_size: int = 512,
        chunk_size: int = 50000,
        smiles_col: int = 0,
    ) -> int:
        """Process a large file with multi-GPU inference.

        Args:
            input_file: Input CSV with SMILES column.
            output_file: Output CSV with scores.
            batch_size: Per-GPU batch size.
            chunk_size: Rows to process at once.
            smiles_col: Column index for SMILES.

        Returns:
            Total rows processed.
        """
        total = 0
        write_header = True

        chunk_smiles: List[str] = []
        chunk_indices: List[int] = []

        with open(input_file, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            for idx, row in enumerate(reader):
                if len(row) > smiles_col and row[smiles_col].strip():
                    chunk_smiles.append(row[smiles_col].strip())
                    chunk_indices.append(idx)

                if len(chunk_smiles) >= chunk_size:
                    self._write_predictions(
                        chunk_smiles, chunk_indices, output_file,
                        batch_size, write_header
                    )
                    total += len(chunk_smiles)
                    write_header = False
                    chunk_smiles, chunk_indices = [], []

        if chunk_smiles:
            self._write_predictions(
                chunk_smiles, chunk_indices, output_file,
                batch_size, write_header
            )
            total += len(chunk_smiles)

        return total

    def _write_predictions(self, smiles_list, indices, output_file,
                           batch_size, write_header):
        scores, valid_mask = self.predict(smiles_list, batch_size=batch_size)

        mode = "w" if write_header else "a"
        with open(output_file, mode, encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["index", "SMILES", "score", "valid"])
            for i, smi in enumerate(smiles_list):
                writer.writerow([
                    indices[i], smi,
                    f"{scores[i]:.6f}" if valid_mask[i] else "",
                    valid_mask[i],
                ])

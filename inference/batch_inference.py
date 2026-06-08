# -*- coding: utf-8 -*-
"""Batch processing pipeline for massive SMILES datasets.

Processes millions of SMILES strings through the trained GNN model
in memory-efficient streaming batches.
"""

import csv
import os
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from siamese_GNN.featurization import smiles_to_graph_gine
from siamese_GNN.model import SiameseRankNet, enable_mc_dropout


def _smiles_stream(file_path: str, smiles_col: int = 0,
                   skip_header: bool = True) -> Iterator[Tuple[int, str]]:
    """Stream SMILES from a CSV file without loading everything into memory."""
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        if skip_header:
            next(reader, None)
        for idx, row in enumerate(reader):
            if len(row) > smiles_col and row[smiles_col].strip():
                yield idx, row[smiles_col].strip()


def _featurize_chunk(smiles_list: List[str]) -> Tuple[List[Data], List[int]]:
    """Convert a chunk of SMILES to PyG graphs, tracking valid indices."""
    graphs = []
    valid_indices = []
    for i, smi in enumerate(smiles_list):
        g = smiles_to_graph_gine(smi)
        if g is not None:
            graphs.append(g)
            valid_indices.append(i)
    return graphs, valid_indices


def batch_predict(
    model: SiameseRankNet,
    smiles_list: List[str],
    device: str = "cpu",
    batch_size: int = 256,
    mc_samples: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[bool]]:
    """Predict scores for a list of SMILES.

    Args:
        model: Trained SiameseRankNet model.
        smiles_list: List of SMILES strings.
        device: Device to run inference on.
        batch_size: Batch size for inference.
        mc_samples: If > 0, use MC dropout and return uncertainties.

    Returns:
        scores: Predicted scores (mean if MC dropout).
        uncertainties: Standard deviations (zeros if mc_samples=0).
        valid_mask: Boolean mask indicating which SMILES were valid.
    """
    graphs, valid_indices = _featurize_chunk(smiles_list)

    if not graphs:
        return (np.array([]), np.array([]),
                [False] * len(smiles_list))

    valid_mask = [False] * len(smiles_list)
    for i in valid_indices:
        valid_mask[i] = True

    model = model.to(device)

    if mc_samples > 0:
        enable_mc_dropout(model)
        samples = np.zeros((mc_samples, len(graphs)), dtype=np.float64)
        with torch.no_grad():
            for t in range(mc_samples):
                loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
                offset = 0
                for batch in loader:
                    batch = batch.to(device)
                    out = model.forward_one(batch).cpu().numpy().flatten()
                    samples[t, offset:offset + len(out)] = out
                    offset += len(out)
        scores = samples.mean(axis=0)
        uncertainties = samples.std(axis=0)
    else:
        model.eval()
        scores_list = []
        with torch.no_grad():
            loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
            for batch in loader:
                batch = batch.to(device)
                out = model.forward_one(batch).cpu().numpy().flatten()
                scores_list.append(out)
        scores = np.concatenate(scores_list) if scores_list else np.array([])
        uncertainties = np.zeros_like(scores)

    # Map back to full array
    full_scores = np.full(len(smiles_list), np.nan)
    full_uncertainties = np.full(len(smiles_list), np.nan)
    for j, idx in enumerate(valid_indices):
        full_scores[idx] = scores[j]
        full_uncertainties[idx] = uncertainties[j]

    return full_scores, full_uncertainties, valid_mask


def batch_predict_from_file(
    model: SiameseRankNet,
    input_file: str,
    output_file: str,
    device: str = "cpu",
    batch_size: int = 256,
    chunk_size: int = 10000,
    mc_samples: int = 0,
    smiles_col: int = 0,
) -> int:
    """Process a large SMILES file in streaming chunks.

    Args:
        model: Trained SiameseRankNet model.
        input_file: Path to input CSV with SMILES.
        output_file: Path to output CSV with predictions.
        device: Device for inference.
        batch_size: Batch size for model forward pass.
        chunk_size: Number of SMILES to process at once.
        mc_samples: MC dropout samples (0 for deterministic).
        smiles_col: Column index of SMILES in input CSV.

    Returns:
        Total number of processed compounds.
    """
    total_processed = 0
    write_header = True

    chunk_smiles: List[str] = []
    chunk_indices: List[int] = []

    for idx, smi in _smiles_stream(input_file, smiles_col=smiles_col):
        chunk_smiles.append(smi)
        chunk_indices.append(idx)

        if len(chunk_smiles) >= chunk_size:
            _write_chunk(model, chunk_smiles, chunk_indices, output_file,
                         device, batch_size, mc_samples, write_header)
            total_processed += len(chunk_smiles)
            write_header = False
            chunk_smiles, chunk_indices = [], []

    # Process remaining
    if chunk_smiles:
        _write_chunk(model, chunk_smiles, chunk_indices, output_file,
                     device, batch_size, mc_samples, write_header)
        total_processed += len(chunk_smiles)

    return total_processed


def _write_chunk(model, smiles_list, indices, output_file, device,
                 batch_size, mc_samples, write_header):
    """Process and write one chunk of predictions."""
    scores, uncertainties, valid_mask = batch_predict(
        model, smiles_list, device=device,
        batch_size=batch_size, mc_samples=mc_samples
    )

    mode = "w" if write_header else "a"
    with open(output_file, mode, encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["index", "SMILES", "score", "uncertainty", "valid"])
        for i, smi in enumerate(smiles_list):
            writer.writerow([
                indices[i], smi,
                f"{scores[i]:.6f}" if valid_mask[i] else "",
                f"{uncertainties[i]:.6f}" if valid_mask[i] else "",
                valid_mask[i],
            ])

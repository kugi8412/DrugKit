# -*- coding: utf-8 -*-
"""Monte Carlo dropout uncertainty for the Siamese RankNet score."""

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from siamese_GNN.model import enable_mc_dropout


def mc_dropout_predict(model, graphs: List, mc_samples: int, device,
                       batch_size: int = 32,
                       seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Run `mc_samples` stochastic passes; return (mean_score, std_score) per graph.

    Dropout stays active while BatchNorm uses running statistics.
    """
    if not graphs:
        return np.array([]), np.array([])
    if seed is not None:
        torch.manual_seed(seed)

    model = model.to(device)
    enable_mc_dropout(model)

    samples = np.zeros((mc_samples, len(graphs)), dtype=np.float64)
    with torch.no_grad():
        for t in range(mc_samples):
            loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
            offset = 0
            for batch in loader:
                batch = batch.to(device)
                scores = model.forward_one(batch).cpu().numpy().flatten()
                samples[t, offset:offset + len(scores)] = scores
                offset += len(scores)
    return samples.mean(axis=0), samples.std(axis=0)

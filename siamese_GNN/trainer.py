# -*- coding: utf-8 -*-
"""Training API for the Siamese RankNet, reusable by the active-learning loop."""

import itertools
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import GroupShuffleSplit
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader

from siamese_GNN.featurization import smiles_to_graph_gine
from siamese_GNN.model import SiameseRankNet, WeightedRankNetLoss

DEFAULT_TRAIN_CFG: Dict[str, Any] = {
    "elite_count": 10,
    "elite_penalty": 5.0,
    "val_target_ratio": 0.15,
    "batch_size": 32,
    "epochs": 40,
    "learning_rate": 0.0004,
    "hidden_dim": 128,
    "dropout": 0.3,
    "seed": 42,
}


def build_labeled_graphs(df: pd.DataFrame, sel_col: str, cluster_col: str,
                         elite_count: int) -> Tuple[List, List]:
    """Build graphs from a labeled DataFrame; mark the lowest-selectivity elites."""
    df = df.dropna(subset=["SMILES", sel_col]).copy()
    df_sorted = df.sort_values(by=sel_col, ascending=True)
    elite, seen = set(), set()
    for smi in df_sorted["SMILES"]:
        if smi not in seen:
            elite.add(smi)
            seen.add(smi)
        if len(elite) >= elite_count:
            break

    graphs, clusters = [], []
    for _, row in df.iterrows():
        smi = row["SMILES"]
        g = smiles_to_graph_gine(smi, selectivity=float(row[sel_col]),
                                 is_elite=(smi in elite))
        if g is not None:
            graphs.append(g)
            clusters.append(row.get(cluster_col, "Unknown"))
    return graphs, clusters


class AllPairsDataset(torch.utils.data.Dataset):
    def __init__(self, graphs, elite_penalty: float):
        self.graphs = graphs
        self.elite_penalty = elite_penalty
        self.pair_indices = list(itertools.combinations(range(len(graphs)), 2))

    def __len__(self):
        return len(self.pair_indices)

    def __getitem__(self, idx):
        i1, i2 = self.pair_indices[idx]
        g1, g2 = self.graphs[i1], self.graphs[i2]
        diff = g1.y.item() - g2.y.item()
        label = 1.0 if diff > 0 else (0.0 if diff < 0 else 0.5)
        elite = (g1.is_elite.item() > 0.5) or (g2.is_elite.item() > 0.5)
        weight = self.elite_penalty if elite else 1.0
        return (g1, g2,
                torch.tensor([label], dtype=torch.float),
                torch.tensor([weight], dtype=torch.float))


def collate_weighted(batch):
    g1, g2, l, w = zip(*batch)
    return Batch.from_data_list(g1), Batch.from_data_list(g2), torch.stack(l), torch.stack(w)


def split_data_enforce_ratio(graphs, cluster_ids, target_ratio, seed, logger,
                             max_attempts=50):
    best_split, best_diff = None, 1.0
    for i in range(max_attempts):
        splitter = GroupShuffleSplit(n_splits=1, test_size=target_ratio,
                                     random_state=seed + i)
        try:
            train_idx, val_idx = next(splitter.split(graphs, y=None, groups=cluster_ids))
            ratio = len(val_idx) / len(graphs)
            diff = abs(ratio - target_ratio)
            if diff < best_diff:
                best_diff, best_split = diff, (train_idx, val_idx)
            if diff < 0.05:
                break
        except Exception:
            continue
    if best_split is None:
        if logger:
            logger.warning("Cluster split failed; random split.")
        idx = list(range(len(graphs)))
        random.Random(seed).shuffle(idx)
        cut = max(1, int(len(graphs) * (1 - target_ratio)))
        return idx[:cut], idx[cut:]
    return best_split


def evaluate_metrics(model, graphs, pair_loader, criterion, device, batch_size):
    model.eval()
    single = DataLoader(graphs, batch_size=batch_size, shuffle=False)
    true, pred = [], []
    with torch.no_grad():
        for batch in single:
            batch = batch.to(device)
            scores = model.forward_one(batch)
            pred.extend(scores.cpu().numpy().flatten())
            true.extend(batch.y.cpu().numpy().flatten())
    rho = 0.0
    if len(set(true)) > 1 and len(true) > 1:
        r, _ = spearmanr(true, pred)
        rho = 0.0 if np.isnan(r) else r

    total_loss, total = 0.0, 0
    with torch.no_grad():
        for g1, g2, labels, weights in pair_loader:
            g1, g2, labels, weights = (g1.to(device), g2.to(device),
                                       labels.to(device), weights.to(device))
            s1, s2 = model(g1, g2)
            loss = criterion(s1, s2, labels, weights)
            total_loss += loss.item() * len(labels)
            total += len(labels)
    return rho, (total_loss / total if total else 0.0)


def train_ranknet(graphs, cluster_ids, node_dim, edge_dim, cfg, device,
                  logger=None, model_save_path: Optional[str] = None):
    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    random.seed(cfg["seed"])

    train_idx, val_idx = split_data_enforce_ratio(
        graphs, cluster_ids, cfg["val_target_ratio"], cfg["seed"], logger)
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx] or [graphs[i] for i in train_idx[:1]]

    train_ds = AllPairsDataset(train_graphs, cfg["elite_penalty"])
    val_ds = AllPairsDataset(val_graphs, cfg["elite_penalty"])
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"],
                              shuffle=True, collate_fn=collate_weighted)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"],
                            shuffle=False, collate_fn=collate_weighted)

    model = SiameseRankNet(node_dim, edge_dim, cfg["hidden_dim"], cfg["dropout"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["learning_rate"])
    criterion = WeightedRankNetLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=max(1, cfg["epochs"]), eta_min=1e-5)

    history = {"train_loss": [], "val_loss": [], "train_rho": [], "val_rho": []}
    best_rho, best_state = -1.0, None

    for epoch in range(cfg["epochs"]):
        model.train()
        running = 0.0
        for g1, g2, labels, weights in train_loader:
            g1, g2, labels, weights = (g1.to(device), g2.to(device),
                                       labels.to(device), weights.to(device))
            optimizer.zero_grad()
            s1, s2 = model(g1, g2)
            loss = criterion(s1, s2, labels, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item()
        avg_train = running / max(1, len(train_loader))

        train_rho, _ = evaluate_metrics(model, train_graphs, train_loader,
                                        criterion, device, cfg["batch_size"])
        val_rho, avg_val = evaluate_metrics(model, val_graphs, val_loader,
                                            criterion, device, cfg["batch_size"])
        scheduler.step()

        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)
        history["train_rho"].append(train_rho)
        history["val_rho"].append(val_rho)
        if logger:
            logger.info(f"Epoch {epoch + 1:02d} | loss T/V {avg_train:.4f}/{avg_val:.4f} "
                        f"| rho T/V {train_rho:.4f}/{val_rho:.4f}")
        if val_rho >= best_rho:
            best_rho = val_rho
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    if model_save_path:
        torch.save(model.state_dict(), model_save_path)
    history["best_val_rho"] = best_rho
    return model, history

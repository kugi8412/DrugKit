#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# siamese_GNN/improved_train.py

import os
import random
import logging
import itertools
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import spearmanr

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch_geometric.nn import GINEConv, global_add_pool

from rdkit import Chem
from sklearn.model_selection import GroupShuffleSplit

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

CONFIG = {
    'input_csv': 'output/docking_energies_etc_wide.csv',
    'model_save_path': 'results/GINE_model.pth',
    'plot_save_path': 'results/training_plots.png',

    'target_cols': ['SIT1_MODEL_00_Energy', '8WM3_Energy', '8I91_Energy'],
    'primary_target': '8I91_Energy',
    'homolog_cols': ['8I92_Energy', '8WBY_Energy'],
    'cluster_col': 'Cluster_ID',

    'elite_count': 10,
    'elite_penalty': 5.0,

    'val_target_ratio': 0.15,
    'batch_size': 32,
    'epochs': 40,
    'learning_rate': 0.0004,
    'hidden_dim': 128,
    'dropout': 0.3,
    'seed': 42
}

torch.manual_seed(CONFIG['seed'])
np.random.seed(CONFIG['seed'])
random.seed(CONFIG['seed'])
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

PERMITTED_ATOMS = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B', 'H']


def one_hot_encoding(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def get_atom_features(atom):
    features = one_hot_encoding(atom.GetSymbol(), PERMITTED_ATOMS)
    features += one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4])
    features += one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    features += one_hot_encoding(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
    features += one_hot_encoding(str(atom.GetHybridization()), ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2'])
    features += [1 if atom.GetIsAromatic() else 0]
    features += [atom.GetMass() * 0.01]
    try:
        chiral = str(atom.GetChiralTag())
        features += one_hot_encoding(chiral, ['CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW'])
    except:
        features += [0, 0, 1]

    return features


def get_bond_features(bond):
    bt = bond.GetBondType()
    features = [
        1 if bt == Chem.rdchem.BondType.SINGLE else 0,
        1 if bt == Chem.rdchem.BondType.DOUBLE else 0,
        1 if bt == Chem.rdchem.BondType.TRIPLE else 0,
        1 if bt == Chem.rdchem.BondType.AROMATIC else 0,
    ]
    features += [1 if bond.GetIsConjugated() else 0]
    features += [1 if bond.IsInRing() else 0]
    stereo = str(bond.GetStereo())
    features += one_hot_encoding(stereo, ['STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS'])
    return features


def smiles_to_graph_gine(smiles, selectivity=None, is_elite=False):
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None
    atom_feats = [get_atom_features(atom) for atom in mol.GetAtoms()]
    x = torch.tensor(atom_feats, dtype=torch.float)

    rows, cols, edge_feats = [], [], []
    for bond in mol.GetBonds():
        start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        b_feat = get_bond_features(bond)
        rows += [start, end];
        cols += [end, start]
        edge_feats += [b_feat, b_feat]

    if not rows:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, len(get_bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))),
                                dtype=torch.float)
    else:
        edge_index = torch.tensor([rows, cols], dtype=torch.long)
        edge_attr = torch.tensor(edge_feats, dtype=torch.float)

    y = torch.tensor([selectivity], dtype=torch.float) if selectivity is not None else None
    elite_flag = torch.tensor([1.0 if is_elite else 0.0], dtype=torch.float)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, is_elite=elite_flag, smiles=smiles)


class GINEEncoder(nn.Module):
    def __init__(self, node_in, edge_in, hidden_dim, dropout=0.3):
        super(GINEEncoder, self).__init__()
        self.node_lin = nn.Linear(node_in, hidden_dim)
        self.edge_lin = nn.Linear(edge_in, hidden_dim)

        def make_gine():
            return GINEConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)))

        self.conv1 = make_gine();
        self.conv2 = make_gine();
        self.conv3 = make_gine()
        self.dropout = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_lin(x)
        if edge_attr.numel() > 0:
            edge_attr = self.edge_lin(edge_attr)
        else:
            edge_attr = torch.zeros((edge_index.size(1), x.size(1)), device=x.device)

        x = self.conv1(x, edge_index, edge_attr);
        x = F.relu(x);
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index, edge_attr);
        x = F.relu(x);
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv3(x, edge_index, edge_attr);
        x = F.relu(x)
        return global_add_pool(x, batch)


class SiameseRankNet(nn.Module):
    def __init__(self, node_in, edge_in, hidden_dim, dropout=0.3):
        super(SiameseRankNet, self).__init__()
        self.encoder = GINEEncoder(node_in, edge_in, hidden_dim, dropout)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 1)
        self.dropout = dropout

    def forward_one(self, data):
        x = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

    def forward(self, data1, data2):
        return self.forward_one(data1), self.forward_one(data2)


class WeightedRankNetLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, s1, s2, target, weights):
        loss = self.bce(s1 - s2, target)
        return (loss * weights).mean()


class AllPairsDataset(torch.utils.data.Dataset):
    def __init__(self, graphs):
        self.graphs = graphs
        self.pair_indices = list(itertools.combinations(range(len(graphs)), 2))

    def __len__(self):
        return len(self.pair_indices)

    def __getitem__(self, idx):
        idx1, idx2 = self.pair_indices[idx]
        g1, g2 = self.graphs[idx1], self.graphs[idx2]
        diff = g1.y.item() - g2.y.item()
        if diff > 0:
            label = 1.0
        elif diff < 0:
            label = 0.0
        else:
            label = 0.5

        is_elite1 = (g1.is_elite.item() > 0.5)
        is_elite2 = (g2.is_elite.item() > 0.5)
        weight = CONFIG['elite_penalty'] if (is_elite1 or is_elite2) else 1.0
        return g1, g2, torch.tensor([label], dtype=torch.float), torch.tensor([weight], dtype=torch.float)


def collate_weighted(batch):
    g1, g2, l, w = zip(*batch)
    return Batch.from_data_list(g1), Batch.from_data_list(g2), torch.stack(l), torch.stack(w)


def evaluate_metrics(model, graphs, pair_loader, criterion):
    model.eval()

    single_loader = DataLoader(graphs, batch_size=CONFIG['batch_size'], shuffle=False)
    true, pred = [], []
    with torch.no_grad():
        for batch in single_loader:
            batch = batch.to(DEVICE)
            scores = model.forward_one(batch)
            pred.extend(scores.cpu().numpy().flatten())
            true.extend(batch.y.cpu().numpy().flatten())
    rho, _ = spearmanr(true, pred)
    if np.isnan(rho): rho = 0.0

    total_loss = 0
    total_pairs = 0
    with torch.no_grad():
        for g1, g2, labels, weights in pair_loader:
            g1, g2, labels, weights = g1.to(DEVICE), g2.to(DEVICE), labels.to(DEVICE), weights.to(DEVICE)
            s1, s2 = model(g1, g2)
            loss = criterion(s1, s2, labels, weights)
            total_loss += loss.item() * len(labels)
            total_pairs += len(labels)

    avg_loss = total_loss / total_pairs if total_pairs > 0 else 0.0

    return rho, avg_loss


def plot_history(history, save_path):
    _, axes = plt.subplots(1, 2, figsize=(14, 6))
    epochs = range(1, len(history['train_loss']) + 1)

    axes[0].plot(epochs, history['train_loss'], label='Train Loss', color='#2c3e50', linewidth=2.5)
    axes[0].plot(epochs, history['val_loss'], label='Validation Loss', color='#e74c3c', linewidth=2.5, linestyle='--')
    axes[0].set_title("RankNet Loss Evolution", fontsize=16, weight='bold')
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs, history['train_rho'], label='Train Rho', color='#3498db', linestyle='--')
    axes[1].plot(epochs, history['val_rho'], label='Validation Rho', color='#2ecc71', linewidth=2.5)
    axes[1].set_title("Ranking Correlation (Spearman)", fontsize=16, weight='bold')
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Rho")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def robust_min(row, cols):
    vals = [row[c] for c in cols if pd.notnull(row[c])]
    return min(vals) if vals else np.nan


def robust_max(row, cols):
    vals = [row[c] for c in cols if pd.notnull(row[c])]
    return max(vals) if vals else np.nan


def split_data_enforce_ratio(graphs, cluster_ids, target_ratio=0.15, max_attempts=50):
    best_split = None
    best_diff = 1.0
    for i in range(max_attempts):
        splitter = GroupShuffleSplit(n_splits=1, test_size=target_ratio, random_state=CONFIG['seed'] + i)
        try:
            train_idx, val_idx = next(splitter.split(graphs, y=None, groups=cluster_ids))
            curr_ratio = len(val_idx) / len(graphs)
            diff = abs(curr_ratio - target_ratio)
            if diff < best_diff:
                best_diff = diff
                best_split = (train_idx, val_idx)
            if diff < 0.05:
                break
        except:
            continue

    if best_split is None:
        logger.warning("We can not compute cluster, random split.")
        indices = list(range(len(graphs)))
        random.shuffle(indices)
        split = int(len(graphs) * (1 - target_ratio))
        return indices[:split], indices[split:]
    t_idx, v_idx = best_split
    logger.info(f"Split: Val Ratio = {len(v_idx) / len(graphs):.2%} (Cel: {target_ratio:.0%})")
    return t_idx, v_idx


def main():
    if not os.path.exists(CONFIG['input_csv']): print("No file ", CONFIG['input_csv']); return
    df = pd.read_csv(CONFIG['input_csv'])

    df['Cluster_ID'] = df['Name'].apply(lambda x: x.split('_of_')[-1] if '_of_' in str(x) else 'Unknown')

    all_energy_cols = CONFIG['target_cols'] + CONFIG['homolog_cols']
    for c in all_energy_cols:
        if c in df.columns: df[c] = pd.to_numeric(df[c], errors='coerce')

    if 'Selectivity' in df.columns:
        df['Selectivity'] = pd.to_numeric(df['Selectivity'], errors='coerce')
    df['target_worst_E'] = df.apply(lambda row: robust_max(row, CONFIG['target_cols']), axis=1)
    df['target_best_E'] = df.apply(lambda row: robust_min(row, CONFIG['target_cols']), axis=1)
    df['homolog_best_E'] = df.apply(lambda row: robust_min(row, CONFIG['homolog_cols']), axis=1)
    df.dropna(subset=['SMILES', 'Selectivity', CONFIG['cluster_col']], inplace=True)

    df['selectivity_score'] = df['Selectivity']

    df_sorted = df.sort_values(by='selectivity_score', ascending=True)
    elite_smiles = []
    seen = set()
    for smi in df_sorted['SMILES']:
        if smi not in seen:
            elite_smiles.append(smi);
            seen.add(smi)
        if len(elite_smiles) >= CONFIG['elite_count']:
            break

    elite_set = set(elite_smiles)

    logger.info(f"Liczba związków: {len(df)}")

    graphs, cluster_ids = [], []
    dummy = smiles_to_graph_gine(df.iloc[0]['SMILES'])
    node_dim, edge_dim = dummy.x.shape[1], dummy.edge_attr.shape[1]

    for _, row in df.iterrows():
        smi = row['SMILES']
        g = smiles_to_graph_gine(smi, selectivity=row['selectivity_score'], is_elite=(smi in elite_set))
        if g:
            graphs.append(g)
            cluster_ids.append(row[CONFIG['cluster_col']])

    train_idx, val_idx = split_data_enforce_ratio(graphs, cluster_ids, target_ratio=CONFIG['val_target_ratio'])
    train_graphs = [graphs[i] for i in train_idx]
    val_graphs = [graphs[i] for i in val_idx]

    logger.info(f"Final Train: {len(train_graphs)}, Final Val: {len(val_graphs)}")

    train_ds = AllPairsDataset(train_graphs)
    val_ds = AllPairsDataset(val_graphs)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, collate_fn=collate_weighted)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, collate_fn=collate_weighted)

    model = SiameseRankNet(node_dim, edge_dim, CONFIG['hidden_dim'], CONFIG['dropout']).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    criterion = WeightedRankNetLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'], eta_min=0.00001)

    best_rho = -1.0
    history = {'train_loss': [], 'val_loss': [], 'train_rho': [], 'val_rho': []}

    logger.info("START ...")
    for epoch in range(CONFIG['epochs']):
        model.train()
        total_loss = 0

        for g1, g2, labels, weights in train_loader:
            g1, g2, labels, weights = g1.to(DEVICE), g2.to(DEVICE), labels.to(DEVICE), weights.to(DEVICE)
            optimizer.zero_grad()
            s1, s2 = model(g1, g2)
            loss = criterion(s1, s2, labels, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        train_rho, _ = evaluate_metrics(model, train_graphs, train_loader, criterion)
        val_rho, avg_val_loss = evaluate_metrics(model, val_graphs, val_loader, criterion)

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_rho'].append(train_rho)
        history['val_rho'].append(val_rho)

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        logger.info(
            f"Epoka {epoch + 1:02d} | Loss (T/V): {avg_train_loss:.4f}/{avg_val_loss:.4f} | Rho (T/V): {train_rho:.4f}/{val_rho:.4f} | LR: {current_lr:.6f}")

        if val_rho > best_rho:
            best_rho = val_rho
            torch.save(model.state_dict(), CONFIG['model_save_path'])

    plot_history(history, CONFIG['plot_save_path'])
    logger.info(f"END. Best Rho: {best_rho:.4f}")

if __name__ == "__main__":
    if not os.path.exists("results"): os.makedirs("results")
    main()

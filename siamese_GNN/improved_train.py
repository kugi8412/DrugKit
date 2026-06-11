#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import logging

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch

from siamese_GNN.featurization import feature_dims
from siamese_GNN.trainer import DEFAULT_TRAIN_CFG, build_labeled_graphs, train_ranknet

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

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def main():
    if not os.path.exists(CONFIG['input_csv']):
        print("No file ", CONFIG['input_csv'])
        return
    df = pd.read_csv(CONFIG['input_csv'])

    df['Cluster_ID'] = df['Name'].apply(lambda x: x.split('_of_')[-1] if '_of_' in str(x) else 'Unknown')

    all_energy_cols = CONFIG['target_cols'] + CONFIG['homolog_cols']
    for c in all_energy_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    if 'Selectivity' in df.columns:
        df['Selectivity'] = pd.to_numeric(df['Selectivity'], errors='coerce')
    df['target_worst_E'] = df.apply(lambda row: robust_max(row, CONFIG['target_cols']), axis=1)
    df['target_best_E'] = df.apply(lambda row: robust_min(row, CONFIG['target_cols']), axis=1)
    df['homolog_best_E'] = df.apply(lambda row: robust_min(row, CONFIG['homolog_cols']), axis=1)
    df.dropna(subset=['SMILES', 'Selectivity', CONFIG['cluster_col']], inplace=True)

    df['selectivity_score'] = df['Selectivity']

    logger.info(f"Liczba związków: {len(df)}")

    graphs, cluster_ids = build_labeled_graphs(
        df, sel_col="selectivity_score", cluster_col=CONFIG['cluster_col'],
        elite_count=CONFIG['elite_count'])
    node_dim, edge_dim = feature_dims()

    train_cfg = dict(DEFAULT_TRAIN_CFG)
    train_cfg.update({
        "elite_count": CONFIG["elite_count"],
        "elite_penalty": CONFIG["elite_penalty"],
        "val_target_ratio": CONFIG["val_target_ratio"],
        "batch_size": CONFIG["batch_size"],
        "epochs": CONFIG["epochs"],
        "learning_rate": CONFIG["learning_rate"],
        "hidden_dim": CONFIG["hidden_dim"],
        "dropout": CONFIG["dropout"],
        "seed": CONFIG["seed"],
    })

    logger.info("START ...")
    _, history = train_ranknet(
        graphs, cluster_ids, node_dim, edge_dim, train_cfg, DEVICE,
        logger=logger, model_save_path=CONFIG['model_save_path'])
    plot_history(history, CONFIG['plot_save_path'])
    logger.info(f"END. Best Rho: {history['best_val_rho']:.4f}")


if __name__ == "__main__":
    if not os.path.exists("results"):
        os.makedirs("results")
    main()

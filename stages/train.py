#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 3: Model Training train the Siamese GNN ranking model.

Usage:
    python -m stages.train --labeled data/labeled.csv --output models/model.pth
    python -m stages.train --labeled data/labeled.csv --epochs 100 --lr 0.0005
    python -m stages.train --labeled data/labeled.csv --hidden-dim 128 --device cuda:0

Parameters:
    --labeled       CSV with SMILES and docking scores (required)
    --output        Output model checkpoint path (default: models/model.pth)
    --smiles-col    SMILES column name (default: "SMILES")
    --score-col     Score column name (default: "score")
    --hidden-dim    GNN hidden dimension (default: 64)
    --num-layers    Number of GIN layers (default: 4)
    --dropout       Dropout rate for MC Dropout (default: 0.1)
    --lr            Learning rate (default: 0.001)
    --epochs        Training epochs (default: 50)
    --batch-size    Training batch size (default: 32)
    --margin        RankNet margin (default: 1.0)
    --patience      Early stopping patience (default: 10)
    --val-frac      Validation fraction (default: 0.15)
    --device        Device: cpu, cuda, cuda:0, etc. (default: auto)
    --seed          Random seed (default: 42)
    --resume        Resume from checkpoint (path)
"""

import argparse
import os
import time
from typing import Optional

import torch


def run_train(
    labeled_file: str,
    output_file: str = "models/model.pth",
    smiles_col: str = "SMILES",
    score_col: str = "score",
    hidden_dim: int = 64,
    dropout: float = 0.1,
    lr: float = 0.001,
    epochs: int = 50,
    batch_size: int = 32,
    val_frac: float = 0.15,
    device: Optional[str] = None,
    seed: int = 42,
) -> str:
    """Train the Siamese GNN ranking model.

    Args:
        labeled_file: CSV with SMILES and docking scores.
        output_file: Where to save the trained model.
        smiles_col: Column name for SMILES.
        score_col: Column name for docking scores.
        hidden_dim: GNN hidden layer dimension.
        dropout: Dropout probability (enables MC Dropout at inference).
        lr: Learning rate.
        epochs: Maximum training epochs.
        batch_size: Batch size for pair generation.
        val_frac: Fraction of data for validation.
        device: Compute device (auto-detected if None).
        seed: Random seed.

    Returns:
        Path to saved model checkpoint.
    """
    import pandas as pd
    import numpy as np
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau

    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Resolve device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dev = torch.device(device)
    print(f"Device: {dev}")

    # Load data
    if not os.path.exists(labeled_file):
        raise FileNotFoundError(f"Labeled file not found: {labeled_file}")

    df = pd.read_csv(labeled_file)
    if smiles_col not in df.columns:
        raise ValueError(f"SMILES column '{smiles_col}' not in {list(df.columns)}")
    if score_col not in df.columns:
        raise ValueError(f"Score column '{score_col}' not in {list(df.columns)}")

    smiles = df[smiles_col].tolist()
    scores = df[score_col].astype(float).tolist()
    print(f"Loaded {len(smiles)} labeled compounds (score range: "
          f"{min(scores):.2f} to {max(scores):.2f})")

    # Featurize using the actual API
    from siamese_GNN.featurization import smiles_to_graph_gine, feature_dims

    graphs = []
    cluster_ids = []
    for smi, sc in zip(smiles, scores):
        g = smiles_to_graph_gine(smi, selectivity=sc, is_elite=False)
        if g is not None:
            graphs.append(g)
            cluster_ids.append("Unknown")

    print(f"Valid graphs: {len(graphs)}/{len(smiles)}")
    if len(graphs) < 10:
        raise ValueError("Too few valid graphs for training (need at least 10).")

    # Get feature dimensions from data
    node_dim, edge_dim = feature_dims()

    train_cfg = {
        "elite_count": 10,
        "elite_penalty": 5.0,
        "val_target_ratio": val_frac,
        "batch_size": batch_size,
        "epochs": epochs,
        "learning_rate": lr,
        "hidden_dim": hidden_dim,
        "dropout": dropout,
        "seed": seed,
    }

    # Train using the actual trainer API
    from siamese_GNN.trainer import train_ranknet

    print(f"\nTraining for up to {epochs} epochs (hidden_dim={hidden_dim})...")
    start = time.time()

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    model, history = train_ranknet(
        graphs=graphs,
        cluster_ids=cluster_ids,
        node_dim=node_dim,
        edge_dim=edge_dim,
        cfg=train_cfg,
        device=dev,
        logger=None,
        model_save_path=output_file,
    )

    elapsed = time.time() - start
    best_rho = history.get("best_val_rho", 0.0)
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Best val Spearman Ď: {best_rho:.4f}")
    print(f"Model saved to: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="DrugKit Stage 3: Siamese GNN Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m stages.train --labeled data/labeled.csv --output models/gnn.pth
  python -m stages.train --labeled data/labeled.csv --epochs 100 --hidden-dim 128
  python -m stages.train --labeled data/labeled.csv --device cuda:0 --lr 0.0005
        """,
    )
    parser.add_argument("--labeled", "-l", required=True,
                        help="CSV with SMILES and docking scores")
    parser.add_argument("--output", "-o", default="models/model.pth",
                        help="Output model path (default: models/model.pth)")
    parser.add_argument("--smiles-col", default="SMILES",
                        help="SMILES column name (default: SMILES)")
    parser.add_argument("--score-col", default="score",
                        help="Score column name (default: score)")
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="GNN hidden dimension (default: 64)")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate (default: 0.1)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Max epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--val-frac", type=float, default=0.15,
                        help="Validation fraction (default: 0.15)")
    parser.add_argument("--device", help="Device: cpu, cuda, cuda:0 (default: auto)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    run_train(
        labeled_file=args.labeled,
        output_file=args.output,
        smiles_col=args.smiles_col,
        score_col=args.score_col,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        val_frac=args.val_frac,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 5: Active Learning iterative model refinement loop.

Usage:
    python -m stages.active_learn --config config/drugkit.yaml
    python -m stages.active_learn --config config/drugkit.yaml --rounds 5 --seed-size 50
    python -m stages.active_learn --labeled data/seed.csv --pool data/pool.csv --rounds 3

Parameters:
    --config            YAML configuration file (provides all settings)
    --labeled           CSV with initial labeled compounds (overrides config)
    --pool              CSV with unlabeled pool (overrides config)
    --output-dir        Directory for results (default: output/active_learning/)
    --rounds            Number of AL rounds (default: 5)
    --seed-size         Initial seed size (default: 50)
    --acquisition-batch Compounds to acquire per round (default: 20)
    --acquisition-fn    Acquisition function: greedy, uncertainty, ucb, thompson, random
    --epochs            Training epochs per round (default: 50)
    --hidden-dim        Model hidden dimension (default: 64)
    --mc-samples        MC Dropout samples for uncertainty (default: 10)
    --device            Device (default: auto)
    --seed              Random seed (default: 42)
    --dock-engine       Docking engine for oracle: smina, vina (default: smina)
    --receptor          Receptor file for on-the-fly docking
    --center            Binding site center x,y,z
"""

import argparse
import os
import time
from pathlib import Path
from typing import Optional


def run_active_learn(
    config_path: Optional[str] = None,
    labeled_file: Optional[str] = None,
    pool_file: Optional[str] = None,
    output_dir: str = "output/active_learning",
    rounds: int = 5,
    seed_size: int = 50,
    acquisition_batch: int = 20,
    acquisition_fn: str = "greedy",
    epochs: int = 50,
    hidden_dim: int = 64,
    mc_samples: int = 10,
    device: Optional[str] = None,
    seed: int = 42,
    dock_engine: str = "smina",
    receptor: Optional[str] = None,
    center: Optional[tuple] = None,
) -> str:
    """Run the active learning loop.

    Args:
        config_path: Path to YAML config (overridden by explicit args).
        labeled_file: CSV with initial labeled data.
        pool_file: CSV with unlabeled pool.
        output_dir: Directory for output files.
        rounds: Number of active learning iterations.
        seed_size: Number of initial seed compounds.
        acquisition_batch: Compounds to acquire per round.
        acquisition_fn: Acquisition strategy name.
        epochs: Training epochs per round.
        hidden_dim: GNN hidden dimension.
        mc_samples: MC Dropout samples.
        device: Compute device.
        seed: Random seed.
        dock_engine: Docking backend for oracle.
        receptor: Receptor file for docking oracle.
        center: Binding site center (x, y, z).

    Returns:
        Path to output directory with results.
    """
    import yaml
    import numpy as np
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Build config dict (matches active_learning.loop.run_active_learning interface)
    from active_learning.config import DEFAULT_CONFIG

    # Start from defaults
    cfg = dict(DEFAULT_CONFIG)

    # Load from YAML if provided
    if config_path:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        # Merge active_learning section
        al_section = raw.get("active_learning", {})
        cfg.update(al_section)

    # CLI args override config
    if labeled_file:
        cfg["seed_file"] = labeled_file
    if pool_file:
        cfg["pool_file"] = pool_file
    cfg["rounds"] = rounds
    cfg["seed_size"] = seed_size
    cfg["acquisition_batch"] = acquisition_batch
    cfg["mc_samples"] = mc_samples
    cfg["epochs"] = epochs
    cfg["hidden_dim"] = hidden_dim
    cfg["seed"] = seed
    cfg["output_dir"] = output_dir

    # Validate files
    if not os.path.exists(cfg["seed_file"]):
        raise FileNotFoundError(f"Seed/labeled file not found: {cfg['seed_file']}")
    if not os.path.exists(cfg["pool_file"]):
        raise FileNotFoundError(f"Pool file not found: {cfg['pool_file']}")
    if receptor and not os.path.exists(receptor):
        raise FileNotFoundError(f"Receptor not found: {receptor}")

    # Ensure off_targets is set (required by run_active_learning)
    if not cfg.get("off_targets"):
        raise ValueError(
            "Selectivity requires at least one off_target. "
            "Set active_learning.off_targets in config.yaml or provide grids with multiple targets."
        )

    os.makedirs(output_dir, exist_ok=True)

    print(f"Active Learning Configuration:")
    print(f"  Rounds: {cfg['rounds']}")
    print(f"  Seed size: {cfg['seed_size']}")
    print(f"  Acquisition batch: {cfg['acquisition_batch']}")
    print(f"  Epochs/round: {cfg['epochs']}")
    print(f"  MC samples: {cfg['mc_samples']}")
    print(f"  Device: {device or 'cpu'}")
    print()

    # Run active learning loop using the real API
    from active_learning.loop import run_active_learning

    start = time.time()
    results = run_active_learning(
        cfg=cfg,
        logger=None,
        dock_fn=None,  # Will build default dock_fn from grids
        device=device or "cpu",
    )
    elapsed = time.time() - start

    print(f"\nActive learning completed in {elapsed:.1f}s")
    print(f"Results saved to: {output_dir}")
    return output_dir


def main():
    parser = argparse.ArgumentParser(
        description="DrugKit Stage 5: Active Learning Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m stages.active_learn --config config/drugkit.yaml
  python -m stages.active_learn --labeled seed.csv --pool pool.csv --rounds 5
  python -m stages.active_learn --config config/drugkit.yaml --acquisition-fn ucb
  python -m stages.active_learn --labeled seed.csv --pool pool.csv \\
      --receptor receptor.pdbqt --center 12.5,3.2,7.8 --dock-engine smina
        """,
    )
    parser.add_argument("--config", "-c", help="YAML configuration file")
    parser.add_argument("--labeled", help="Initial labeled compounds CSV")
    parser.add_argument("--pool", help="Unlabeled pool CSV")
    parser.add_argument("--output-dir", default="output/active_learning",
                        help="Output directory (default: output/active_learning/)")
    parser.add_argument("--rounds", type=int, default=5,
                        help="AL rounds (default: 5)")
    parser.add_argument("--seed-size", type=int, default=50,
                        help="Initial seed compounds (default: 50)")
    parser.add_argument("--acquisition-batch", type=int, default=20,
                        help="Compounds per round (default: 20)")
    parser.add_argument("--acquisition-fn",
                        choices=["greedy", "uncertainty", "ucb", "thompson", "random"],
                        default="greedy",
                        help="Acquisition function (default: greedy)")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs per round (default: 50)")
    parser.add_argument("--hidden-dim", type=int, default=64,
                        help="GNN hidden dimension (default: 64)")
    parser.add_argument("--mc-samples", type=int, default=10,
                        help="MC Dropout samples (default: 10)")
    parser.add_argument("--device", help="Device (default: auto)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--dock-engine", choices=["smina", "vina"], default="smina",
                        help="Docking oracle engine (default: smina)")
    parser.add_argument("--receptor", help="Receptor .pdbqt for docking oracle")
    parser.add_argument("--center", help="Binding site center x,y,z")
    args = parser.parse_args()

    center = None
    if args.center:
        parts = [float(x) for x in args.center.split(",")]
        if len(parts) != 3:
            parser.error("--center must be x,y,z (3 values)")
        center = tuple(parts)

    run_active_learn(
        config_path=args.config,
        labeled_file=args.labeled,
        pool_file=args.pool,
        output_dir=args.output_dir,
        rounds=args.rounds,
        seed_size=args.seed_size,
        acquisition_batch=args.acquisition_batch,
        acquisition_fn=args.acquisition_fn,
        epochs=args.epochs,
        hidden_dim=args.hidden_dim,
        mc_samples=args.mc_samples,
        device=args.device,
        seed=args.seed,
        dock_engine=args.dock_engine,
        receptor=args.receptor,
        center=center,
    )


if __name__ == "__main__":
    main()

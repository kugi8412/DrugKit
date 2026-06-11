#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DrugKit unified pipeline runner.

Provides a single entry point to run the complete drug screening pipeline
with full user configurability at every stage. Users can:
  - Inject their own data at any stage
  - Override any parameter
  - Skip stages (provide pre-computed outputs)
  - Use custom docking/scoring functions
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
import torch
import yaml

from siamese_GNN.featurization import feature_dims, smiles_to_graph_gine
from siamese_GNN.model import SiameseRankNet
from siamese_GNN.trainer import build_labeled_graphs, train_ranknet
from active_learning.acquisition import select_top_uncertain
from active_learning.selectivity import compute_selectivity
from active_learning.uncertainty import mc_dropout_predict


@dataclass
class PipelineConfig:
    """Complete configuration for the DrugKit pipeline.

    Every parameter has a sensible default but can be overridden by the user.
    """

    pool_file: str = "data/pool.csv"
    seed_file: str = "data/seed_ligands.csv"
    grids_file: str = "docking_grids.json"
    output_dir: str = "output/pipeline"
    data_dir: str = "data"

    # Target Definition
    on_targets: List[str] = field(default_factory=lambda: ["TARGET"])
    off_targets: List[str] = field(default_factory=list)

    # Active Learning
    rounds: int = 5
    seed_size: int = 20
    acquisition_batch: int = 10
    mc_samples: int = 30

    # Model Training
    epochs: int = 40
    batch_size: int = 32
    learning_rate: float = 0.0004
    hidden_dim: int = 128
    dropout: float = 0.3
    elite_count: int = 10
    elite_penalty: float = 5.0
    val_target_ratio: float = 0.15
    seed: int = 42

    # Docking Engine
    docking_engine: str = "smina"
    smina_exe: str = "smina"
    exhaustiveness: int = 8
    num_modes: int = 1
    n_cpu: int = 4
    default_baseline: float = -7.0

    # Inference
    inference_batch_size: int = 512
    inference_chunk_size: int = 50000
    inference_mc_samples: int = 0
    gpu_ids: Optional[List[int]] = None

    # Filtering
    min_molecular_weight: float = 150.0
    max_molecular_weight: float = 500.0
    min_logp: float = -1.0
    max_logp: float = 5.0
    lipinski_strict: bool = True
    veber_tpsa_cutoff: float = 140.0
    veber_rotatable_cutoff: int = 10

    # Device
    device: str = "auto"  # "auto", "cpu", "cuda", "cuda:0", etc.

    def resolve_device(self) -> str:
        if self.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load config from YAML file, merging with defaults."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"Config file not found: {path}")
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        # Flatten nested sections into single dict
        flat = {}
        for section in ("project", "active_learning", "docking_smina",
                        "ligand_expansion", "inference"):
            if section in raw:
                flat.update(raw[section])

        # Map known keys
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in flat.items() if k in field_names}
        return cls(**kwargs)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PipelineConfig":
        """Create config from a plain dictionary."""
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        kwargs = {k: v for k, v in d.items() if k in field_names}
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        """Export as plain dict for serialization."""
        import dataclasses
        return dataclasses.asdict(self)


class PipelineError(Exception):
    """Raised when pipeline encounters an unrecoverable error."""
    pass


def validate_csv(path: str, required_columns: List[str], name: str) -> pd.DataFrame:
    """Load and validate a CSV file.

    Args:
        path: Path to CSV file.
        required_columns: Columns that must be present.
        name: Human-readable name for error messages.

    Returns:
        Validated DataFrame.

    Raises:
        PipelineError: If file missing or columns invalid.
    """
    if not os.path.exists(path):
        raise PipelineError(
            f"{name} not found: {path}\n"
            f"Expected CSV with columns: {required_columns}"
        )

    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise PipelineError(
            f"{name} ({path}) missing required columns: {missing}\n"
            f"Found columns: {list(df.columns)}\n"
            f"Required: {required_columns}"
        )

    if len(df) == 0:
        raise PipelineError(f"{name} ({path}) is empty.")

    return df


def validate_grids(path: str) -> Dict[str, Any]:
    """Load and validate docking grids JSON.

    Returns:
        Grids dictionary.

    Raises:
        PipelineError: If file missing or format invalid.
    """
    if not os.path.exists(path):
        raise PipelineError(
            f"Grids file not found: {path}\n"
            f"Expected JSON format:\n"
            f'{{"TARGET_NAME": [{{"id": "pocket_1", "center": [x,y,z], "size": [sx,sy,sz]}}]}}'
        )

    with open(path, "r", encoding="utf-8") as f:
        grids = json.load(f)

    if not isinstance(grids, dict) or len(grids) == 0:
        raise PipelineError(f"Grids file ({path}) must be a non-empty JSON object.")

    for target_name, pockets in grids.items():
        if not isinstance(pockets, list):
            raise PipelineError(
                f"Grids['{target_name}'] must be a list of pocket definitions."
            )
        for i, pocket in enumerate(pockets):
            for key in ("id", "center", "size"):
                if key not in pocket:
                    raise PipelineError(
                        f"Grids['{target_name}'][{i}] missing required key: '{key}'"
                    )
            if len(pocket["center"]) != 3 or len(pocket["size"]) != 3:
                raise PipelineError(
                    f"Grids['{target_name}'][{i}]: 'center' and 'size' must be [x, y, z]."
                )
    return grids


def validate_smiles(smiles: str) -> bool:
    """Check if a SMILES string can be featurized."""
    return smiles_to_graph_gine(smiles) is not None


class DrugKitPipeline:
    """Unified pipeline for finding selective drug candidates.

    Usage:
        pipeline = DrugKitPipeline(config)
        pipeline.run()

    Or run individual stages:
        pipeline.validate_inputs()
        pipeline.run_bootstrap()
        pipeline.run_active_learning()
        pipeline.run_inference("library.csv", "output.csv")
    """

    def __init__(self, config: PipelineConfig,
                 dock_fn: Optional[Callable] = None,
                 selectivity_fn: Optional[Callable] = None,
                 logger: Optional[logging.Logger] = None):
        """
        Args:
            config: Pipeline configuration.
            dock_fn: Custom docking function. Signature:
                     (records: List[Dict]) -> pd.DataFrame
                     Input dicts have keys: "Name", "SMILES"
                     Output DataFrame must have: Name, SMILES, Target, Pocket_ID, Energy
            selectivity_fn: Custom selectivity scoring. Signature:
                            (on_target_best: float, offtarget_best: float) -> float
            logger: Optional logger. If None, creates one.
        """
        self.cfg = config
        self.dock_fn = dock_fn
        self.selectivity_fn = selectivity_fn
        self.logger = logger or self._make_logger()

        self.device = config.resolve_device()
        self.model: Optional[SiameseRankNet] = None
        self.labeled: Optional[pd.DataFrame] = None
        self.history: List[Dict[str, Any]] = []

    def _make_logger(self) -> logging.Logger:
        logger = logging.getLogger("drugkit.pipeline")
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            handler.setFormatter(logging.Formatter(
                "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%H:%M:%S"
            ))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def log(self, msg: str):
        self.logger.info(msg)

    # Validation

    def validate_inputs(self):
        """Validate all input files and configuration before running."""
        self.log("Validating inputs...")

        # Check pool file
        pool_df = validate_csv(self.cfg.pool_file, ["Name", "SMILES"], "Pool file")
        self.log(f"  Pool: {len(pool_df)} compounds")

        # Check seed file
        seed_df = validate_csv(self.cfg.seed_file, ["Name", "SMILES"], "Seed file")
        self.log(f"  Seeds: {len(seed_df)} compounds")

        # Validate SMILES quality (sample)
        sample_size = min(20, len(pool_df))
        valid_count = sum(
            validate_smiles(s) for s in pool_df["SMILES"].head(sample_size)
        )
        if valid_count < sample_size * 0.5:
            raise PipelineError(
                f"Too many invalid SMILES in pool (only {valid_count}/{sample_size} valid). "
                "Check your SMILES format."
            )

        # Targets
        if not self.cfg.on_targets:
            raise PipelineError("on_targets cannot be empty.")

        # Grids (only if using built-in docking)
        if self.dock_fn is None:
            grids = validate_grids(self.cfg.grids_file)
            self.log(f"  Grids: {len(grids)} targets, "
                     f"{sum(len(p) for p in grids.values())} pockets")

            # Validate targets match grids
            all_grid_targets = set(grids.keys())
            for t in self.cfg.on_targets + self.cfg.off_targets:
                if t not in all_grid_targets:
                    raise PipelineError(
                        f"Target '{t}' not found in grids file. "
                        f"Available: {sorted(all_grid_targets)}"
                    )
        else:
            self.log("  Using custom dock_fn (skipping grids/receptor validation)")

        # Check output directory
        os.makedirs(self.cfg.output_dir, exist_ok=True)

        self.log("Validation passed.")

    # Bootstrap

    def run_bootstrap(self, pre_labeled: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Run bootstrap docking of seed compounds.

        Args:
            pre_labeled: If provided, skip docking and use this as initial
                         labeled set. Must have columns: Name, SMILES, selectivity, Cluster_ID.

        Returns:
            Initial labeled DataFrame.
        """
        if pre_labeled is not None:
            self.log(f"Using pre-labeled data ({len(pre_labeled)} compounds)")
            required = ["Name", "SMILES", "selectivity"]
            missing = [c for c in required if c not in pre_labeled.columns]
            if missing:
                raise PipelineError(
                    f"pre_labeled DataFrame missing columns: {missing}"
                )
            if "Cluster_ID" not in pre_labeled.columns:
                pre_labeled = pre_labeled.copy()
                pre_labeled["Cluster_ID"] = "Unknown"
            self.labeled = pre_labeled
            return self.labeled

        self.log("Bootstrap: docking seed compounds...")
        seed_df = validate_csv(self.cfg.seed_file, ["Name", "SMILES"], "Seed file")
        seed_df = seed_df.head(self.cfg.seed_size)

        records = seed_df[["Name", "SMILES"]].to_dict("records")
        dock_fn = self._get_dock_fn()
        results = dock_fn(records)

        if results.empty:
            raise PipelineError(
                "Bootstrap docking returned no results. "
                "Check receptor files and docking configuration."
            )

        self.labeled = compute_selectivity(
            results,
            on_targets=self.cfg.on_targets,
            off_targets=self.cfg.off_targets,
            selectivity_fn=self.selectivity_fn,
        )

        if self.labeled.empty or len(self.labeled) < 2:
            raise PipelineError(
                f"Bootstrap produced only {len(self.labeled)} labeled compounds "
                "(need >= 2). Check that on_targets/off_targets match your grids."
            )

        if "Cluster_ID" not in self.labeled.columns:
            self.labeled["Cluster_ID"] = "Unknown"

        self.log(f"Bootstrap complete: {len(self.labeled)} labeled compounds")
        return self.labeled

    # Active Learning

    def run_active_learning(self,
                            pre_labeled: Optional[pd.DataFrame] = None,
                            ) -> Dict[str, Any]:
        """Run the full active learning loop.

        Args:
            pre_labeled: Optional pre-computed labeled set (skips bootstrap).

        Returns:
            Dict with keys: "labeled", "model", "history"
        """
        self.validate_inputs()

        # Bootstrap
        if self.labeled is None:
            self.run_bootstrap(pre_labeled)

        # Load pool
        pool_df = validate_csv(self.cfg.pool_file, ["Name", "SMILES"], "Pool file")
        attempted = set(self.labeled["Name"].tolist())
        dock_fn = self._get_dock_fn()

        node_dim, edge_dim = feature_dims()
        self.history = []

        for rnd in range(self.cfg.rounds):
            self.log(f"=== Round {rnd + 1}/{self.cfg.rounds} "
                     f"(labeled={len(self.labeled)}) ===")

            # Train model
            graphs, clusters = build_labeled_graphs(
                self.labeled, sel_col="selectivity",
                cluster_col="Cluster_ID", elite_count=self.cfg.elite_count
            )

            if len(graphs) < 2:
                self.log("Not enough valid graphs for training. Stopping.")
                break

            train_cfg = {
                "elite_penalty": self.cfg.elite_penalty,
                "val_target_ratio": self.cfg.val_target_ratio,
                "batch_size": self.cfg.batch_size,
                "epochs": self.cfg.epochs,
                "learning_rate": self.cfg.learning_rate,
                "hidden_dim": self.cfg.hidden_dim,
                "dropout": self.cfg.dropout,
                "seed": self.cfg.seed,
            }

            self.model, train_hist = train_ranknet(
                graphs, clusters, node_dim, edge_dim,
                train_cfg, self.device, logger=self.logger
            )

            # Score candidates
            candidates = pool_df[~pool_df["Name"].isin(attempted)]
            if candidates.empty:
                self.log("Pool exhausted.")
                break

            cand_graphs, cand_names = [], []
            for _, row in candidates.iterrows():
                g = smiles_to_graph_gine(row["SMILES"])
                if g is not None:
                    cand_graphs.append(g)
                    cand_names.append(row["Name"])

            if not cand_graphs:
                self.log("No valid candidate graphs. Stopping.")
                break

            # MC Dropout uncertainty
            means, stds = mc_dropout_predict(
                self.model, cand_graphs, self.cfg.mc_samples,
                self.device, batch_size=self.cfg.batch_size, seed=self.cfg.seed
            )

            # Acquire most uncertain
            picked = select_top_uncertain(
                cand_names, stds, self.cfg.acquisition_batch, attempted
            )
            self.log(f"Selected {len(picked)} compounds (highest uncertainty)")

            # Dock selected compounds
            name_to_smi = dict(zip(candidates["Name"], candidates["SMILES"]))
            records = [{"Name": n, "SMILES": name_to_smi[n]} for n in picked]
            new_results = dock_fn(records)
            attempted.update(picked)

            if not new_results.empty:
                new_labeled = compute_selectivity(
                    new_results,
                    on_targets=self.cfg.on_targets,
                    off_targets=self.cfg.off_targets,
                    selectivity_fn=self.selectivity_fn,
                )
                if not new_labeled.empty:
                    if "Cluster_ID" not in new_labeled.columns:
                        new_labeled["Cluster_ID"] = "Unknown"
                    self.labeled = pd.concat(
                        [self.labeled, new_labeled], ignore_index=True
                    ).drop_duplicates(subset=["SMILES"], keep="first")

            # Log metrics
            unc_map = dict(zip(cand_names, stds))
            sel_unc = [unc_map.get(n, 0.0) for n in picked]
            self.history.append({
                "round": rnd + 1,
                "labeled_size": len(self.labeled),
                "selected": len(picked),
                "val_rho": float(train_hist.get("best_val_rho", 0.0)),
                "mean_uncertainty": float(sum(sel_unc) / max(len(sel_unc), 1)),
                "max_uncertainty": float(max(sel_unc)) if sel_unc else 0.0,
            })

        # Save outputs
        self._save_outputs()

        return {
            "labeled": self.labeled,
            "model": self.model,
            "history": self.history,
        }


    def run_inference(self, input_file: str, output_file: str,
                      model: Optional[SiameseRankNet] = None,
                      mc_samples: Optional[int] = None) -> int:
        """Score a large SMILES library with the trained model.

        Args:
            input_file: CSV with SMILES column.
            output_file: Output CSV path.
            model: Model to use. If None, uses self.model.
            mc_samples: Override MC samples. None uses config value.

        Returns:
            Number of compounds processed.
        """
        from inference.batch_inference import batch_predict_from_file

        model = model or self.model
        if model is None:
            raise PipelineError("No trained model available. Run active learning first.")

        mc = mc_samples if mc_samples is not None else self.cfg.inference_mc_samples

        self.log(f"Scoring {input_file} â†’ {output_file} (mc_samples={mc})")
        total = batch_predict_from_file(
            model,
            input_file=input_file,
            output_file=output_file,
            device=self.device,
            batch_size=self.cfg.inference_batch_size,
            chunk_size=self.cfg.inference_chunk_size,
            mc_samples=mc,
        )
        self.log(f"Processed {total} compounds.")
        return total


    def _get_dock_fn(self) -> Callable:
        """Get the docking function (custom or built-in)."""
        if self.dock_fn is not None:
            return self.dock_fn

        # Build from config
        from docking_common.io_utils import load_grids
        from docking_common.receptor_io import build_receptor_map
        from active_learning.docking_adapter import dock_compounds

        grids = validate_grids(self.cfg.grids_file)
        rec_map = build_receptor_map(grids, self.cfg.data_dir, self.logger)
        if not rec_map:
            raise PipelineError(
                f"No receptor files found in '{self.cfg.data_dir}/'. "
                f"Expected: {[f'{t}.pdbqt' for t in grids.keys()]}"
            )

        poses_dir = os.path.join(self.cfg.output_dir, "poses")
        os.makedirs(poses_dir, exist_ok=True)

        cfg_dict = {
            "smina_exe": self.cfg.smina_exe,
            "exhaustiveness": self.cfg.exhaustiveness,
            "num_modes": self.cfg.num_modes,
            "n_cpu": self.cfg.n_cpu,
            "default_baseline": self.cfg.default_baseline,
        }

        def dock_fn(records):
            return dock_compounds(
                records, rec_map, grids, cfg_dict, poses_dir, self.logger
            )

        return dock_fn

    def _save_outputs(self):
        """Save labeled data, history, and model."""
        os.makedirs(self.cfg.output_dir, exist_ok=True)

        if self.labeled is not None:
            path = os.path.join(self.cfg.output_dir, "labeled.csv")
            self.labeled.to_csv(path, index=False)
            self.log(f"Saved labeled data: {path}")

        if self.history:
            path = os.path.join(self.cfg.output_dir, "al_history.csv")
            pd.DataFrame(self.history).to_csv(path, index=False)
            self.log(f"Saved history: {path}")

        if self.model is not None:
            path = os.path.join(self.cfg.output_dir, "model.pth")
            torch.save(self.model.state_dict(), path)
            self.log(f"Saved model: {path}")


def main():
    """Run pipeline from command line with config.yaml."""
    import argparse

    parser = argparse.ArgumentParser(description="DrugKit Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Config YAML file")
    parser.add_argument("--stage", choices=["validate", "bootstrap", "train", "infer", "full"],
                        default="full", help="Pipeline stage to run")
    parser.add_argument("--input", help="Input file (for inference stage)")
    parser.add_argument("--output", help="Output file (for inference stage)")
    parser.add_argument("--device", help="Override device (cpu/cuda)")
    parser.add_argument("--rounds", type=int, help="Override AL rounds")
    parser.add_argument("--epochs", type=int, help="Override training epochs")
    parser.add_argument("--mc-samples", type=int, help="Override MC samples")
    parser.add_argument("--hidden-dim", type=int, help="Override hidden dimension")
    parser.add_argument("--batch-size", type=int, help="Override batch size")
    parser.add_argument("--exhaustiveness", type=int, help="Override docking exhaustiveness")
    parser.add_argument("--n-cpu", type=int, help="Override CPU count")

    args = parser.parse_args()

    # Load config
    cfg = PipelineConfig.from_yaml(args.config)

    # Apply CLI overrides
    if args.device:
        cfg.device = args.device
    if args.rounds:
        cfg.rounds = args.rounds
    if args.epochs:
        cfg.epochs = args.epochs
    if args.mc_samples is not None:
        cfg.mc_samples = args.mc_samples
    if args.hidden_dim:
        cfg.hidden_dim = args.hidden_dim
    if args.batch_size:
        cfg.batch_size = args.batch_size
    if args.exhaustiveness:
        cfg.exhaustiveness = args.exhaustiveness
    if args.n_cpu:
        cfg.n_cpu = args.n_cpu

    pipeline = DrugKitPipeline(cfg)

    if args.stage == "validate":
        pipeline.validate_inputs()
    elif args.stage == "bootstrap":
        pipeline.validate_inputs()
        pipeline.run_bootstrap()
    elif args.stage == "train":
        pipeline.validate_inputs()
        pipeline.run_active_learning()
    elif args.stage == "infer":
        if not args.input or not args.output:
            parser.error("--input and --output required for inference stage")
        pipeline.run_inference(args.input, args.output)
    else:
        result = pipeline.run_active_learning()
        if args.input:
            pipeline.run_inference(args.input, args.output or "output/predictions.csv")


if __name__ == "__main__":
    main()

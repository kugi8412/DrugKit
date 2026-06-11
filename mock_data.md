# DrugKit — End-to-End Experiment with Mock Data (Package Usage)

This guide runs the **complete DrugKit pipeline** as an installed Python package.
All examples use either CLI entry points (`drugkit-*`) or the public Python API (`from stages import ...`).

## Installation

```bash
pip install -r requirments

pip install -i https://test.pypi.org/simple/ drugkit==0.1.2
# or editable dev install:
cd DrugKit && pip install -e ".[dev]"
```

Verify:
```bash
drugkit-featurize --help
drugkit-dock --help
drugkit-train --help
drugkit-predict --help
drugkit-active-learn --help
drugkit-select --help
drugkit-expand --help
```

---

## Pipeline Overview

```
drugkit-featurize → drugkit-dock → drugkit-train → drugkit-predict
                                        ↑                ↓
                                   drugkit-active-learn (iterates train+predict+dock)
                                        ↓
                              drugkit-select → drugkit-expand
```

---

## Step 0: Generate Mock Data

```python
"""generate_mock_data.py — creates all mock inputs for the full experiment."""
import os
import json
import csv
import pandas as pd
import numpy as np

os.makedirs("data", exist_ok=True)

# === Mock Pool (100 drug-like SMILES) ===
smiles_pool = [
    "CCO", "CCN", "CCC", "CCCC", "CC=O", "CC(=O)O", "c1ccccc1",
    "Oc1ccccc1", "Nc1ccccc1", "CCBr", "CCCl", "CCS", "CCCO",
    "CC(C)O", "CCOCC", "CC(=O)N", "c1ccncc1", "c1ccoc1",
    "CC(C)CC", "CCCCO", "CCCCN", "CCC=O", "CC(C)=O", "CCOC",
    "c1ccc(O)cc1", "c1ccc(N)cc1", "CC(O)C", "CCC(=O)O",
    "CCCCC", "CCCCCC", "c1ccccc1O", "c1ccccc1N",
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CC(=O)Nc1ccc(O)cc1",
    "OC(=O)c1ccccc1", "c1ccc2c(c1)cccc2", "Clc1ccccc1",
    "FC(F)(F)c1ccccc1", "O=Cc1ccccc1", "c1cnc2ccccc2n1",
    "c1ccc(-c2ccccc2)cc1", "CC(=O)OC", "CCOC(=O)C",
    "OCCN", "OCC(O)CO", "c1ccc(Cl)cc1", "CC#N",
    "O=C(O)CC(=O)O", "NCCN", "c1ccc(F)cc1",
]
# Extend to 100
np.random.seed(42)
while len(smiles_pool) < 100:
    base = np.random.choice(smiles_pool[:20])
    smiles_pool.append(base + "C")

pool_df = pd.DataFrame({
    "Name": [f"MOL_{i:04d}" for i in range(len(smiles_pool))],
    "SMILES": smiles_pool,
})
pool_df.to_csv("data/pool.csv", index=False)
print(f"Pool: {len(pool_df)} compounds → data/pool.csv")

# === Mock Seed Ligands (known binders) ===
seeds_df = pd.DataFrame({
    "Name": ["Ibuprofen", "Acetaminophen", "Naphthalene", "BenzoicAcid"],
    "SMILES": [
        "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
        "CC(=O)Nc1ccc(O)cc1",
        "c1ccc2c(c1)cccc2",
        "OC(=O)c1ccccc1",
    ],
})
seeds_df.to_csv("data/seed_ligands.csv", index=False)
print(f"Seeds: {len(seeds_df)} compounds → data/seed_ligands.csv")

# === Mock Docking Grids JSON ===
grids = {
    "TARGET_A": [
        {"id": "pocket_1", "center": [12.5, 34.2, 8.7], "size": [20, 20, 20]},
    ],
    "OFFTARGET_B": [
        {"id": "pocket_1", "center": [5.1, 22.8, 15.3], "size": [18, 18, 18]},
    ],
}
with open("data/docking_grids.json", "w") as f:
    json.dump(grids, f, indent=2)
print("Grids → data/docking_grids.json")

# === Mock Labeled Data (simulates docked results for training) ===
np.random.seed(42)
n_labeled = 40
labeled_smiles = np.random.choice(pool_df["SMILES"].values, n_labeled, replace=False)
labeled_df = pd.DataFrame({
    "Name": [f"DOCK_{i:03d}" for i in range(n_labeled)],
    "SMILES": labeled_smiles,
    "score": -5.0 - np.random.exponential(1.5, n_labeled).round(2),
})
labeled_df.to_csv("data/labeled.csv", index=False)
print(f"Labeled: {n_labeled} compounds → data/labeled.csv")

# === Mock Large Library (for inference stage) ===
with open("data/large_library.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["SMILES"])
    for i in range(10000):
        writer.writerow([smiles_pool[i % len(smiles_pool)]])
print("Large library: 10,000 compounds → data/large_library.csv")

# === Mock Multi-Target Docking Results (for selectivity) ===
selectivity_df = pd.DataFrame({
    "Name": pool_df["Name"].values[:50],
    "SMILES": pool_df["SMILES"].values[:50],
    "score_CDK2": -6.0 - np.random.exponential(1.0, 50).round(2),
    "score_CDK4": -4.0 - np.random.exponential(1.0, 50).round(2),
    "score_CDK6": -4.5 - np.random.exponential(0.8, 50).round(2),
})
selectivity_df.to_csv("data/multi_target_scores.csv", index=False)
print("Multi-target scores → data/multi_target_scores.csv")

print("\n✓ All mock data generated. Ready to run pipeline.")
```

Run it:
```bash
python generate_mock_data.py
```

---

## Stage 1: Featurize SMILES -> PyG Graphs

### CLI

```bash
# Featurize the pool
drugkit-featurize --input data/pool.csv --output output/pool_graphs.pt --encoder rdkit

# Validate SMILES only (no graph generation)
drugkit-featurize --input data/pool.csv --validate

# Use custom RDKit-free encoder
# drugkit-featurize --input data/pool.csv --output output/graphs_custom.pt --encoder custom

# Featurize inline SMILES
# drugkit-featurize --smiles "CCO" "c1ccccc1" "CC(=O)O" --output output/test.pt
```

### Python API

```python
from stages import run_featurize

# Featurize from CSV
graphs, names, failed = run_featurize(
    input_file="data/pool.csv",
    output_file="output/pool_graphs.pt",
    encoder="rdkit",       # or "custom" for RDKit-free
    batch_size=500,
)
print(f"Generated {len(graphs)} graphs, {len(failed)} failed")

# Featurize inline
graphs, names, failed = run_featurize(
    smiles_list=["CCO", "c1ccccc1", "INVALID"],
    output_file="output/inline_test.pt",
    encoder="rdkit",
)
# graphs[0].x → node features, graphs[0].edge_index → bonds
```

### Parameters

| Parameter | CLI flag | Default | Description |
|-----------|----------|---------|-------------|
| `input_file` | `--input` | — | CSV with SMILES column |
| `output_file` | `--output` | `output/graphs.pt` | Output .pt file |
| `smiles_list` | `--smiles` | — | Inline SMILES (alternative) |
| `smiles_col` | `--smiles-col` | `"SMILES"` | Column name or index |
| `encoder` | `--encoder` | `"rdkit"` | `rdkit` or `custom` |
| `batch_size` | `--batch-size` | `1000` | Processing chunk size |
| `validate_only` | `--validate` | `False` | Only check valid SMILES |

---

## Stage 2: Dock — Molecular Docking

### CLI

```bash
# Convert to pdbqt
cd data
obabel HIVPRO_1HSG.pdb -O HIVPRO_1HSG.pdbqt -p 7.4 --partialcharge gasteiger
cd ..

# Smina
wget https://sourceforge.net/projects/smina/files/smina.static/download -O smina
chmod +x smina

# Use grid definitions from JSON
drugkit-dock --grids data/docking_grids.json --ligands data/pool.csv --smina-exe /home/jgiezgala/DrugKit/smina
```

### Python API

```python
from stages import run_dock

result_path = run_dock(
    receptor="data/TARGET_A.pdbqt",
    ligands="data/pool.csv",
    output_file="output/docking_results.csv",
    engine="smina",          # or "vina"
    center=(12.5, 34.2, 8.7),
    box_size=(20.0, 20.0, 20.0),
    exhaustiveness=8,
    n_cpu=4,
    n_poses=5,
    seed=42,
)
```

### Parameters

| Parameter | CLI flag | Default | Description |
|-----------|----------|---------|-------------|
| `receptor` | `--receptor` | — | Receptor .pdbqt file |
| `ligands` | `--ligands` | — | CSV with SMILES or directory of .pdbqt |
| `engine` | `--engine` | `"smina"` | `smina` or `vina` |
| `center` | `--center` | — | Binding site center `x,y,z` |
| `box_size` | `--box-size` | `20,20,20` | Search box dimensions |
| `exhaustiveness` | `--exhaustiveness` | `8` | Docking thoroughness |
| `n_cpu` | `--n-cpu` | `4` | Parallel CPU cores |
| `n_poses` | `--n-poses` | `5` | Poses per ligand |
| `seed` | `--seed` | — | Random seed |
| `grids_json` | `--grids` | — | JSON grid definitions |

---

## Stage 3: Train — Siamese GNN Model

### CLI

```bash
# Fast train
drugkit-train --labeled output/docking_results.csv --output models/model.pth --epochs 5 --score-col Energy

# Train with defaults
drugkit-train --labeled output/docking_results.csv --output models/model.pth

# Custom hyperparameters
drugkit-train --labeled data/labeled.csv --output models/model.pth \
    --epochs 100 --hidden-dim 128 --lr 0.0005 --dropout 0.2

# GPU training
drugkit-train --labeled data/labeled.csv --output models/model.pth --device cuda:0

# Resume from checkpoint
drugkit-train --labeled data/labeled.csv --output models/model.pth \
    --resume models/checkpoint.pth
```

### Python API

```python
from stages import run_train

model_path = run_train(
    labeled_file="data/labeled.csv",
    output_file="models/model.pth",
    smiles_col="SMILES",
    score_col="score",
    hidden_dim=64,
    num_layers=4,
    dropout=0.1,
    lr=0.001,
    epochs=50,
    batch_size=32,
    margin=1.0,
    patience=10,
    val_frac=0.15,
    device="cpu",      # or "cuda", "cuda:0"
    seed=42,
)
print(f"Model saved: {model_path}")
```

### Parameters

| Parameter | CLI flag | Default | Description |
|-----------|----------|---------|-------------|
| `labeled_file` | `--labeled` | — | CSV with SMILES + scores |
| `output_file` | `--output` | `models/model.pth` | Model checkpoint path |
| `smiles_col` | `--smiles-col` | `"SMILES"` | SMILES column name |
| `score_col` | `--score-col` | `"score"` | Score column name |
| `hidden_dim` | `--hidden-dim` | `64` | GNN hidden dimension |
| `num_layers` | `--num-layers` | `4` | GIN convolution layers |
| `dropout` | `--dropout` | `0.1` | MC Dropout rate |
| `lr` | `--lr` | `0.001` | Learning rate |
| `epochs` | `--epochs` | `50` | Max training epochs |
| `batch_size` | `--batch-size` | `32` | Training batch size |
| `margin` | `--margin` | `1.0` | RankNet margin |
| `patience` | `--patience` | `10` | Early stopping patience |
| `val_frac` | `--val-frac` | `0.15` | Validation fraction |
| `device` | `--device` | auto | `cpu`, `cuda`, `cuda:0` |
| `seed` | `--seed` | `42` | Random seed |
| `resume` | `--resume` | — | Resume from checkpoint |

---

## Stage 4: Predict — Score Compounds with Trained Model

### CLI

```bash
# Basic prediction
drugkit-predict --model models/model.pth --input data/pool.csv

# MC Dropout uncertainty with 20 passes
drugkit-predict --model models/model.pth --input data/pool.csv --mc-samples 20

# Multi-GPU inference
drugkit-predict --model models/model.pth --input data/large_library.csv \
    --gpu-ids 0,1,2,3 --batch-size 1024

# Streaming mode for billion-scale
drugkit-predict --model models/model.pth --input data/large_library.csv \
    --streaming --chunk-size 100000

# Keep only top 1000 predictions
drugkit-predict --model models/model.pth --input data/pool.csv --top-k 1000
```

### Python API

```python
from stages import run_predict

output_path = run_predict(
    model_path="models/model.pth",
    input_file="data/pool.csv",
    output_file="output/predictions.csv",
    smiles_col="SMILES",
    batch_size=256,
    mc_samples=10,         # 0 = deterministic (fast)
    device="cpu",
    streaming=False,       # True for very large files
    top_k=50,             # keep only top 50
)

# Read results
import pandas as pd
preds = pd.read_csv(output_path)
print(preds[["SMILES", "predicted_score", "uncertainty"]].head(10))
```

### Parameters

| Parameter | CLI flag | Default | Description |
|-----------|----------|---------|-------------|
| `model_path` | `--model` | — | Trained model .pth |
| `input_file` | `--input` | — | CSV with SMILES |
| `output_file` | `--output` | `output/predictions.csv` | Output CSV |
| `batch_size` | `--batch-size` | `256` | Inference batch size |
| `mc_samples` | `--mc-samples` | `10` | MC Dropout passes |
| `device` | `--device` | auto | Compute device |
| `gpu_ids` | `--gpu-ids` | — | Multi-GPU: `0,1,2,3` |
| `streaming` | `--streaming` | `False` | Stream large files |
| `chunk_size` | `--chunk-size` | `100000` | Streaming chunk |
| `top_k` | `--top-k` | all | Only output top K |

---

## Stage 5: Active Learn — Full Iterative Loop

### CLI

```bash
# From YAML config
drugkit-active-learn --config config/drugkit.yaml

# Override parameters
drugkit-active-learn --config config/drugkit.yaml \
    --rounds 5 --acquisition-fn ucb --acquisition-batch 20

# Without config file
drugkit-active-learn --labeled data/seed_ligands.csv --pool data/pool.csv \
    --rounds 3 --seed-size 10 --acquisition-batch 5 --epochs 30

# With docking oracle
drugkit-active-learn --labeled data/seed_ligands.csv --pool data/pool.csv \
    --receptor data/TARGET_A.pdbqt --center 12.5,34.2,8.7 --dock-engine smina
```

### Python API

```python
from stages import run_active_learn

output_dir = run_active_learn(
    labeled_file="data/seed_ligands.csv",
    pool_file="data/pool.csv",
    output_dir="output/al_experiment",
    rounds=3,
    seed_size=4,
    acquisition_batch=5,
    acquisition_fn="greedy",    # greedy, uncertainty, ucb, thompson, random
    epochs=30,
    hidden_dim=64,
    mc_samples=10,
    device="cpu",
    seed=42,
)
# Results in output/al_experiment/
```

### Parameters

| Parameter | CLI flag | Default | Description |
|-----------|----------|---------|-------------|
| `config_path` | `--config` | — | YAML config file |
| `labeled_file` | `--labeled` | — | Initial labeled CSV |
| `pool_file` | `--pool` | — | Unlabeled pool CSV |
| `output_dir` | `--output-dir` | `output/active_learning` | Results dir |
| `rounds` | `--rounds` | `5` | AL iterations |
| `seed_size` | `--seed-size` | `50` | Initial seed size |
| `acquisition_batch` | `--acquisition-batch` | `20` | Compounds/round |
| `acquisition_fn` | `--acquisition-fn` | `"greedy"` | Acquisition function |
| `epochs` | `--epochs` | `50` | Epochs per round |
| `hidden_dim` | `--hidden-dim` | `64` | GNN dimension |
| `mc_samples` | `--mc-samples` | `10` | Uncertainty samples |
| `device` | `--device` | auto | Compute device |
| `dock_engine` | `--dock-engine` | `"smina"` | Docking backend |
| `receptor` | `--receptor` | — | Receptor for oracle |
| `center` | `--center` | — | Binding site `x,y,z` |

---

## Stage 6: Select — Selectivity Analysis

### CLI

```bash
# Ratio-based selectivity
drugkit-select --input data/multi_target_scores.csv \
    --on-targets CDK2 --off-targets CDK4 CDK6

# Pareto-based with top-K filter
drugkit-select --input data/multi_target_scores.csv \
    --on-targets CDK2 --off-targets CDK4 CDK6 \
    --method pareto --top-k 20

# From per-target result files
drugkit-select --docking-results output/per_target/ \
    --on-targets TARGET_A --off-targets OFFTARGET_B
```

### Python API

```python
from stages import run_select

result_path = run_select(
    input_file="data/multi_target_scores.csv",
    output_file="output/selective_hits.csv",
    on_targets=["CDK2"],
    off_targets=["CDK4", "CDK6"],
    method="ratio",          # ratio, difference, pareto
    threshold=1.5,
    top_k=20,
    ascending=True,          # lower score = better (docking)
)

import pandas as pd
hits = pd.read_csv(result_path)
print(hits[["Name", "SMILES", "score_CDK2", "score_CDK4", "selectivity"]].head())
```

### Parameters

| Parameter | CLI flag | Default | Description |
|-----------|----------|---------|-------------|
| `input_file` | `--input` | — | CSV with multi-target scores |
| `docking_results_dir` | `--docking-results` | — | Directory of per-target CSVs |
| `on_targets` | `--on-targets` | — | Targets to optimize FOR |
| `off_targets` | `--off-targets` | — | Targets to penalize |
| `method` | `--method` | `"ratio"` | `ratio`, `difference`, `pareto` |
| `threshold` | `--threshold` | `2.0` | Selectivity filter |
| `top_k` | `--top-k` | all | Keep only top K |
| `ascending` | `--no-ascending` | `True` | Lower = better |

---

## Stage 7: Expand — Ligand Expansion

### CLI

```bash
# Basic expansion
drugkit-expand --seeds data/seed_ligands.csv --output output/expanded.csv

# More analogs, higher similarity
drugkit-expand --seeds data/seed_ligands.csv --output output/expanded.csv \
    --max-analogs 50 --similarity-cutoff 0.7

# Scaffold hopping
drugkit-expand --seeds data/seed_ligands.csv --method scaffold_hopping

# With external database
drugkit-expand --seeds data/seed_ligands.csv --database chembl_subset.csv
```

### Python API

```python
from stages import run_expand

result_path = run_expand(
    seeds_file="data/seed_ligands.csv",
    output_file="output/expanded_library.csv",
    method="similarity",        # similarity, scaffold_hopping, fragment
    max_analogs=20,
    similarity_cutoff=0.6,
    clustering_cutoff=0.8,
    deduplicate=True,
)

import pandas as pd
expanded = pd.read_csv(result_path)
print(f"Expanded library: {len(expanded)} compounds")
```

### Parameters

| Parameter | CLI flag | Default | Description |
|-----------|----------|---------|-------------|
| `seeds_file` | `--seeds` | — | CSV with seed SMILES |
| `output_file` | `--output` | `output/expanded_library.csv` | Output CSV |
| `method` | `--method` | `"similarity"` | Expansion strategy |
| `max_analogs` | `--max-analogs` | `20` | Analogs per seed |
| `similarity_cutoff` | `--similarity-cutoff` | `0.6` | Min Tanimoto |
| `clustering_cutoff` | `--clustering-cutoff` | `0.8` | Cluster dedup |
| `database` | `--database` | — | External analog DB |
| `deduplicate` | `--no-deduplicate` | `True` | Remove duplicates |

---

## Full Experiment — End-to-End Script

```bash
#!/bin/bash
# run_mock_experiment.sh — Full DrugKit pipeline with mock data

set -e

echo "=== Step 0: Generate mock data ==="
python generate_mock_data.py

echo "=== Step 1: Featurize pool ==="
drugkit-featurize --input data/pool.csv --output output/graphs.pt --encoder rdkit

echo "=== Step 2: Train initial model on labeled data ==="
drugkit-train --labeled data/labeled.csv --output models/model.pth \
    --epochs 30 --hidden-dim 64 --device cpu

echo "=== Step 3: Predict on full pool ==="
drugkit-predict --model models/model.pth --input data/pool.csv \
    --output output/predictions.csv --mc-samples 10

echo "=== Step 4: Selectivity analysis ==="
drugkit-select --input data/multi_target_scores.csv \
    --on-targets CDK2 --off-targets CDK4 CDK6 \
    --output output/selective.csv --top-k 20

echo "=== Step 5: Expand top hits ==="
drugkit-expand --seeds output/selective.csv --output output/expanded.csv \
    --max-analogs 10 --similarity-cutoff 0.6

echo "=== Step 6: Score expanded library ==="
drugkit-predict --model models/model.pth --input output/expanded.csv \
    --output output/final_ranked.csv --mc-samples 20 --top-k 50

echo "=== Final candidates in output/final_ranked.csv ==="
```

Or equivalently in Python:

```python
"""run_mock_experiment.py — Full pipeline via Python API."""
from stages import (
    run_featurize, run_train, run_predict, run_select, run_expand
)

# Step 1: Featurize
graphs, names, failed = run_featurize(
    input_file="data/pool.csv",
    output_file="output/graphs.pt",
    encoder="rdkit",
)

# Step 2: Train
model_path = run_train(
    labeled_file="data/labeled.csv",
    output_file="models/model.pth",
    epochs=30,
    hidden_dim=64,
    device="cpu",
)

# Step 3: Predict
run_predict(
    model_path="models/model.pth",
    input_file="data/pool.csv",
    output_file="output/predictions.csv",
    mc_samples=10,
)

# Step 4: Selectivity
run_select(
    input_file="data/multi_target_scores.csv",
    output_file="output/selective.csv",
    on_targets=["CDK2"],
    off_targets=["CDK4", "CDK6"],
    top_k=20,
)

# Step 5: Expand
run_expand(
    seeds_file="output/selective.csv",
    output_file="output/expanded.csv",
    max_analogs=10,
)

# Step 6: Re-score expanded library
run_predict(
    model_path="models/model.pth",
    input_file="output/expanded.csv",
    output_file="output/final_ranked.csv",
    mc_samples=20,
    top_k=50,
)

print("Done! Final candidates: output/final_ranked.csv")
```

---

## Dependencies Required

```
numpy>=1.24
pandas>=1.5
torch>=2.0
torch-geometric>=2.3
rdkit>=2023.03
scipy>=1.10
scikit-learn>=1.2
pyyaml>=6.0
tqdm>=4.60
meeko>=0.5       # for PDBQT ligand prep (only needed for docking)
biopython>=1.81  # for PDB parsing (only needed for structure prep)
```

Install all at once:
```bash
pip install drugkit          # pulls all required deps
# or manually:
pip install numpy pandas torch torch-geometric scipy scikit-learn pyyaml tqdm
conda install -c conda-forge rdkit
```

**External tools** (only needed for real docking, not mock experiments):
- `smina` binary — download from https://sourceforge.net/projects/smina/
- `vina` binary — download from https://github.com/ccsb-scripps/AutoDock-Vina
- `p2rank` — download from https://github.com/rdk/p2rank (for pocket detection)

---

## Data Format Reference

| Stage | Input | Required Columns | Output |
|-------|-------|-----------------|--------|
| Featurize | CSV | `SMILES` | `.pt` file with PyG graphs |
| Dock | CSV + receptor | `SMILES` | CSV: `Name,SMILES,score` |
| Train | CSV | `SMILES`, score col | `.pth` model checkpoint |
| Predict | CSV + model | `SMILES` | CSV: `predicted_score,uncertainty` |
| Active Learn | CSV + pool | `SMILES` | directory with model + history |
| Select | CSV | `score_<target>` cols | CSV: `selectivity` column added |
| Expand | CSV | `SMILES` | CSV: expanded analogs |

---

## Comparison with Original Deep Docking Protocol

| Aspect | Original DD | DrugKit (this package) |
|--------|-------------|------------------------|
| **Install** | Clone repo + manual scripts | `pip install drugkit` |
| **Featurization** | Morgan FP 1024-bit | Graph (42 atom + 11 bond features) |
| **Model** | 3-layer MLP, binary | GINE Siamese RankNet, continuous |
| **Iteration** | Manual re-run per phase | `drugkit-active-learn --rounds 5` |
| **Selection** | Random sampling | Uncertainty-guided acquisition |
| **Selectivity** | Post-hoc filtering | Built into training loop |
| **Scale** | Single script per file | `--streaming --gpu-ids 0,1,2,3` |
| **Uncertainty** | None | MC Dropout σ per compound |
| **Reproducibility** | Multiple scripts + paths | Single YAML config + `--seed` |

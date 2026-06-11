# DrugKit — Usage Guide

Complete guide for using DrugKit: a GNN-Powered Deep Docking framework for accelerated virtual screening.

## Installation

### From TestPyPI

```bash
pip install --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ \
    drugkit
```

### From Source (Development)

```bash
git clone https://github.com/kugi8412/DrugKit/
cd DrugKit
conda env create -f config/drugkit.yaml -n drugkit
conda activate drugkit
pip install -e ".[dev]"
```

## Quick Start

### 1. SMILES to Graph Conversion (RDKit-free)

```python
from smiles_processing import smiles_to_pyg, batch_smiles_to_pyg, ATOM_FEATURE_DIM, BOND_FEATURE_DIM

# Single molecule
data = smiles_to_pyg("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
print(f"Atoms: {data.x.shape[0]}, Features: {data.x.shape[1]}")
print(f"Bonds: {data.edge_index.shape[1] // 2}, Edge features: {data.edge_attr.shape[1]}")

# Batch processing
smiles_list = ["CCO", "c1ccccc1", "CC(=O)O", "CCN"]
graphs = batch_smiles_to_pyg(smiles_list)
```

**Feature dimensions:**
- Node features: 42 (atom type, degree, H-count, charge, hybridization, aromaticity, mass, chirality)
- Edge features: 11 (bond type, conjugation, ring membership, stereochemistry)

### 2. RDKit-Based Featurization

```python
from siamese_GNN import smiles_to_graph_gine, feature_dims

# Get feature dimensions
node_dim, edge_dim = feature_dims()  # (42, 11)

# Convert with selectivity label
graph = smiles_to_graph_gine("CCO", selectivity=-7.5, is_elite=True)
print(f"Label: {graph.y}, Elite: {graph.is_elite}")
```

### 3. Model Training (Siamese RankNet)

```python
import torch
import pandas as pd
import numpy as np
from siamese_GNN import SiameseRankNet, feature_dims
from siamese_GNN.trainer import build_labeled_graphs, train_ranknet

# Generate synthetic docking_results.csv data
np.random.seed(42)

# Synthetic SMILES
smiles_list = [
    "CCO",  # Ethanol
    "c1ccccc1",  # Benzene
    "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
    "CCN(CC)c1ccc(cc1)N(CC)CC",  # Dye
    "CC(C)CC(N)C(=O)O",  # Leucine
    "c1ccc2c(c1)ccc3c2cccc3",  # Anthracene
    "Cc1ccccc1S(=O)(=O)N",  # Toluenesulfonamide
    "CC(C)C(C(=O)O)N",  # Valine
    "c1ccc(cc1)C(=O)O",  # Benzoic acid
    "CCCCCCc1ccccc1",  # Hexylbenzene
]

# Generate synthetic dataset
n_samples = 200
data = {
    "SMILES": [smiles_list[i % len(smiles_list)] for i in range(n_samples)],
    "selectivity": np.random.uniform(-10, -5, n_samples),  # Binding affinity (kcal/mol)
    "Cluster_ID": np.random.randint(0, 10, n_samples),  # 10 clusters
}

df = pd.DataFrame(data)
print("Generated synthetic docking results:")
print(df.head(10))
print(f"Total records: {len(df)}")

# Prepare labeled data (from docking results)
graphs, clusters = build_labeled_graphs(
    df, sel_col="selectivity", cluster_col="Cluster_ID", elite_count=10
)

# Train
node_dim, edge_dim = feature_dims()
cfg = {
    "elite_penalty": 5.0,
    "val_target_ratio": 0.15,
    "batch_size": 32,
    "epochs": 40,
    "learning_rate": 0.0004,
    "hidden_dim": 128,
    "dropout": 0.3,
    "seed": 42,
}
model, history = train_ranknet(graphs, clusters, node_dim, edge_dim, cfg, device="cuda")

# Save
torch.save(model.state_dict(), "model.pth")
print("Model saved to model.pth")
```

### 4. Active Learning Loop

```python
from active_learning import run_active_learning

cfg = {
    "pool_file": "data/pool.csv",          # large SMILES library
    "seed_file": "data/seed_ligands.csv",  # initial known binders
    "grids_file": "docking_grids.json",    # pocket definitions
    "on_targets": ["SLC6A20"],
    "off_targets": ["SLC6A19"],
    "rounds": 5,
    "seed_size": 20,
    "acquisition_batch": 10,
    "mc_samples": 30,
    "epochs": 40,
    "hidden_dim": 128,
    "dropout": 0.3,
    "output_dir": "output/active_learning"
}

result = run_active_learning(cfg, device="cuda")
print(f"Final labeled set: {len(result['labeled'])} compounds")
print(f"Best validation Spearman rho: {result['history'][-1]['val_rho']:.3f}")
```

### 5. Billion-Scale Inference

```python
from inference import batch_predict, batch_predict_from_file, MultiGPUPredictor

# Single batch
from siamese_GNN import SiameseRankNet, feature_dims
model = SiameseRankNet(*feature_dims(), hidden_dim=128, dropout=0.3)
model.load_state_dict(torch.load("model.pth"))

scores, uncertainties, valid_mask = batch_predict(
    model, smiles_list,
    device="cuda",
    batch_size=512,
    mc_samples=30,  # 0 for deterministic
)

# Stream from file
total = batch_predict_from_file(
    model,
    input_file="zinc_billion.csv",
    output_file="predictions.csv",
    device="cuda",
    chunk_size=50000,
    batch_size=512,
)

# Multi-GPU
predictor = MultiGPUPredictor(model, gpu_ids=[0, 1, 2, 3])
scores, valid_mask = predictor.predict(smiles_list, batch_size=1024)

# Multi-GPU file processing
predictor.predict_file(
    "zinc_billion.csv", "predictions.csv",
    batch_size=1024, chunk_size=100000
)
```

### 6. Docking (Smina/Vina)

```python
from docking_common import load_grids, build_receptor_map, prepare_ligand
from docking_smina import run_smina_scoring

# Prepare receptor
grids = load_grids("docking_grids.json", logger)
rec_map = build_receptor_map(grids, data_dir="data/", logger=logger)

# Dock a single molecule
pdbqt_string, mol = prepare_ligand("CC(=O)Oc1ccccc1C(=O)O", "aspirin")
score, pose = run_smina_scoring(
    receptor_path="data/receptor.pdbqt",
    pdbqt_ligand=pdbqt_string,
    center=[10.0, 20.0, 30.0],
    size=[20.0, 20.0, 20.0],
    base_exhaustiveness=8,
)
print(f"Binding energy: {score:.2f} kcal/mol")
```

### 7. Pocket Detection

```python
from pocket_logic import ConfigLoader

loader = ConfigLoader("config.yaml")
```

## Pipeline Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐
│ SMILES Pool │────▶│ Featurize    │────▶│ GNN Prediction  │
│ (billions)  │     │ (PyG graphs) │     │ (score + σ)     │
└─────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                    ┌──────────────┐               │ MC Dropout
                    │ Exact Dock   │◀──────────────┘ Uncertainty
                    │ (Smina/Vina) │  Top-K uncertain
                    └──────┬───────┘
                           │
                    ┌──────▼───────┐     ┌─────────────────┐
                    │ Selectivity  │────▶│ Retrain GNN     │
                    │ Scoring      │     │ (RankNet loss)  │
                    └──────────────┘     └─────────────────┘
                                               ↺ repeat
```

## Configuration

All pipeline settings are in `config.yaml`:

```yaml
project:
  data_dir: data

active_learning:
  pool_file: data/pool.csv
  seed_file: data/seed_ligands.csv
  grids_file: docking_grids.json
  on_targets: [SLC6A20]
  off_targets: [SLC6A19]
  rounds: 5
  epochs: 40
  hidden_dim: 128
  mc_samples: 30
  smina_exe: smina
  exhaustiveness: 8
```

## File Formats

### Pool CSV (input)
```csv
Name,SMILES
Compound_001,CCO
Compound_002,c1ccccc1
```

### Docking Grids JSON
```json
{
  "SLC6A20": [
    {"id": "pocket_1", "center": [10.0, 20.0, 30.0], "size": [20.0, 20.0, 20.0]}
  ]
}
```

### Results CSV (output)
```csv
Name,SMILES,Target,Pocket_ID,Energy,Beat_Baseline,Selectivity
mol_001,CCO,SLC6A20,pocket_1,-8.3,True,-2.1
```

## Running Tests

```bash
# All tests
pytest tests/ -v

# Quick (no integration)
pytest tests/ -v -k "not test_full_active_learning"

# Encoder parity (requires ESOL dataset)
pytest tests/test_encoder_parity.py -v

# Pipeline integration
pytest tests/test_pipeline_integration.py -v
```

## Module Reference

| Module | Purpose |
|--------|---------|
| `smiles_processing` | RDKit-free SMILES → PyG conversion |
| `siamese_GNN` | GINE Siamese RankNet model + training |
| `active_learning` | Iterative AL loop with MC dropout |
| `inference` | Batch + multi-GPU prediction |
| `docking_common` | Shared docking utilities |
| `docking_smina` | Smina docking backend |
| `docking_vina` | AutoDock Vina backend |
| `docking_vina_etc` | Extended Vina with pocket selection |
| `pocket_logic` | Pocket detection (P2Rank) |
| `expand_ligands` | Tanimoto clustering + SmallWorld expansion |
| `final_docking` | Final validation docking |
| `structure_prep` | PDB structure preparation |
| `summarizer` | Selectivity aggregation |

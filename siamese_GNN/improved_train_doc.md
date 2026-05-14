# Purpose

Train a graph neural network ranking model to score ligands by selectivity using docking-derived features.
The script converts ligand SMILES into molecular graphs, builds all pairwise comparisons inside train/validation splits, trains a Siamese GINE-based RankNet with weighted loss (higher weight for "elite" ligands), tracks Spearman correlation between true selectivity and predicted scores, saves the best model, and exports training plots.

---

# Process Flow

1. Load configuration (paths, targets, training hyperparameters, seed).
2. Read wide docking CSV (`output/docking_energies_etc_wide.csv`).
3. Derive `Cluster_ID` from `Name` (suffix after `_of_`) to use as group labels.
4. Convert configured energy columns and `Selectivity` to numeric.
5. Compute helper energy summaries per row:

   * `target_worst_E` = max energy among `target_cols`
   * `target_best_E` = min energy among `target_cols`
   * `homolog_best_E` = min energy among `homolog_cols`
6. Filter rows with missing `SMILES`, `Selectivity`, or `Cluster_ID`.
7. Define training label:

   * `selectivity_score = Selectivity`
8. Select "elite" ligands:

   * take `elite_count` unique SMILES from the lowest `selectivity_score` rows (ascending sort)
   * mark them with `is_elite = 1`
9. Convert each SMILES into a PyTorch Geometric graph:

   * node features from atom properties
   * edge features from bond properties
   * graph label `y = selectivity_score`
10. Split graphs into train/validation using `GroupShuffleSplit` on `Cluster_ID` to approximate `val_target_ratio`.
11. Build pairwise datasets (`AllPairsDataset`) using all combinations of graphs:

* label = 1 if y1 > y2, 0 if y1 < y2, 0.5 if equal
* weight = `elite_penalty` if either graph is elite, else 1.0

12. Train `SiameseRankNet` (GINE encoder + MLP head) for `epochs`:

* loss: weighted BCE on (s1 - s2) vs label
* gradient clipping
* LR schedule: cosine annealing

13. Each epoch evaluate:

* Spearman rho between true selectivity and predicted single-graph scores
* average pairwise loss for train and validation

14. Save best model by highest validation rho to `results/GINE_model.pth`.
15. Save training curves plot to `results/training_plots.png`.

---

# Inputs

* **CSV (wide docking results)**: `output/docking_energies_etc_wide.csv`
  Must contain at least:

  * `Name` (used to derive `Cluster_ID`)
  * `SMILES` (ligand structure)
  * `Selectivity` (training target)
  * Energy columns listed in:

    * `CONFIG['target_cols']` (default: `['SIT1_MODEL_00_Energy','8WM3_Energy','8I91_Energy']`)
    * `CONFIG['homolog_cols']` (default: `['8I92_Energy','8WBY_Energy']`)
* **Config constants (in-script)**:

  * Paths: `model_save_path`, `plot_save_path`
  * Train setup: `epochs`, `batch_size`, `learning_rate`, `hidden_dim`, `dropout`, `seed`
  * Split: `val_target_ratio`, `cluster_col`
  * Elite weighting: `elite_count`, `elite_penalty`
* **Environment / dependencies**:

  * Python + PyTorch + PyTorch Geometric
  * RDKit
  * NumPy, Pandas, SciPy, scikit-learn
  * Matplotlib (Seaborn is imported but not required by core flow)

---

# Outputs

* **Trained model weights**: `results/GINE_model.pth`
  Saved when validation Spearman rho improves.
* **Training plots**: `results/training_plots.png`
  Two panels:

  * train/val RankNet loss over epochs
  * train/val Spearman rho over epochs
* **Logs to stdout**:

  * dataset size, split ratio, epoch metrics, best rho

---

# Command

```bash
python src/siamese_GNN/improved_train.py
```

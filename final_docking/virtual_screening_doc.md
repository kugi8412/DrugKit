# Purpose

Perform GNN-guided virtual screening and analog expansion starting from docking results.  
The script selects high-quality seed ligands based on affinity and selectivity, scores them with a pretrained Siamese GINE RankNet model, retrieves structurally similar compounds from PubChem, re-scores the analogs using the GNN, and outputs a balanced set of seeds and top-performing novel candidates.

The goal is to move beyond raw docking by using a learned ranking model to prioritize compounds with improved predicted performance.

---

# Process Flow

1. Load configuration (paths, thresholds, model parameters, device).
2. Read docking results CSV (`train_data`).
3. Identify target and off-target energy columns using configured protein keys.
4. Compute per-ligand metrics:
   - `Best_Target` = best (lowest) binding energy to target
   - `Best_OffTarget` = best (lowest) binding energy to off-target
   - `Selectivity` = Best_OffTarget − Best_Target
5. Filter out weak or invalid binders using `min_binding_energy`.
6. Select seed ligands:
   - Top-N by best target affinity
   - Top-N by highest selectivity
7. Load pretrained Siamese GINE RankNet model.
8. Convert seed SMILES to molecular graphs and predict GNN scores.
9. For each seed:
   - query PubChem for structurally similar compounds (similarity search),
   - remove compounds already present in the docking dataset,
   - convert analogs to graphs and predict GNN scores in batches,
   - keep top analogs that outperform the parent seed.
10. Collect seeds and selected analogs into a final results table.
11. Save results to CSV.

---

# Inputs

- **Docking results CSV** (`CONFIG['train_data']`, default: `smina_results_exact.csv`)  
  Required columns:
  - `Name`
  - `SMILES`
  - docking energy columns containing:
    - target protein key (e.g. `SLC6A20`)
    - off-target protein key (e.g. `SLC6A19`)

- **Pretrained GNN model** (`CONFIG['model_path']`)  
  Siamese GINE RankNet trained to rank ligands by selectivity.

- **External services**
  - PubChem (via `pubchempy`) for similarity-based analog retrieval.

- **Key configuration parameters**
  - Seed selection: `top_n_affinity`, `top_n_selectivity`
  - Energy filter: `min_binding_energy`
  - Analog search: `similarity_threshold`, `candidates_pool_size`
  - Analog selection: `keep_top_analogs`
  - Model: `hidden_dim`, `batch_size`, `device`

---

# Outputs

- **Final candidates CSV** (`CONFIG['output_file']`, default: `final_candidates_mixed_v2.csv`)  
  Contains:
  - `SMILES`
  - `Name`
  - `GNN_Score`
  - `Source` (seed type or analog origin)
  - `Parent_Score` (GNN score of the seed)

- **Console logs**
  - number of valid compounds after filtering,
  - selected seeds and their metrics,
  - number of analogs retrieved and selected per seed,
  - final count of collected structures.

---

# Command

```bash
python src/final_docking/virtual_screenig.py
```


## Stage 5 - Docking Expansion

### Purpose

This stage performs **docking expansion** for clustered ligands against **multiple protein conformations** using only the **best pocket per target** selected in Stage 4.  
It computes: per-target docking energies, baseline comparison (hit / no-hit) and saves poses only for ligands that beat the baseline.

This stage is designed to answer:  
**Which cluster members generalize well across targets and are selective for a given protein?**

---

### Files

Main entry point:
- `docking_vina_etc.py`

Core modules:
- `fs_utils.py` - filesystem helpers
- `grids_io.py` - pocket grid loader
- `receptor_io.py` - receptor (.pdbqt) discovery
- `ligand_prep.py` - SMILES -> 3D -> PDBQT
- `pocket_selection.py` - best pocket & baseline resolution
- `vina_runner.py` - AutoDock Vina wrapper
- `worker.py` - per-ligand multiprocessing worker
- `selectivity.py` - cross-protein selectivity logic

---

### Workflow

1. Load configuration and resolve paths
2. Load **Stage 4 representative docking results**
3. Identify:
   - best pocket per (Cluster_ID, Target)
   - baseline energy per target
4. Load cluster members (ligands to expand)
5. Load pocket grids (Stage 3 output)
6. Resolve receptor `.pdbqt` files
7. Parallel docking:
   - each ligand docked only into its cluster-selected pockets
8. Save:
   - raw energy table
   - final table with selectivity
   - poses for baseline-beating hits only

---

### Inputs

#### 1. Configuration (`config.yaml`)

Relevant keys (under `docking`):

- `grids_file` - pocket grids JSON (Stage 3)
- `results_reps_file` - Stage 4 representative results
- `results_known_file` - baseline docking of known ligands
- `cluster_members_file` - cluster expansion set
- `exhaustiveness_etc` - base Vina exhaustiveness (default: 16)
- `n_cpu` - number of parallel workers
- `default_baseline` - fallback baseline energy (default: -7.0)

#### 2. Cluster members CSV

Required columns:
- `Name`
- `SMILES`
- `Cluster_ID`

#### 3. Pocket grids JSON

Format:
```json
{
  "TargetID": [
    { "id": "...", "center": [], "size": [] }
  ]
}
```

#### 4. Receptors

PDBQT files named as target conformation keys, searched in:

* `output/`
* `data/`
* `data_dir` from config

---

### Outputs

#### Main outputs

* `output/docking_energies_etc.csv`
  Long-format table:

  * `Name, SMILES, Cluster_ID, Target, Pocket_ID, Energy, Beat_Baseline`

* `output/docking_results_etc.csv`
  Final ranked table with:

  * `Selectivity_Min` added
  * sorted by `Beat_Baseline`, `Selectivity_Min`, `Energy`

#### Poses

* `output/etc_poses/*.pdbqt`
* saved **only if** `Energy < Baseline_Value`
* filename:

  ```
  <LigandName>_<TargetID>_<PocketID>.pdbqt
  ```

#### Logs

* `logs/docking_vina_etc.log`

---

### Logic

#### Pocket selection

For each `Cluster_ID` and `Target`:

* choose the **single pocket** with lowest energy from Stage 4 results
* pockets are fixed for all cluster members

#### Baseline logic

Baseline per target is:

* minimum energy from known-compound docking (Stage 4), or
* `default_baseline` if unavailable

A ligand is a **hit** if:

```
Energy < baseline[target]
```

#### Selectivity

For each ligand and target:

```
Selectivity(target) =
  min(E_other_targets_from_other_proteins - E_target)
```

Interpretation:

* higher value = better selectivity
* 0.0 if no other proteins exist
* NaN if undefined

---

### Execution

From project root (with `config.yaml` present):

```bash
python src/docking_vina_etc/docking_vina_etc_r.py
```

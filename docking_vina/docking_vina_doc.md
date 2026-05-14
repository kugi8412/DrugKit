## Stage 4 - Docking (AutoDock Vina, docking_vina.py)

This stage docks **known compounds (baseline)** and then **candidate compounds** into all generated pockets (Stage 3 grids) for each target structure. It outputs a ranked docking table and saves PDBQT poses for hits.

Files involved: `docking_vina.py`, `src/docking_vina/*`.

---

### What this script does

1. **Loads config.yaml** and merges it with defaults (docking section).
2. **Loads pocket grids JSON** (output from Stage 3).
3. **Builds a receptor map**: for each target key (structure), locate a matching receptor `.pdbqt`.
4. **Step 1: Baseline / cross-docking of known ligands**

   * docks each known compound against *every* target + pocket
   * computes per-target **baseline thresholds** from best (minimum) energy
   * saves raw baseline results to `*_known_compounds.csv`
5. **Step 2: Docking candidates**

   * docks each candidate ligand against every target + pocket
   * marks a candidate as a **hit** if it beats the baseline threshold for that target
   * saves PDBQT pose files only for hits
6. Writes:

   * main results CSV
   * summary matrix CSV (energies per ligand per target)

---

### Inputs

#### 1) Receptors (from Stage 2)

This docking stage requires receptor files in **PDBQT** format (`*.pdbqt`). The script searches for each target key in multiple locations, including `data_dir`. 

Search order examples:

* `output/<target>.pdbqt`
* `data/<target>.pdbqt`
* `<data_dir>/<target>.pdbqt`
  (and same with sanitized key where `/` and `\` become `_`). 

#### 2) Pocket grids JSON (from Stage 3)

Path: `docking.grids_file` (default `docking_grids.json`).
Expected format: `{"TargetKey": [{"id":..., "center":[...], "size":[...]}, ...], ...}`.

#### 3) Ligand tables (CSV)

* `docking.known_compounds` (baseline set)
* `docking.candidates_file` (screening set)

Required columns (both):

* `Name`
* `SMILES`

Optional columns used:

* `Target` for known compounds (stored as `Original_Target`) 
* `Cluster_ID` for candidates (propagated to output) 

---

### Outputs

Configured by `docking.results_file` (default `output/docking_results.csv`).

Creates:

1. `output/docking_results.csv`

   * one row per (ligand, target, pocket)
   * columns: `Name, SMILES, Target, Pocket_ID, Energy, Beat_Baseline, Baseline_Value, Cluster_ID` 
2. `output/docking_results_summary_matrix.csv`

   * pivot table: best energy per ligand per target 
3. `output/docking_results_known_compounds.csv`

   * baseline docking results of known ligands 
4. `output/top_poses/*.pdbqt`

   * saved only for hits (Beat_Baseline == True)
   * filename pattern: `<LigandName>_<TargetKey>_<PocketID>.pdbqt` (sanitized)

Logs:

* `logs/docking_vina.log` 

---

### How ligand preparation works

Each SMILES is converted into a 3D molecule and then into a PDBQT string:

* RDKit: add H, generate conformer (ETKDG), optimize (UFF when possible) 
* Meeko: generate a Vina-compatible PDBQT string (legacy writer) 

If embedding fails, the ligand is skipped (no docking).

---

### Docking Vina engine

Docking is executed via the Python `vina` package
Per docking call:

* `v.set_receptor(receptor_path)`
* `v.set_ligand_from_string(pdbqt_ligand)`
* `v.compute_vina_maps(center=<center>, box_size=<size>)`
* `v.dock(exhaustiveness=<...>, n_poses=1)`
* returns best affinity (kcal/mol) and the top pose string

Dynamic exhaustiveness:

* base value from config (`docking.exhaustiveness`, default 8)
* if pocket volume `size[0]*size[1]*size[2] > 27000 A^3`, exhaustiveness is raised to at least 32 

---

### Baseline logic

Baseline step computes, for each target key:

* threshold = **minimum energy** among all dockings of known compounds into that target (across all pockets) 

Candidate is a hit if:

* `Energy < threshold[target]`

If known compounds file is missing or docking fails:

* thresholds default to `-7.0` for all targets 

---

### Parallelization (CPU)

Docking is parallelized with `ProcessPoolExecutor(max_workers=n_cpu)` for both baseline and candidates. 

`n_cpu` is computed from config field `docking.n_poses` (name is misleading: it is used as CPU count). If missing, it falls back to `os.cpu_count()`. 

---

### Run

From repo root (where `config.yaml` exists):

```bash
python src/docking_vina/docking_vina.py
```
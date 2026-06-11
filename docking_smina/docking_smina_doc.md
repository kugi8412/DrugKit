## Stage 4b - Docking (Smina, docking_smina)

This stage mirrors `docking_vina/` but uses the Smina CLI for higher-precision docking. It shares ligand preparation, grid loading, and receptor discovery via `docking_common/`.

Files involved: `docking_smina/docking_smina_r.py`, `docking_smina/*`, `docking_common/*`.

---

### What this script does

Same workflow as AutoDock Vina stage:

1. Loads `config.yaml` (`docking_smina:` section)
2. Loads pocket grids JSON
3. Builds receptor map (PDB or PDBQT; prefers PDBQT when both exist)
4. Baseline docking of known compounds → per-target thresholds
5. Candidate docking with hit detection vs baseline
6. Writes CSV results, summary matrix, and hit poses

---

### Config (`docking_smina:`)

```yaml
docking_smina:
  known_compounds: data/known_compounds.csv
  candidates_file: data/candidates.csv
  grids_file: docking_grids.json
  results_file: output/docking_smina_results.csv
  smina_exe: smina
  exhaustiveness: 16
  num_modes: 1
  n_cpu: 4
  default_baseline: -7.0
```

---

### Outputs

- `output/docking_smina_results.csv`
- `output/docking_smina_results_known_compounds.csv`
- `output/docking_smina_results_summary_matrix.csv`
- `output/smina_top_poses/*.pdbqt` (hits only)
- `logs/docking_smina.log`

---

### Run

```bash
python docking_smina/docking_smina_r.py
```

---

### Smina engine

`docking_smina/smina_engine.py` exposes:

```python
run_smina_scoring(receptor_path, pdbqt_ligand, center, size, base_exhaustiveness, ...) -> (energy, pose_str)
```

Receptors may be `.pdb` or `.pdbqt`. Affinity is parsed from output PDBQT remarks, with stdout fallback.

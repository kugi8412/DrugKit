# Purpose

Run final high-precision docking with Smina for a set of selected ligands and multiple protein pockets.  
The script takes candidate ligands (SMILES) from a CSV, converts them to PDBQT, docks them against all targets/pockets defined in JSON grid files, extracts binding affinities, and writes a consolidated results CSV with one score column per (target, pocket).

---

# Process Flow

1. Load configuration (paths, Smina settings, PDB mapping).
2. Read input candidates CSV (`input_csv`).
3. Load docking grids:
   - scan current directory for `*.json`
   - merge all JSON contents into a single `grids_data` mapping: `target_key -> pockets[]`
4. Create a temporary workspace folder (`temp_docking/`), clearing it if it already exists.
5. For each `target_key` in `grids_data`:
   - resolve receptor PDB filename using `pdb_map` (fallback: `<target_key>.pdb`)
   - search for the PDB file recursively under `structures_dir`
   - skip target if structure is missing
6. For each pocket entry of the target:
   - read `pocket_id`, `center` and `size`
   - create output column name: `{target_key}_{pocket_id}_Smina`
   - if the column does not exist, initialize it with NaNs
7. For each ligand (row) in the CSV:
   - skip if the docking score for this column is already present
   - prepare ligand PDBQT from SMILES using Meeko:
     - SMILES -> RDKit molecule -> add hydrogens -> 3D embed (ETKDG) -> write PDBQT
   - run Smina docking with pocket grid parameters and configured exhaustiveness / CPU
   - parse affinity score:
     - first from docked PDBQT remarks
     - fallback to parsing Smina stdout
   - store the affinity in the corresponding `{target_key}_{pocket_id}_Smina` column
8. After each pocket, save intermediate results to `output_csv`.
9. Remove the temporary directory and print the final output path.

---

# Inputs

- **Candidates CSV** (`CONFIG['input_csv']`, default: `final_candidates_mixed_v2.csv`)  
  Required columns:
  - `Name`
  - `SMILES`

- **Docking grid JSON files** (all `*.json` in the current working directory)  
  Expected structure per target:
  - key: `target_key`
  - value: list of pockets with fields:
    - `id`
    - `center` = `[cx, cy, cz]`
    - `size` = `[sx, sy, sz]`

- **Receptor structures (PDB)**  
  Located under `structures_dir` (searched recursively).  
  Mapping is controlled by `CONFIG['pdb_map']`, with fallback to `<target_key>.pdb`.

- **External dependencies**
  - `smina` executable accessible via `CONFIG['smina_exe']`
  - RDKit (SMILES parsing + 3D embedding)
  - Meeko (PDBQT preparation)

- **Key configuration parameters**
  - Docking: `exhaustiveness`, `num_modes`, `cpu_cores`, `box_padding` (currently unused)
  - Paths: `structures_dir`, `input_csv`, `output_csv`
  - PDB mapping: `pdb_map`

---

# Outputs

- **Docking results CSV** (`CONFIG['output_csv']`, default: `final_smina.csv`)  
  Contains original ligand rows plus generated docking score columns:
  - `{target_key}_{pocket_id}_Smina` (one column per pocket)

- **Temporary files** (deleted at end)
  - `temp_docking/<ligand>.pdbqt`
  - `temp_docking/<ligand>_docked.pdbqt`

- **Console output**
  - loaded candidate count,
  - skipped targets if PDB missing,
  - per-target and per-pocket progress via tqdm.

---

# Command

```bash
python src/final_docking/final_docking.py
```

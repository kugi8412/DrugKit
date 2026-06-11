# Finalize Selectivity

This script finalizes docking results by selecting the best docking pose per ligand-target pair, reshaping the data into a wide per-ligand format, computing ligand selectivity toward a chosen focus protein, and exporting both the full results and a ranked top-10 subset. The key output metric is **Selectivity**, which measures preference for the focus protein over all other proteins.

Selectivity is computed per (Name, SMILES) as the difference between the best (lowest) binding energy among all non-focus proteins and the worst (highest) binding energy among all conformations of the focus protein, formally defined as Selectivity = min(E_others) - max(E_focus), if any focus-protein conformation defined in the configuration is missing, the selectivity is set to NaN, while the absence of competitors results in a selectivity of 0.0. 

For example, if a ligand binds to the focus protein with energies -9.2 and -8.5 kcal/mol (worst focus energy = -8.5) and to other proteins with energies -7.1 and -6.8 kcal/mol (best competitor energy = -7.1), then the selectivity equals -7.1 − (-8.5) = 1.4, indicating preferential binding to the focus protein.

---

## Process Flow

1. Load configuration from `config.yaml`
2. Resolve input/output paths and focus protein
3. Load docking energies CSV
4. Validate and normalize input columns and types
5. Select the lowest-energy row per `(Name, SMILES, Target)`
6. Pivot target-level data into wide ligand-level format
7. Compute selectivity per ligand
8. Merge wide data with selectivity values
9. Save:

   * full wide results CSV
   * top-10 ligands ranked by selectivity

---

### Structures

Used to define mappings:

* `conformation_key` -> `protein_key`

These mappings are required to group targets by protein during selectivity calculation.

### Relevant Configuration Keys

* `raw.docking.results_energies_file`
  Input docking results CSV
  Default: `output/docking_energies_etc.csv`

* `raw.docking.results_etc_wide_file`
  Output wide-format CSV
  Default: `output/docking_energies_etc_wide.csv`

* `raw.project.final_results`
  Output CSV for top-10 ligands
  Default: `<project.output_dir>/final_results.csv`

* `raw.docking.focus_protein`
  Protein used for selectivity computation
  Default: `SLC6A20`

If paths are missing, defaults under `output/` are used automatically.

---

## Input Data

### Input File

CSV file containing docking results
(Default: `output/docking_energies_etc.csv`)

### Required Columns

* `Name`
* `SMILES`
* `Target`
* `Pocket_ID`
* `Energy`
* `Beat_Baseline`
* `Baseline_Value`

### Input Rules

* `Energy` and `Baseline_Value` are converted to numeric values
* `Beat_Baseline` is parsed as boolean
* Multiple rows per `(Name, SMILES, Target)` are allowed

---

## Output Data

### 1. Wide Results CSV

Path: `results_etc_wide_file`

One row per `(Name, SMILES)` with dynamically generated columns:

* `<Target>_Pocket_ID`
* `<Target>_Energy`
* `<Target>_Baseline_Value`
* `<Target>_Beat`
* `Selectivity`

---

### 2. Final Results (Top 10)

Path: `final_results.csv`

Contains up to 10 ligands with the highest `Selectivity`, sorted in descending order.

Rows with missing selectivity are excluded.

---

## Execution

### Command

```bash
python src/summarizer/finalize_selectivity.py
```
## Stage 2 - Structure preparation

This step fetches or reads raw PDB structures and produces clean, 
docking-ready receptor files (`.pdb` and `.pdbqt`). 
It also optionally flags non-trivial topology (knots) 
via AlphaKnot and highlights target chains by encoding 
them in the B-factor field (useful for quick visualization in PyMOL).


---

### How it works

For each entry under `structures:` in `config.yaml`:

1. **Resolve input structure**

   * Uses `pdb_path` if provided and exists
   * Otherwise downloads from PDB using `pdb_id`

2. **Parse structure** (BioPython `PDBParser`)

3. **Annotate B-factors to mark chains of interest**

   * Target chains (from config) -> B-factor = `100.0`
   * Other chains -> B-factor = `0.0`
   * This does not change coordinates - only metadata.

4. **Clean the structure**

   * Removes waters, except structural waters detected as bridging waters
   * Removes hetero residues, except a whitelist of important ions

     * kept ions: `NA/SOD, CL/CLA, ZN, MG, CA, MN, K`

5. **Export cleaned PDB**

   * Writes `<DATA_DIR>/<name>.pdb`

6. **Convert to PDBQT using OpenBabel**

   * Produces `<DATA_DIR>/<name>.pdbqt`
   * Adds polar hydrogens and prepares the receptor for docking.

7. **Optional topology check**

   * If `uniprot_id` is provided, queries AlphaKnot website and logs whether the protein is reported as knotted / unknotted / artifact.

---

### Inputs

#### 1) config.yaml fields used

```yaml
project:
  data_dir: "Data"

structures:
  SIT1_8I91:
    pdb_id: "8I91"          # optional if pdb_path is provided
    pdb_path: null          # optional local path to .pdb or .pdb.gz
    uniprot_id: "Q9NP91"    # optional; enables AlphaKnot check
    hydrogen_cutoff: 3.5    # used for bridging-water detection
    chain: ["A", "C"]       # string or list; chains to highlight via B-factor
```

Notes:

* `chain` can be a single string (`"A"`) or a list (`["A","B"]`). If missing/empty, B-factor annotation is skipped.
* `hydrogen_cutoff` must be numeric (string is OK, the script casts to float).

#### 2) Local PDB inputs (optional)

* `pdb_path` can point to `.pdb` or `.pdb.gz`.

---

### Outputs

All outputs go into `project.data_dir` (e.g., `Data/`):

* `Data/<name>.pdb` - cleaned receptor structure
* `Data/<name>.pdbqt` - docking-ready receptor (OpenBabel)

Logs:

* `logs/structure_prep.log` (rotating file, debug-level)

---

### How bridging waters are detected

The script keeps only waters that likely stabilize 
the protein by contacting at least two different residues:

* Only water oxygen atoms are considered (`element == "O"`)
* A water is kept if within `hydrogen_cutoff` (default used in config) to protein atoms belonging to >= 2 distinct residues
* This is a geometric heuristic (not full H-bond physics).

---

### Run

From the project root (where `config.yaml` exists):

```bash
python src/structure_rep/structure_rep_r.py
```

---
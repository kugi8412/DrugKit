## Stage 3 - Pocket generation (pocket_logic.py + predictors)

This stage builds **docking grids/pockets** (center + box size) for each prepared structure and each selected chain. It merges three sources of pocket definitions:

1. **Expert pockets** (from a curated CSV of residues) - **pockets.csv**
2. **GeneoNet pockets** (CSV predictions, only if the PDB ID is an official RCSB entry)
3. **P2Rank pockets** (predicted pockets, run locally or loaded from cached results)

Final output is a single JSON file with pockets grouped by structure key.

---

### How it works

For each structure configured in `config.yaml`:

1. **Load structure PDB** from `project.data_dir` (expects Stage 2 output) 
2. **Load global predictors**

   * P2Rank: run or load cached predictions CSV 
   * GeneoNet: locate a matching CSV and parse it (optional) 
3. For each chain listed in the structure config:

   * Add **expert pockets** (residue list -> coords -> centered box)
   * Add **GeneoNet pockets** (filtered by chain proximity + dedup + quality gate)
   * Add **P2Rank pockets** (dedup + per-pocket box from residue coords when available)
4. Save all pockets to the JSON grids file 

---

### Inputs

#### 1) Cleaned structures from Stage 2

This stage expects the PDBs already exist in `project.data_dir` and will fail if not found. 

#### 2) config.yaml sections used

You need at least:

* `project.data_dir`
* `pocket_analysis` (paths + parameters)
* `structures` (per-structure chain list + IDs)

Key parameters used by pocket logic:

* `pocket_analysis.p2rank_path` - executable name/path for P2Rank (optional but recommended)
* `pocket_analysis.geneonet_path` - directory containing GeneoNet CSVs (optional)
* `pocket_analysis.pockets_csv` - expert pocket residue definitions CSV
* `pocket_analysis.buffer_size` - extra padding added to pocket box sizes
* `pocket_analysis.overlap` - distance threshold used for deduplication (Angstrom)
* `pocket_analysis.p2rank_top_n` - max number of pockets to take from P2Rank per chain
* `pocket_analysis.geneonet_top_n` - max number of pockets to take from GeneoNet per chain
* `pocket_analysis.grids_file` - output JSON path 

Structure config fields used:

* `conformation_key` (used as output key)
* `pdb_id` / `pdb_id_lower` (used for GeneoNet enablement and search)
* `chains` (which chains to generate pockets for)
* `conformation` (used to match expert pockets by "Conformation")

---

### Outputs

A JSON file (path from config: `pocket_analysis.grids_file`) with structure-level keys and lists of pockets:

```json
{
  "SIT1_8I91": [
    {
      "id": "A_expert_Orthosteric_Pro_Occluded",
      "center": [12.345, -1.234, 56.789],
      "size": [24.0, 24.0, 24.0],
      "source": "Expert",
      "validation": {"status": "High Confidence", "distance": 3.42}
    },
    {
      "id": "A_p2rank_r1",
      "center": [...],
      "size": [...],
      "source": "P2Rank",
      "validation": {"status": "High (Source)", "dist": 0.0}
    }
  ]
}
```

Each pocket always includes:

* `id` - unique pocket identifier (chain + source + rank/name)
* `center` - docking box center `[x,y,z]`
* `size` - docking box size `[sx,sy,sz]` (in Angstroms)
* `source` - `Expert` / `GeneoNet` / `P2Rank`
* `validation` - confidence estimated by distance to nearest P2Rank pocket (details below)



---

### Pocket sources in detail

#### A) Expert pockets

* Loaded from `pocket_analysis.pockets_csv` 
* Filter rule: only rows where `Conformation == s_cfg.conformation` and `Protein` == `s_cfg.protein`
* Residues are parsed from the `Residue` column by extracting residue numbers with regex
* Pocket center and size are computed from CA atoms of those residues:

  * center = mean of coordinates
  * size = based on max deviation from center, plus `buffer_size`
  * if no coords are found -> pocket is skipped

#### B) GeneoNet pockets (optional)

Enabled only when amatching GeneoNet CSV exists in `pocket_analysis.geneonet_path`, AND the `pdb_id` is confirmed as an official RCSB entry via `data.rcsb.org` check 

Each GeneoNet pocket is filtered by:

* **Chain proximity**: pocket center must be within `buffer_size * 2` of any CA atom of the target chain
* **Deduplication**: skipped if closer than `overlap` to an already-added pocket
* **Quality gate**: must not be "Low Confidence" relative to P2Rank pockets (distance-based)

Box size computed from GeneoNet radius: `side = min(radius*margin + buffer, 30*margin)`

#### C) P2Rank pockets

* If predictions CSV does not exist, it tries to run P2Rank:
  * `p2rank_exec predict -f <pdb> -o <output_dir>` 
* Otherwise it loads cached `<pdb_name>_predictions.csv`
* For each pocket:
  * if residue IDs are available -> compute center/size from residue coords
  * else fallback to P2Rank-reported center and a default 30A cube
* P2Rank pockets are always labeled as:
  * `validation: {"status": "High (Source)", "dist": 0.0}` 

---

### Validation logic (confidence score)

`validate_quality()` assigns confidence by **distance from a pocket center to the nearest P2Rank pocket center**: 

* `< 5.0 A` -> High Confidence
* `< 10.0 A` -> Medium Confidence
* otherwise -> Low Confidence

This is used to annotate expert pockets and filter GeneoNet pockets (drops Low Confidence)

---

### Deduplication rule

Any new pocket is rejected if its center is within `overlap` Angstrom of an already added pocket for that structure+chain. 

---

### Run

From repo root (where `config.yaml` exists):

```bash
python src/pocket_logic/pocket_logic_r.py
```

---
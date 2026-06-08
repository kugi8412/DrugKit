# Stage 1: Ligand Expansion

## Purpose
Expand seed ligands by searching chemical space (SmallWorld), filtering drug-like candidates, and selecting diverse representatives via clustering.

## Files
- `expand_ligands.py` - main execution script  
- `config.yaml` - configuration (paths, filters, clustering, API)
- `logs/expand_ligands.log` - execution logs

## Workflow

1. Load seed ligands (SMILES) from input CSV and initialize SmallWorld client.
2. Query SmallWorld for analogs, deduplicate, and apply physicochemical filters.
3. Cluster valid candidates and export cluster representatives and full cluster mapping.

## Inputs

### Ligands
Seed ligands from CSV (`SMILES` required) - **known_compounds.csv**

## Outputs
### Tables
- `candidates.csv` - cluster representatives  
- `cluster_members.csv` - all clustered molecules

### Logs
- Console (INFO)
- `logs/expand_ligands.log` (DEBUG, rotating)

## Configuration
Defined in `config.yaml`:
- input/output paths
- SmallWorld query limits and delays
- Lipinski and Veber rules
  - lipinski_strict: True
  - veber_tpsa_cutoff: 140.0
  - veber_rotatable_cutoff: 10
- custom MW / logP filters
  - min_mw: 80 
  - max_mw: 650
  - min_logp: -5.0
  - max_logp: 6.5
- clustering cutoff: - 0.5

## Algorithms
### Preparation
SMILES parsing, descriptor calculation, fingerprint generation (Morgan).
SmallWorld analog search with retries and filtering.

### Scoring / Thresholds
- Lipinski (strict or relaxed)
- Veber (TPSA, rotatable bonds)
- Tanimoto distance for clustering (Butina)

## Execution
```bash
python etc/expand_ligands/expand_ligands.py

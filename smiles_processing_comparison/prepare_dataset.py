#!/usr/bin/env python3
# comparison/prepare_dataset.py
"""
Download and prepare the ESOL (Delaney) dataset for the encoder comparison.

Source: deepchem/datasets/delaney-processed.csv on GitHub
Size:   1128 molecules, ~96 KB — deliberately small.
Format: CSV with a 'smiles' column.

Run once to create comparison/esol_filtered.csv.  Subsequent test runs
read that file rather than re-downloading.

Filtering applied:
  1. Both RDKit and our parser must succeed (strict=True).
  2. Atom count must match (sanity check on graph topology).
  3. Bond count must match.

Molecules failing any filter are written to esol_rejected.csv with a reason.
"""
from __future__ import annotations

import csv
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent

DATASET_URL = (
    "https://raw.githubusercontent.com/deepchem/deepchem"
    "/master/datasets/delaney-processed.csv"
)
RAW_CSV     = HERE / "esol_raw.csv"
FILTERED_CSV = HERE / "esol_filtered.csv"
REJECTED_CSV = HERE / "esol_rejected.csv"


def download(url: str, dest: Path) -> None:
    print(f"Downloading {url} -> {dest}")
    urllib.request.urlretrieve(url, dest)
    print(f"  {dest.stat().st_size:,} bytes")


def prepare(force: bool = False) -> Path:
    """Download and filter the dataset.  Returns path to filtered CSV."""
    if not RAW_CSV.exists() or force:
        download(DATASET_URL, RAW_CSV)

    if FILTERED_CSV.exists() and not force:
        with open(FILTERED_CSV) as f:
            n = sum(1 for _ in csv.DictReader(f))
        print(f"Using existing filtered dataset: {FILTERED_CSV} ({n} molecules)")
        return FILTERED_CSV

    # Lazy imports — only needed when actually building the filtered set
    from rdkit import Chem
    sys.path.insert(0, str(HERE.parent))
    from smiles_processing.smiles_parser import parse_smiles

    accepted = []
    rejected = []

    with open(RAW_CSV) as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        smi = row["smiles"].strip()

        # ── RDKit parse ──────────────────────────────────────────────────
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            rejected.append({**row, "reject_reason": "rdkit_parse_fail"})
            continue

        # ── Our parser (strict) ──────────────────────────────────────────
        try:
            g = parse_smiles(smi, strict=True)
        except Exception as exc:
            rejected.append({**row, "reject_reason": f"our_parse_fail: {exc}"})
            continue

        # ── Topology sanity check ────────────────────────────────────────
        if mol.GetNumAtoms() != len(g["atoms"]):
            rejected.append({**row, "reject_reason": "atom_count_mismatch"})
            continue
        if mol.GetNumBonds() != len(g["bonds"]):
            rejected.append({**row, "reject_reason": "bond_count_mismatch"})
            continue

        accepted.append(row)

    # Write outputs
    if rows:
        fieldnames = list(rows[0].keys())
        with open(FILTERED_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(accepted)

        reject_fields = fieldnames + ["reject_reason"]
        with open(REJECTED_CSV, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=reject_fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(rejected)

    print(f"\nDataset prepared:")
    print(f"  Accepted : {len(accepted):4d}  -> {FILTERED_CSV}")
    print(f"  Rejected : {len(rejected):4d}  -> {REJECTED_CSV}")
    return FILTERED_CSV


if __name__ == "__main__":
    prepare(force="--force" in sys.argv)

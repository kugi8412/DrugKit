#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Prepare a real docking experiment with HIV-1 Protease (PDB: 1HSG).

This script downloads the receptor, extracts the binding pocket from the
co-crystallized ligand, prepares known inhibitors as seeds, and builds a
pool of drug-like compounds from the ESOL dataset for testing.

Usage:
    python scripts/prepare_experiment.py

Output:
    data/HIVPRO_1HSG.pdb          - cleaned receptor
    data/HIVPRO_1HSG.pdbqt        - receptor in PDBQT format (if obabel available)
    data/known_compounds.csv      - known HIV protease inhibitors
    data/seed_ligands.csv         - seed ligands for active learning
    data/pool.csv                 - candidate pool for screening
    data/docking_grids.json       - pocket definitions for docking
"""

import csv
import json
import os
import ssl
import subprocess
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data"
ESOL = REPO / "testing_data" / "esol_filtered.csv"

PDB_ID = "1HSG"
TARGET_KEY = "HIVPRO_1HSG"

# Binding pocket center (from co-crystallized ligand MK-639 in 1HSG)
# Determined from HETATM coordinates of the MK-639 inhibitor
POCKET_CENTER = [3.5, 12.0, 8.5]
POCKET_SIZE = [22.0, 22.0, 22.0]

# Known HIV-1 protease inhibitors (FDA-approved drugs) as SMILES
KNOWN_INHIBITORS = [
    ("Indinavir", "CC(C)(C)NC(=O)C1CC2CCCCC2CN1CC(O)CC(Cc1ccccc1)C(=O)NC1c2ccccc2CC1O"),
    ("Ritonavir", "CC(C)c1nc(cs1)CN(C)C(=O)NC(C(=O)NC(CC1CCCCC1)C(O)CN1CC2CCCCC2CC1=O)C(C)C"),
    ("Saquinavir", "CC(C)(C)NC(=O)C1CC2CCCCC2CN1CC(O)C(Cc1ccccc1)NC(=O)C(CC(N)=O)NC(=O)c1ccc2ccccc2n1"),
    ("Nelfinavir", "Oc1ccc(cc1)C1CC(=O)N(C1CC(O)C(Cc1ccccc1)NC(=O)C(C)c1ccccn1)CSCC(=O)N"),
    ("Lopinavir", "CC(C)c1nc(cs1)CN(C)C(=O)NC(C(=O)NC(Cc1ccccc1)CC(O)C(Cc1ccccc1)NC(=O)COc1c(C)cccc1C)C(C)C"),
]


def download_pdb(pdb_id: str, dest: str) -> None:
    """Download PDB from RCSB."""
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  [exists] {dest}")
        return

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    print(f"  Downloading {url}...")

    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    try:
        subprocess.run(
            ["curl", "-fsSL", "-o", dest, url],
            check=True, capture_output=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        urllib.request.urlretrieve(url, dest)

    print(f"  Downloaded: {dest} ({os.path.getsize(dest)} bytes)")


def clean_receptor(pdb_path: str, output_path: str) -> None:
    """Strip water, ligands, and alternate conformations from PDB."""
    with open(pdb_path) as f:
        lines = f.readlines()

    protein_lines = []
    for line in lines:
        record = line[:6].strip()
        if record == "ATOM":
            alt_loc = line[16]
            if alt_loc in (" ", "A"):
                protein_lines.append(line)
        elif record in ("TER", "END"):
            protein_lines.append(line)

    with open(output_path, "w") as f:
        f.writelines(protein_lines)
    print(f"  Cleaned receptor: {output_path} ({len(protein_lines)} lines)")


def convert_to_pdbqt(pdb_path: str, pdbqt_path: str) -> bool:
    """Convert PDB to PDBQT using Open Babel (if available)."""
    try:
        subprocess.run(
            ["obabel", pdb_path, "-O", pdbqt_path, "-xr"],
            check=True, capture_output=True
        )
        print(f"  Converted to PDBQT: {pdbqt_path}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  [skip] obabel not found — PDBQT conversion skipped")
        print("         Install Open Babel: conda install -c conda-forge openbabel")
        return False


def build_pool_from_esol(esol_path: str, output_path: str, max_compounds: int = 100) -> int:
    """Build a screening pool from the ESOL dataset (no external deps)."""
    import csv as _csv

    with open(esol_path, newline="", encoding="utf-8") as f:
        reader = _csv.DictReader(f)
        rows = []
        for row in reader:
            smi = row.get("smiles", row.get("SMILES", "")).strip()
            name = row.get("Compound ID", row.get("Name", f"mol_{len(rows)}")).strip()
            if not smi:
                continue
            rows.append({"Name": name, "SMILES": smi})
            if len(rows) >= max_compounds:
                break

    with open(output_path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=["Name", "SMILES"])
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main():
    print("=" * 60)
    print("  DrugKit Experiment Setup: HIV-1 Protease (1HSG)")
    print("=" * 60)

    DATA.mkdir(parents=True, exist_ok=True)

    # 1. Download receptor
    print("\n[1/5] Downloading receptor...")
    raw_pdb = str(DATA / f"{PDB_ID}.pdb")
    download_pdb(PDB_ID, raw_pdb)

    # 2. Clean receptor
    print("\n[2/5] Cleaning receptor...")
    clean_pdb = str(DATA / f"{TARGET_KEY}.pdb")
    clean_receptor(raw_pdb, clean_pdb)

    # Try PDBQT conversion
    pdbqt_path = str(DATA / f"{TARGET_KEY}.pdbqt")
    convert_to_pdbqt(clean_pdb, pdbqt_path)

    # 3. Create known inhibitors / seed ligands
    print("\n[3/5] Writing known inhibitors...")
    with open(DATA / "known_compounds.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "SMILES", "Target"])
        for name, smi in KNOWN_INHIBITORS:
            writer.writerow([name, smi, TARGET_KEY])
    print(f"  Known compounds: {len(KNOWN_INHIBITORS)} → data/known_compounds.csv")

    with open(DATA / "seed_ligands.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "SMILES", "Target", "Role"])
        for name, smi in KNOWN_INHIBITORS:
            writer.writerow([name, smi, TARGET_KEY, "seed"])
    print(f"  Seed ligands: {len(KNOWN_INHIBITORS)} → data/seed_ligands.csv")

    # 4. Build screening pool
    print("\n[4/5] Building screening pool from ESOL dataset...")
    n = build_pool_from_esol(str(ESOL), str(DATA / "pool.csv"), max_compounds=100)
    print(f"  Pool: {n} compounds → data/pool.csv")

    # 5. Create docking grids
    print("\n[5/5] Writing docking grids...")
    grids = {
        TARGET_KEY: [
            {
                "id": "active_site",
                "center": POCKET_CENTER,
                "size": POCKET_SIZE,
            }
        ]
    }
    with open(DATA / "docking_grids.json", "w") as f:
        json.dump(grids, f, indent=2)
    print(f"  Grids: {DATA / 'docking_grids.json'}")
    print(f"  Pocket center: {POCKET_CENTER}")
    print(f"  Box size: {POCKET_SIZE}")

    # Summary
    print("\n" + "=" * 60)
    print("  Experiment ready!")
    print("=" * 60)
    print(f"""
Files created:
  data/{TARGET_KEY}.pdb           - cleaned receptor
  data/known_compounds.csv        - 5 known HIV protease inhibitors
  data/seed_ligands.csv           - seeds for active learning
  data/pool.csv                   - 100 compounds for screening
  data/docking_grids.json         - binding pocket definition

Next steps:
  1. Install smina: conda install -c conda-forge smina
  2. Convert receptor: obabel data/{TARGET_KEY}.pdb -O data/{TARGET_KEY}.pdbqt -xr
  3. Run docking:
     drugkit-dock --receptor data/{TARGET_KEY}.pdbqt --ligands data/pool.csv \\
         --center {POCKET_CENTER[0]},{POCKET_CENTER[1]},{POCKET_CENTER[2]} --n-cpu 4
  4. Or run full pipeline:
     drugkit-active-learn --config config/drugkit.yaml
""")


if __name__ == "__main__":
    main()

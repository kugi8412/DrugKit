#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build docking ligand CSVs from testing_data/esol_filtered.csv."""

from pathlib import Path

import pandas as pd
from rdkit import Chem

REPO = Path(__file__).resolve().parents[1]
ESOL = REPO / "testing_data" / "esol_filtered.csv"
DATA = REPO / "data"
TARGET = "HIVPRO_1HSG"


def valid_smiles(smiles: str) -> bool:
    return Chem.MolFromSmiles(smiles) is not None


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ESOL)
    smi_col = "smiles" if "smiles" in df.columns else "SMILES"
    name_col = "Compound ID" if "Compound ID" in df.columns else "Name"

    rows = []
    for _, row in df.iterrows():
        smi = str(row[smi_col]).strip()
        if not valid_smiles(smi):
            continue
        rows.append(
            {
                "Name": str(row[name_col]).strip(),
                "SMILES": smi,
                "Target": TARGET,
            }
        )
        if len(rows) >= 40:
            break

    if len(rows) < 5:
        raise SystemExit("Not enough valid SMILES in ESOL subset.")

    known = pd.DataFrame(rows[:3])
    seeds = pd.DataFrame(rows[:3])
    seeds["Role"] = "seed"
    candidates = pd.DataFrame(rows[3:18])
    candidates["Cluster_ID"] = [i // 5 for i in range(len(candidates))]

    known.to_csv(DATA / "known_compounds.csv", index=False)
    seeds.to_csv(DATA / "seed_ligands.csv", index=False)
    candidates.to_csv(DATA / "candidates.csv", index=False)
    print(f"Wrote {len(known)} known, {len(seeds)} seeds, {len(candidates)} candidates.")


if __name__ == "__main__":
    main()

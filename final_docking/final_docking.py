#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# final_docking/final_docking.py

import json
import os
import shutil

import numpy as np
import pandas as pd
from tqdm import tqdm

from docking_common.ligands import prepare_ligand
from docking_smina.smina_engine import run_smina_scoring


CONFIG = {
    "input_csv": "final_candidates_mixed_v2.csv",
    "output_csv": "final_smina.csv",
    "structures_dir": ".",
    "smina_exe": "smina",
    "box_padding": 0.0,
    "exhaustiveness": 16,
    "num_modes": 1,
    "cpu_cores": 4,
    "pdb_map": {
        "8I91": "8I91.pdb",
        "8I92": "8I92.pdb",
        "8WBY": "8WBY.pdb",
        "8WM3": "8WM3.pdb",
        "SIT1_MODEL_00": "SIT1_Model_OO.pdb",
        "SLC6A20": "8I91.pdb",
        "SLC6A19": "8I92.pdb",
    },
}


def find_file_recursive(name, search_path="."):
    for root, _, files in os.walk(search_path):
        if name in files:
            return os.path.join(root, name)
    return None


def load_all_grids():
    grids = {}
    json_files = [f for f in os.listdir(".") if f.endswith(".json")]
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
                grids.update(data)
        except Exception:
            pass
    return grids


def main():
    print("=== FINAL SMINA DOCKING ===")

    if not os.path.exists(CONFIG["input_csv"]):
        print(f"Error: {CONFIG['input_csv']} not found.")
        return

    df = pd.read_csv(CONFIG["input_csv"])
    print(f"Loaded {len(df)} candidates.")

    grids_data = load_all_grids()
    if not grids_data:
        print("Error: No docking grids (JSON) found.")
        return

    temp_dir = "temp_docking"
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)

    for target_key, pockets in grids_data.items():
        pdb_name = CONFIG["pdb_map"].get(target_key, f"{target_key}.pdb")
        pdb_path = find_file_recursive(pdb_name, CONFIG["structures_dir"])

        if not pdb_path:
            print(f"[SKIP] Structure not found for: {target_key}")
            continue

        print(f"\n<== Processing Target: {target_key} ({pdb_path}) ==>")

        for pocket in pockets:
            pocket_id = pocket["id"]
            print(f" > Pocket: {pocket_id}")

            col_name = f"{target_key}_{pocket_id}_Smina"
            if col_name not in df.columns:
                df[col_name] = np.nan

            cx, cy, cz = pocket["center"]
            sx, sy, sz = pocket["size"]

            for idx, row in tqdm(df.iterrows(), total=len(df), leave=False):
                if pd.notnull(row[col_name]):
                    continue

                name = str(row["Name"]).replace(" ", "_")
                smiles = row["SMILES"]

                lig_pdbqt = os.path.join(temp_dir, f"{name}.pdbqt")
                if not os.path.exists(lig_pdbqt):
                    prep = prepare_ligand(smiles, name)
                    if not prep:
                        continue
                    with open(lig_pdbqt, "w", encoding="utf-8") as f:
                        f.write(prep[0])

                energy, _ = run_smina_scoring(
                    receptor_path=pdb_path,
                    pdbqt_ligand=open(lig_pdbqt, encoding="utf-8").read(),
                    center=[cx, cy, cz],
                    size=[sx, sy, sz],
                    base_exhaustiveness=CONFIG["exhaustiveness"],
                    smina_exe=CONFIG["smina_exe"],
                    num_modes=CONFIG["num_modes"],
                    cpu=CONFIG["cpu_cores"],
                )
                df.at[idx, col_name] = energy

            df.to_csv(CONFIG["output_csv"], index=False, float_format="%.3f")

    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    print(f"\nDone. Results saved to {CONFIG['output_csv']}")


if __name__ == "__main__":
    main()

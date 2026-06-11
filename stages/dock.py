#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 2: Molecular Docking run docking against receptor grids.

Usage:
    python -m stages.dock --receptor receptor.pdbqt --ligands pool.csv --output results.csv
    python -m stages.dock --receptor receptor.pdbqt --ligands pool.csv --engine vina
    python -m stages.dock --grids grids.json --ligands pool.csv --exhaustiveness 16

Parameters:
    --receptor      Path to receptor .pdbqt file
    --ligands       CSV file with SMILES column (or directory of .pdbqt files)
    --output        Output CSV with docking scores (default: output/docking_results.csv)
    --engine        "smina" or "vina" (default: smina)
    --grids         Path to JSON grid definitions (alternative to receptor)
    --center        Binding site center x,y,z (e.g. "12.5,3.2,7.8")
    --box-size      Search box dimensions x,y,z (default: "20,20,20")
    --exhaustiveness Docking thoroughness (default: 8)
    --n-cpu         Number of CPU cores (default: 4)
    --n-poses       Number of poses to generate (default: 1)
    --seed          Random seed for reproducibility
    --smina-exe     Path to smina executable (default: "smina")
"""


import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


def run_dock(
    receptor: Optional[str] = None,
    ligands: Optional[str] = None,
    output_file: str = "output/docking_results.csv",
    engine: str = "smina",
    grids_json: Optional[str] = None,
    center: Optional[Tuple[float, float, float]] = None,
    box_size: Tuple[float, float, float] = (20.0, 20.0, 20.0),
    exhaustiveness: int = 8,
    n_cpu: int = 4,
    n_poses: int = 1,
    smina_exe: str = "smina",
) -> str:
    """Run docking against a receptor.

    Args:
        receptor: Path to receptor .pdbqt file.
        ligands: Path to CSV with SMILES or directory of .pdbqt files.
        output_file: Path to output results CSV.
        engine: "smina" or "vina".
        grids_json: JSON file with grid definitions (maps target names to pockets).
        center: (x, y, z) center of search box (used with single --receptor).
        box_size: (sx, sy, sz) dimensions of search box.
        exhaustiveness: Docking search thoroughness.
        n_cpu: Number of parallel CPU cores.
        n_poses: Number of poses per ligand.
        smina_exe: Path to smina binary.

    Returns:
        Path to results CSV file.
    """
    import pandas as pd
    from docking_common.ligands import prepare_ligand

    # Validate inputs
    if receptor and not os.path.exists(receptor):
        raise FileNotFoundError(f"Receptor not found: {receptor}")
    if ligands and not os.path.exists(ligands):
        raise FileNotFoundError(f"Ligands not found: {ligands}")
    if grids_json and not os.path.exists(grids_json):
        raise FileNotFoundError(f"Grids JSON not found: {grids_json}")
    if not receptor and not grids_json:
        raise ValueError("Provide either --receptor or --grids.")

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)

    # Build grids dict: {target_name: [{id, center, size}]}
    if grids_json:
        with open(grids_json) as f:
            grids = json.load(f)
        # Expect format: {"TARGET": [{"id":..., "center":[x,y,z], "size":[sx,sy,sz]}]}
    else:
        if center is None:
            raise ValueError(
                "Binding site center not specified. "
                "Use --center x,y,z or provide --grids JSON."
            )
        target_name = Path(receptor).stem
        grids = {
            target_name: [{
                "id": "pocket_1",
                "center": list(center),
                "size": list(box_size),
            }]
        }

    # Build receptor map: {target_name: path_to_pdbqt}
    if receptor:
        # Single receptor for all targets
        rec_map = {t: receptor for t in grids}
    else:
        # Expect receptors in same directory as grids file, named <target>.pdbqt
        grids_dir = os.path.dirname(os.path.abspath(grids_json))
        rec_map = {}
        for target_name in grids:
            candidate = os.path.join(grids_dir, f"{target_name}.pdbqt")
            if os.path.exists(candidate):
                rec_map[target_name] = candidate
            else:
                raise FileNotFoundError(
                    f"Receptor for target '{target_name}' not found at {candidate}"
                )

    # Load ligands
    if os.path.isdir(ligands):
        pdbqt_files = sorted(Path(ligands).glob("*.pdbqt"))
        print(f"Found {len(pdbqt_files)} .pdbqt ligand files")

        ligand_records = []
        for pf in pdbqt_files:
            ligand_records.append({
                "Name": pf.stem,
                "SMILES": "",
                "_pdbqt": pf.read_text(),
            })
    else:
        df = pd.read_csv(ligands)
        for col in ("SMILES", "smiles", "Smiles", "canonical_smiles"):
            if col in df.columns:
                smiles_col = col
                break
        else:
            raise ValueError(f"No SMILES column found in {ligands}")

        name_col = None
        for col in ("Name", "name", "ID", "id"):
            if col in df.columns:
                name_col = col
                break

        ligand_records = []
        for i, row in df.iterrows():
            name = row[name_col] if name_col else f"mol_{i}"
            ligand_records.append({"Name": str(name), "SMILES": row[smiles_col]})

    print(f"Engine: {engine} | Targets: {list(grids.keys())}")
    print(f"Exhaustiveness: {exhaustiveness} | CPUs: {n_cpu} | Ligands: {len(ligand_records)}")

    # Dock
    start = time.time()
    results = []

    if engine == "smina":
        from docking_smina.smina_engine import run_smina_scoring

        def _dock_one_smina(record, target, pocket):
            pdbqt_str = record.get("_pdbqt")
            if not pdbqt_str:
                prep = prepare_ligand(record["SMILES"], record["Name"])
                if prep is None:
                    return None
                pdbqt_str = prep[0]

            energy, _ = run_smina_scoring(
                receptor_path=rec_map[target],
                pdbqt_ligand=pdbqt_str,
                center=pocket["center"],
                size=pocket["size"],
                base_exhaustiveness=exhaustiveness,
                smina_exe=smina_exe,
                num_modes=n_poses,
                cpu=1,
            )
            return {
                "Name": record["Name"],
                "SMILES": record.get("SMILES", ""),
                "Target": target,
                "Pocket_ID": pocket["id"],
                "Energy": energy,
            }

        with ProcessPoolExecutor(max_workers=n_cpu) as executor:
            futures = []
            for rec in ligand_records:
                for target, pockets in grids.items():
                    for pocket in pockets:
                        futures.append(
                            executor.submit(_dock_one_smina, rec, target, pocket)
                        )
            for fut in as_completed(futures):
                r = fut.result()
                if r is not None:
                    results.append(r)

    elif engine == "vina":
        from docking_vina.vina_engine import run_vina_scoring

        for rec in ligand_records:
            pdbqt_str = rec.get("_pdbqt")
            if not pdbqt_str:
                prep = prepare_ligand(rec["SMILES"], rec["Name"])
                if prep is None:
                    continue
                pdbqt_str = prep[0]

            for target, pockets in grids.items():
                for pocket in pockets:
                    energy, _ = run_vina_scoring(
                        pdbqt_ligand=pdbqt_str,
                        receptor_path=rec_map[target],
                        center=pocket["center"],
                        size=pocket["size"],
                        base_exhaustiveness=exhaustiveness,
                    )
                    results.append({
                        "Name": rec["Name"],
                        "SMILES": rec.get("SMILES", ""),
                        "Target": target,
                        "Pocket_ID": pocket["id"],
                        "Energy": energy,
                    })
    else:
        raise ValueError(f"Unknown engine: '{engine}'. Use 'smina' or 'vina'.")

    elapsed = time.time() - start
    print(f"Docking completed in {elapsed:.1f}s ({len(results)} results)")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print(f"Saved {len(results_df)} results to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="DrugKit Stage 2: Molecular Docking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m stages.dock --receptor data/receptor.pdbqt --ligands data/pool.csv \\
      --center 12.5,3.2,7.8 --exhaustiveness 16
  python -m stages.dock --grids config/grids.json --ligands data/pool.csv --engine vina
  python -m stages.dock --receptor receptor.pdbqt --ligands ligands_dir/ --n-cpu 8
        """,
    )
    parser.add_argument("--receptor", "-r", help="Receptor .pdbqt file path")
    parser.add_argument("--ligands", "-l", required=True,
                        help="CSV with SMILES or directory of .pdbqt files")
    parser.add_argument("--output", "-o", default="output/docking_results.csv",
                        help="Output results CSV")
    parser.add_argument("--engine", choices=["smina", "vina"], default="smina",
                        help="Docking engine (default: smina)")
    parser.add_argument("--grids", help="JSON file with grid definitions")
    parser.add_argument("--center", help="Binding site center: x,y,z")
    parser.add_argument("--box-size", default="20,20,20",
                        help="Search box dimensions: x,y,z (default: 20,20,20)")
    parser.add_argument("--exhaustiveness", type=int, default=8,
                        help="Docking thoroughness (default: 8)")
    parser.add_argument("--n-cpu", type=int, default=4,
                        help="Number of CPU cores (default: 4)")
    parser.add_argument("--n-poses", type=int, default=1,
                        help="Poses per ligand (default: 1)")
    parser.add_argument("--smina-exe", default="smina",
                        help="Path to smina binary (default: smina)")
    args = parser.parse_args()

    # Parse center
    center = None
    if args.center:
        parts = [float(x) for x in args.center.split(",")]
        if len(parts) != 3:
            parser.error("--center must be x,y,z (3 values)")
        center = tuple(parts)

    # Parse box size
    box_parts = [float(x) for x in args.box_size.split(",")]
    if len(box_parts) != 3:
        parser.error("--box-size must be x,y,z (3 values)")
    box_size = tuple(box_parts)

    run_dock(
        receptor=args.receptor,
        ligands=args.ligands,
        output_file=args.output,
        engine=args.engine,
        grids_json=args.grids,
        center=center,
        box_size=box_size,
        exhaustiveness=args.exhaustiveness,
        n_cpu=args.n_cpu,
        n_poses=args.n_poses,
        smina_exe=args.smina_exe,
    )


if __name__ == "__main__":
    main()

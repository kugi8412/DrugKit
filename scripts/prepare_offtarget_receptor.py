#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Download a PDB structure, strip solvent, and register a docking grid.

Used to add an off-target receptor for selectivity-based active learning.
The binding-pocket grid is derived from the largest non-solvent HETATM
ligand in the structure (its centroid + padded bounding box). If no such
ligand exists, the protein centroid is used with a large default box.

Usage:
    python scripts/prepare_offtarget_receptor.py <PDB_ID> <TARGET_KEY> \
        [--data-dir data] [--grids-file docking_grids.json] [--pad 8.0]
"""

import argparse
import json
import os
import ssl
import subprocess
import sys
import urllib.request
from typing import Dict, List, Tuple

# Residue names that are solvent / ions / common crystallization additives,
# not a real binding-site ligand.
_NON_LIGAND_RESN = {
    "HOH", "WAT", "DOD", "NA", "CL", "K", "MG", "CA", "ZN", "MN", "FE",
    "SO4", "PO4", "GOL", "EDO", "ACT", "DMS", "PEG", "BME", "TRS", "FMT",
    "IOD", "BR", "NO3", "CO3", "NH4", "CD", "NI", "CU", "HG", "MES", "EPE",
}

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"


def download_pdb(pdb_id: str, dest: str) -> None:
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"[INFO] Reusing existing download: {dest}")
        return

    url = RCSB_URL.format(pdb_id=pdb_id.upper())
    print(f"[INFO] Downloading {url}")

    # Prefer curl (handles system CA store); fall back to urllib.
    try:
        subprocess.run(
            ["curl", "-fsSL", "-o", dest, url], check=True, capture_output=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with urllib.request.urlopen(url, context=ctx) as resp, open(dest, "wb") as out:
            out.write(resp.read())

    if not os.path.exists(dest) or os.path.getsize(dest) == 0:
        sys.exit(f"[CRITICAL] Download failed or empty: {dest}")


def _parse_coords(line: str) -> Tuple[float, float, float]:
    return (
        float(line[30:38]),
        float(line[38:46]),
        float(line[46:54]),
    )


def parse_structure(pdb_path: str) -> Tuple[List[str], Dict[str, List[Tuple[float, float, float]]]]:
    """Return protein ATOM lines and a map of ligand-id -> HETATM coords."""
    protein_lines: List[str] = []
    ligands: Dict[str, List[Tuple[float, float, float]]] = {}

    with open(pdb_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            record = line[:6].strip()
            if record == "ATOM":
                protein_lines.append(line)
            elif record == "HETATM":
                resn = line[17:20].strip()
                if resn in _NON_LIGAND_RESN:
                    continue
                chain = line[21:22].strip()
                resseq = line[22:26].strip()
                ligand_id = f"{resn}_{chain}_{resseq}"
                ligands.setdefault(ligand_id, []).append(_parse_coords(line))
            elif record in ("TER", "END"):
                if record == "TER":
                    protein_lines.append(line)

    return protein_lines, ligands


def _centroid(coords: List[Tuple[float, float, float]]) -> List[float]:
    n = len(coords)
    return [sum(c[i] for c in coords) / n for i in range(3)]


def _box_size(coords: List[Tuple[float, float, float]], pad: float) -> List[float]:
    dims = []
    for i in range(3):
        vals = [c[i] for c in coords]
        extent = (max(vals) - min(vals)) + 2.0 * pad
        dims.append(round(min(max(extent, 18.0), 30.0), 3))
    return dims


def compute_grid(
    protein_lines: List[str],
    ligands: Dict[str, List[Tuple[float, float, float]]],
    pad: float,
) -> Tuple[List[float], List[float], str]:
    if ligands:
        ligand_id = max(ligands.items(), key=lambda kv: len(kv[1]))[0]
        coords = ligands[ligand_id]
        center = [round(v, 3) for v in _centroid(coords)]
        size = _box_size(coords, pad)
        print(f"[INFO] Pocket from ligand {ligand_id} ({len(coords)} atoms)")
        return center, size, ligand_id

    coords = [_parse_coords(l) for l in protein_lines if l[:6].strip() == "ATOM"]
    if not coords:
        sys.exit("[CRITICAL] No protein atoms found.")
    center = [round(v, 3) for v in _centroid(coords)]
    print("[WARN] No ligand found; using protein centroid with 26 A box.")
    return center, [26.0, 26.0, 26.0], "protein_centroid"


def write_receptor(protein_lines: List[str], out_path: str) -> None:
    with open(out_path, "w", encoding="utf-8") as f:
        f.writelines(protein_lines)
        f.write("END\n")
    print(f"[INFO] Receptor written: {out_path} ({len(protein_lines)} atom lines)")


def update_grids(
    grids_file: str,
    target_key: str,
    center: List[float],
    size: List[float],
    pdb_id: str,
    source_ligand: str,
) -> None:
    grids: Dict[str, object] = {}
    if os.path.exists(grids_file):
        with open(grids_file, "r", encoding="utf-8") as f:
            grids = json.load(f)

    grids[target_key] = [
        {
            "id": f"{target_key}_pocket_{source_ligand}",
            "center": center,
            "size": size,
            "source": f"RCSB:{pdb_id.upper()}",
            "validation": {"status": "auto_ligand_centroid", "dist": 0.0},
        }
    ]

    with open(grids_file, "w", encoding="utf-8") as f:
        json.dump(grids, f, indent=4)
    print(f"[INFO] Grid registered for '{target_key}' in {grids_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb_id", help="RCSB PDB id, e.g. 2V0Z")
    parser.add_argument("target_key", help="Receptor key, e.g. RENIN_2V0Z")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--grids-file", default="docking_grids.json")
    parser.add_argument("--pad", type=float, default=8.0)
    parser.add_argument("--keep-raw", action="store_true", help="Keep the raw downloaded PDB")
    args = parser.parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    raw_path = os.path.join(args.data_dir, f"{args.pdb_id.upper()}_raw.pdb")
    receptor_path = os.path.join(args.data_dir, f"{args.target_key}.pdb")

    download_pdb(args.pdb_id, raw_path)
    protein_lines, ligands = parse_structure(raw_path)
    if not protein_lines:
        sys.exit("[CRITICAL] No protein ATOM records parsed.")

    center, size, source_ligand = compute_grid(protein_lines, ligands, args.pad)
    write_receptor(protein_lines, receptor_path)
    update_grids(args.grids_file, args.target_key, center, size, args.pdb_id, source_ligand)

    if not args.keep_raw and os.path.exists(raw_path):
        os.remove(raw_path)

    print(f"[DONE] center={center} size={size}")


if __name__ == "__main__":
    main()

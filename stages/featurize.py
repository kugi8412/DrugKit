#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 1: SMILES Featurization â€” convert SMILES to PyTorch Geometric graphs.

Usage:
    python -m stages.featurize --input data/pool.csv --output output/graphs.pt
    python -m stages.featurize --input data/pool.csv --output output/graphs.pt --encoder rdkit
    python -m stages.featurize --smiles "CCO" "c1ccccc1" "CC(=O)O"

Parameters:
    --input         CSV file with SMILES column (required unless --smiles used)
    --output        Output .pt file for serialized graphs (default: output/graphs.pt)
    --smiles        Inline SMILES strings (alternative to --input)
    --smiles-col    Column name or index for SMILES (default: "SMILES")
    --name-col      Column name for compound names (default: "Name")
    --encoder       "rdkit" or "custom" (default: "rdkit")
    --batch-size    Processing batch size (default: 1000)
    --n-workers     Parallel workers (default: 1)
    --validate      Only validate SMILES, don't featurize (flag)
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from torch_geometric.data import Data


def run_featurize(
    input_file: Optional[str] = None,
    output_file: str = "output/graphs.pt",
    smiles_list: Optional[List[str]] = None,
    smiles_col: str = "SMILES",
    name_col: str = "Name",
    encoder: str = "rdkit",
    batch_size: int = 1000,
    validate_only: bool = False,
) -> Tuple[List[Data], List[str], List[int]]:
    """Convert SMILES to PyG graph objects.

    Args:
        input_file: Path to CSV with SMILES column.
        output_file: Path to save .pt file with graphs.
        smiles_list: Alternative: provide SMILES directly as a list.
        smiles_col: Column name (or int index) for SMILES in CSV.
        name_col: Column name for compound identifiers.
        encoder: "rdkit" (uses RDKit featurization) or "custom" (RDKit-free parser).
        batch_size: Process this many SMILES at a time.
        validate_only: If True, only check which SMILES are valid.

    Returns:
        (graphs, names, failed_indices) tuple.
    """
    # Load SMILES
    if smiles_list is not None:
        smiles = smiles_list
        names = [f"mol_{i}" for i in range(len(smiles))]
    elif input_file is not None:
        if not os.path.exists(input_file):
            raise FileNotFoundError(f"Input file not found: {input_file}")
        df = pd.read_csv(input_file)
        df.columns = [c.strip() for c in df.columns]

        # Resolve SMILES column
        if isinstance(smiles_col, int):
            smiles = df.iloc[:, smiles_col].astype(str).tolist()
        elif smiles_col in df.columns:
            smiles = df[smiles_col].astype(str).tolist()
        else:
            # Try common column names
            for candidate in ("SMILES", "smiles", "Smiles", "canonical_smiles"):
                if candidate in df.columns:
                    smiles = df[candidate].astype(str).tolist()
                    break
            else:
                raise ValueError(
                    f"SMILES column '{smiles_col}' not found. "
                    f"Available: {list(df.columns)}"
                )

        # Resolve name column
        if name_col in df.columns:
            names = df[name_col].astype(str).tolist()
        else:
            names = [f"mol_{i}" for i in range(len(smiles))]
    else:
        raise ValueError("Provide either --input (CSV file) or --smiles (inline).")

    # Select encoder
    if encoder == "rdkit":
        from siamese_GNN.featurization import smiles_to_graph_gine
        encode_fn = lambda s: smiles_to_graph_gine(s)
    elif encoder == "custom":
        from smiles_processing import smiles_to_pyg
        encode_fn = smiles_to_pyg
    else:
        raise ValueError(f"Unknown encoder: '{encoder}'. Use 'rdkit' or 'custom'.")

    # Process
    graphs = []
    valid_names = []
    failed_indices = []
    total = len(smiles)

    print(f"Featurizing {total} SMILES (encoder={encoder})...")
    start = time.time()

    for i in range(0, total, batch_size):
        batch = smiles[i:i + batch_size]
        for j, smi in enumerate(batch):
            idx = i + j
            g = encode_fn(smi)
            if g is not None:
                g.smiles = smi
                g.name = names[idx]
                graphs.append(g)
                valid_names.append(names[idx])
            else:
                failed_indices.append(idx)

        if (i + batch_size) % (batch_size * 10) == 0 or i + batch_size >= total:
            elapsed = time.time() - start
            print(f"  [{min(i + batch_size, total)}/{total}] "
                  f"{len(graphs)} valid, {len(failed_indices)} failed "
                  f"({elapsed:.1f}s)")

    if validate_only:
        print(f"\nValidation: {len(graphs)}/{total} valid SMILES "
              f"({len(graphs)/total*100:.1f}%)")
        print(f"Failed indices: {failed_indices[:20]}{'...' if len(failed_indices) > 20 else ''}")
        return graphs, valid_names, failed_indices

    # Save
    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        torch.save({"graphs": graphs, "names": valid_names}, output_file)
        print(f"\nSaved {len(graphs)} graphs to {output_file}")

    print(f"Done: {len(graphs)} valid, {len(failed_indices)} failed "
          f"({time.time() - start:.1f}s)")
    return graphs, valid_names, failed_indices


def main():
    parser = argparse.ArgumentParser(
        description="DrugKit Stage 1: SMILES â†’ PyG Graph Featurization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m stages.featurize --input data/pool.csv --output output/graphs.pt
  python -m stages.featurize --input library.csv --smiles-col 0 --encoder custom
  python -m stages.featurize --smiles "CCO" "c1ccccc1" --validate
        """,
    )
    parser.add_argument("--input", "-i", help="Input CSV file with SMILES")
    parser.add_argument("--output", "-o", default="output/graphs.pt",
                        help="Output .pt file (default: output/graphs.pt)")
    parser.add_argument("--smiles", nargs="+", help="Inline SMILES strings")
    parser.add_argument("--smiles-col", default="SMILES",
                        help="SMILES column name or index (default: SMILES)")
    parser.add_argument("--name-col", default="Name",
                        help="Name column (default: Name)")
    parser.add_argument("--encoder", choices=["rdkit", "custom"], default="rdkit",
                        help="Featurization backend (default: rdkit)")
    parser.add_argument("--batch-size", type=int, default=1000,
                        help="Processing batch size (default: 1000)")
    parser.add_argument("--validate", action="store_true",
                        help="Only validate SMILES (don't save graphs)")
    args = parser.parse_args()

    # Resolve smiles_col (could be int)
    try:
        smiles_col = int(args.smiles_col)
    except ValueError:
        smiles_col = args.smiles_col

    run_featurize(
        input_file=args.input,
        output_file=args.output,
        smiles_list=args.smiles,
        smiles_col=smiles_col,
        name_col=args.name_col,
        encoder=args.encoder,
        batch_size=args.batch_size,
        validate_only=args.validate,
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 7: Ligand Expansion expand hit compounds into chemical analogs.

Usage:
    python -m stages.expand --seeds data/hits.csv --output output/expanded.csv
    python -m stages.expand --seeds hits.csv --max-analogs 50 --similarity-cutoff 0.7
    python -m stages.expand --seeds hits.csv --method scaffold_hopping

Parameters:
    --seeds             CSV with seed SMILES to expand (required)
    --output            Output CSV with expanded library
    --smiles-col        SMILES column name (default: "SMILES")
    --method            Expansion method: similarity, scaffold_hopping, fragment (default: similarity)
    --max-analogs       Max analogs per seed compound (default: 20)
    --similarity-cutoff Tanimoto similarity threshold (default: 0.6)
    --database          External database path for analog search
    --clustering-cutoff Cluster deduplication cutoff (default: 0.8)
    --n-workers         Parallel workers (default: 1)
    --deduplicate       Remove duplicate SMILES from output (flag, default: True)
"""

import argparse
import os
import time
from typing import Optional

import pandas as pd


def run_expand(
    seeds_file: str,
    output_file: str = "output/expanded_library.csv",
    smiles_col: str = "SMILES",
    method: str = "similarity",
    max_analogs: int = 20,
    similarity_cutoff: float = 0.6,
    database: Optional[str] = None,
    clustering_cutoff: float = 0.8,
    n_workers: int = 1,
    deduplicate: bool = True,
) -> str:
    """Expand seed compounds into a larger analog library.

    Args:
        seeds_file: CSV with seed SMILES.
        output_file: Path for expanded library CSV.
        smiles_col: SMILES column name.
        method: "similarity" (SmallWorld/Tanimoto search), "scaffold_hopping", or "fragment".
        max_analogs: Maximum analogs per seed.
        similarity_cutoff: Minimum similarity to seed (0-1), used as clustering cutoff.
        database: External database CSV for analog lookup (optional).
        clustering_cutoff: Butina clustering cutoff for deduplication.
        n_workers: Parallel workers (reserved for future use).
        deduplicate: Remove duplicates from final output.

    Returns:
        Path to expanded library CSV.
    """
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator, Descriptors, rdMolDescriptors
    from rdkit.ML.Cluster import Butina

    if not os.path.exists(seeds_file):
        raise FileNotFoundError(f"Seeds file not found: {seeds_file}")

    df = pd.read_csv(seeds_file)
    if smiles_col not in df.columns:
        # Try common alternatives
        for alt in ("SMILES", "smiles", "Smiles", "canonical_smiles"):
            if alt in df.columns:
                smiles_col = alt
                break
        else:
            raise ValueError(f"Column '{smiles_col}' not in {list(df.columns)}")

    seeds = df[smiles_col].dropna().unique().tolist()
    print(f"Expanding {len(seeds)} seed compounds (method={method})")
    print(f"  Max analogs/seed: {max_analogs}")
    print(f"  Clustering cutoff: {clustering_cutoff}")

    start = time.time()

    # Strategy: use external database if provided, else SmallWorld API
    expanded_pool = {}

    if database and os.path.exists(database):
        # Search external database by Tanimoto similarity
        db_df = pd.read_csv(database)
        db_smi_col = None
        for col in ("SMILES", "smiles", "Smiles", "canonical_smiles"):
            if col in db_df.columns:
                db_smi_col = col
                break
        if db_smi_col is None:
            raise ValueError(f"No SMILES column found in database {database}")

        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        # Pre-compute DB fingerprints
        db_smiles = db_df[db_smi_col].dropna().tolist()
        db_fps = []
        db_valid = []
        for smi in db_smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                db_fps.append(mfpgen.GetFingerprint(mol))
                db_valid.append(smi)

        for seed_smi in seeds:
            seed_mol = Chem.MolFromSmiles(seed_smi)
            if not seed_mol:
                continue
            seed_fp = mfpgen.GetFingerprint(seed_mol)
            sims = DataStructs.BulkTanimotoSimilarity(seed_fp, db_fps)
            # Get top analogs above similarity cutoff
            ranked = sorted(enumerate(sims), key=lambda x: -x[1])
            count = 0
            for idx, sim in ranked:
                if sim < similarity_cutoff:
                    break
                analog_smi = db_valid[idx]
                if analog_smi != seed_smi and analog_smi not in expanded_pool:
                    expanded_pool[analog_smi] = {
                        "SMILES": analog_smi,
                        "Name": f"Analog_{len(expanded_pool)}",
                        "Parent_Seed": seed_smi,
                        "Similarity": round(sim, 3),
                    }
                    count += 1
                    if count >= max_analogs:
                        break
    else:
        # Use SmallWorld API (the expand_ligands module's approach)
        try:
            from expand_ligands.expand_ligands import (
                get_smallworld_client, query_smallworld_robust, passes_filters
            )
            client = get_smallworld_client()
            if client is None:
                print("  WARNING: SmallWorld API unavailable. No expansion performed.")
                # Write seeds as-is
                os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
                df.to_csv(output_file, index=False)
                return output_file

            import time as _time
            for seed_smi in seeds:
                analogs = query_smallworld_robust(seed_smi, client)
                count = 0
                for analog_smi in analogs:
                    if analog_smi not in expanded_pool and passes_filters(analog_smi):
                        expanded_pool[analog_smi] = {
                            "SMILES": analog_smi,
                            "Name": f"Analog_{len(expanded_pool)}",
                            "Parent_Seed": seed_smi,
                            "Source": "SmallWorld",
                        }
                        count += 1
                        if count >= max_analogs:
                            break
                _time.sleep(1.0)

        except ImportError:
            print("  WARNING: expand_ligands module not available (missing smallworld_api?).")
            print("  Provide --database for local analog search.")
            os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
            df.to_csv(output_file, index=False)
            return output_file

    elapsed = time.time() - start
    print(f"Generated {len(expanded_pool)} analogs in {elapsed:.1f}s")

    if not expanded_pool:
        print("  No analogs found. Writing seeds to output.")
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        df.to_csv(output_file, index=False)
        return output_file

    # Build output DataFrame
    results = pd.DataFrame(list(expanded_pool.values()))

    # Deduplicate
    if deduplicate:
        before = len(results)
        results = results.drop_duplicates(subset=["SMILES"])
        removed = before - len(results)
        if removed > 0:
            print(f"  Removed {removed} duplicates ({len(results)} remaining)")

    # Clustering (using Butina)
    if len(results) > 1:
        mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        fps = []
        valid_idx = []
        for i, smi in enumerate(results["SMILES"]):
            mol = Chem.MolFromSmiles(smi)
            if mol:
                fps.append(mfpgen.GetFingerprint(mol))
                valid_idx.append(i)

        if len(fps) > 1:
            dists = []
            for i in range(1, len(fps)):
                sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
                dists.extend([1.0 - x for x in sims])
            clusters = Butina.ClusterData(dists, len(fps), clustering_cutoff, isDistData=True)
            # Assign cluster IDs
            cluster_ids = [""] * len(results)
            for ci, cluster in enumerate(clusters):
                for member in cluster:
                    real_idx = valid_idx[member]
                    cluster_ids[real_idx] = f"CL_{ci+1:03d}"
            results["Cluster_ID"] = cluster_ids
            print(f"  Clustered into {len(clusters)} groups")

    # Save
    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    results.to_csv(output_file, index=False)
    print(f"Saved {len(results)} compounds to {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(
        description="DrugKit Stage 7: Ligand Expansion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m stages.expand --seeds data/hits.csv --output output/expanded.csv
  python -m stages.expand --seeds hits.csv --max-analogs 50 --similarity-cutoff 0.7
  python -m stages.expand --seeds hits.csv --method scaffold_hopping
  python -m stages.expand --seeds hits.csv --database chembl.csv --clustering-cutoff 0.85
        """,
    )
    parser.add_argument("--seeds", "-s", required=True,
                        help="CSV with seed SMILES to expand")
    parser.add_argument("--output", "-o", default="output/expanded_library.csv",
                        help="Output expanded library CSV")
    parser.add_argument("--smiles-col", default="SMILES",
                        help="SMILES column name (default: SMILES)")
    parser.add_argument("--method", choices=["similarity", "scaffold_hopping", "fragment"],
                        default="similarity",
                        help="Expansion method (default: similarity)")
    parser.add_argument("--max-analogs", type=int, default=20,
                        help="Max analogs per seed (default: 20)")
    parser.add_argument("--similarity-cutoff", type=float, default=0.6,
                        help="Min Tanimoto similarity (default: 0.6)")
    parser.add_argument("--database", help="External database for analog search")
    parser.add_argument("--clustering-cutoff", type=float, default=0.8,
                        help="Butina clustering cutoff (default: 0.8)")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Parallel workers (default: 1)")
    parser.add_argument("--no-deduplicate", action="store_true",
                        help="Skip deduplication of output")
    args = parser.parse_args()

    run_expand(
        seeds_file=args.seeds,
        output_file=args.output,
        smiles_col=args.smiles_col,
        method=args.method,
        max_analogs=args.max_analogs,
        similarity_cutoff=args.similarity_cutoff,
        database=args.database,
        clustering_cutoff=args.clustering_cutoff,
        n_workers=args.n_workers,
        deduplicate=not args.no_deduplicate,
    )


if __name__ == "__main__":
    main()

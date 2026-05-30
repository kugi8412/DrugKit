#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# expand_ligands/expand_ligands.py

import os
import sys
import yaml
import time
import warnings

import logging
import logging.handlers

import pandas as pd

from rdkit.ML.Cluster import Butina
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem import Descriptors, rdMolDescriptors

from smallworld_api import SmallWorld
from typing import List, Optional, Tuple, Dict, Any

# Config
CONFIG_PATH = "config.yaml"
if not os.path.exists(CONFIG_PATH):
    sys.exit("[CRITICAL]: config.yaml not found.")

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Logger
os.makedirs("logs", exist_ok=True)

formatter = logging.Formatter(
    '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%dT%H:%M:%S'
)

formatter.converter = time.gmtime
console_h = logging.StreamHandler(sys.stdout)
console_h.setLevel(logging.INFO)
console_h.setFormatter(formatter)

file_h = logging.handlers.RotatingFileHandler(
    filename="logs/expand_ligands.log",
    maxBytes=8 * 1024 * 1024,
    backupCount=5,
    encoding="utf-8"
)

file_h.setLevel(logging.DEBUG)
file_h.setFormatter(formatter)

logging.basicConfig(
    level=logging.DEBUG,
    handlers=[console_h, file_h]
)

logger = logging.getLogger(__name__)
logger.info("Logging configured: console + logs/expand_ligands.log")

# Params
DATA_DIR = config["project"].get("data_dir", "data/")
FILTERS = config["ligand_expansion"].get("filters", {})
API_DELAY = float(config["ligand_expansion"].get("api_delay", 2.0))
CLUSTER_CUTOFF = float(config["ligand_expansion"].get("clustering_cutoff", 0.5))
MAX_ANALOGS_PER_SEED = int(config["ligand_expansion"].get("max_analogs_per_seed", 50))
OUTPUT_MAIN = config["ligand_expansion"].get("output_smiles", "data/candidates.csv")
OUTPUT_CLUSTERS = os.path.join(os.path.dirname(OUTPUT_MAIN), "cluster_members.csv")
LIPINSKI_STRICT = bool(config["ligand_expansion"].get("lipinski_strict", True))
VE_TPSA = float(config["ligand_expansion"].get("veber_tpsa_cutoff", 140.0))
VE_ROTATABLE = int(config["ligand_expansion"].get("veber_rotatable_cutoff", 10))
MFPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)


def get_smallworld_client() -> Optional[SmallWorld]:
    """ Function to query a SmallWorld chemical space search server.
    """
    try:
        return SmallWorld(update_dbs=False)
    except TypeError:
        try:
            return SmallWorld()
        except Exception:
            try:
                return SmallWorld.__new__(SmallWorld)
            except:
                return None
    except Exception:
        return None


def query_smallworld_robust(smiles: str,
                            client: SmallWorld,
                            distance: int = 25,
                            database_1: str = "REAL_dataset",
                            database_2: str = "REALDB-2025-07.smi.anon"
                            ) -> List[str]:
    """ Querying selected databases (database_1 > database_2) with
    fallback and multiple retires with API delay.
    """
    if client is None:
        return []
    else:
        database = getattr(client, database_1, None) or database_2
    
    # Multiple retries (Idea of the D4S1[https://github.com/D4S1] concept)
    for attempt in range(1, 5):
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                results = client.search(smiles, db=database, dist=distance, length=MAX_ANALOGS_PER_SEED)
            
            if results is None:
                raise ValueError("API None")
            
            if isinstance(results, pd.DataFrame):
                if results.empty:
                    return []

                # Most popular databases
                cols = ["smiles", "SMILES", "hitSmiles", "hit_smiles"]
                f_col = next((c for c in cols if c in results.columns), results.columns[0])
                return results[f_col].dropna().astype(str).tolist()
                
            return [str(results)]

        except Exception:
            time.sleep(2.0 * attempt)

    return []


def passes_filters(smiles: str) -> bool:
    """ Function to check if smiles restricted:
    1) Custom limitations
    2) Lipinski's Rule of Five
    3) Veber's rules
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        
        # Custom 
        if "min_mw" in FILTERS and mw < float(FILTERS["min_mw"]):
            return False
        if "max_mw" in FILTERS and mw > float(FILTERS["max_mw"]):
            return False
        if "min_logp" in FILTERS and logp < float(FILTERS["min_logp"]):
            return False
        if "max_logp" in FILTERS and logp > float(FILTERS["max_logp"]):
            return False

        # Lipinski's Rule of Five
        violations = 0
        if mw > 500: violations += 1
        if logp > 5: violations += 1
        if rdMolDescriptors.CalcNumHBD(mol) > 5:
            violations += 1
        if rdMolDescriptors.CalcNumHBA(mol) > 10:
            violations += 1

        if LIPINSKI_STRICT and violations > 0:
            return False
        if not LIPINSKI_STRICT and violations > 1:
            return False

        # Veber's rules
        if rdMolDescriptors.CalcTPSA(mol) > VE_TPSA:
            return False
        if rdMolDescriptors.CalcNumRotatableBonds(mol) > VE_ROTATABLE:
            return False

        return True

    except:
        return False


def cluster_and_map(candidates_data: List[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
    """ Clusters molecules with butin algorithm and returns two lists:
    1) representatives: list of dictionaries (only cluster centroids)
    2) cluster_map: list of dictionaries (all members with assigned Cluster_ID)
    """
    if not candidates_data:
        return [], []

    # Morgan's Fingerprints
    mols, fingerprints, valid_indices = [], [], []
    
    for i, item in enumerate(candidates_data):
        m = Chem.MolFromSmiles(item['SMILES'])
        if m:
            mols.append(m)
            valid_indices.append(i)
            fingerprints.append(MFPGEN.GetFingerprint(m))

    if not fingerprints:
        return [], []
    
    # Distance Matrix
    dists = []
    nfps = len(fingerprints)
    for i in range(1, nfps):
        sims = DataStructs.BulkTanimotoSimilarity(fingerprints[i], fingerprints[:i])
        dists.extend([1.0 - x for x in sims])

    # Clustering
    logger.info(f"[Clustering] {nfps} molecules (Cutoff={CLUSTER_CUTOFF})...")
    clusters = Butina.ClusterData(dists, nfps, CLUSTER_CUTOFF, isDistData=True)
    representatives = []
    all_members_map = []

    for cluster_idx, cluster in enumerate(clusters):
        # Cluster ID (e.g. CL_001)
        c_id = f"CL_{cluster_idx+1:03d}"
        
        # Centroid selection as first smiles
        centroid_real_idx = valid_indices[cluster[0]]
        centroid_obj = candidates_data[centroid_real_idx].copy()
        centroid_obj['Cluster_ID'] = c_id
        centroid_obj['Cluster_Size'] = len(cluster)
        centroid_obj['Is_Representative'] = True
        representatives.append(centroid_obj)
        
        # Save cluster memebers
        for member_local_idx in cluster:
            real_idx = valid_indices[member_local_idx]
            member_obj = candidates_data[real_idx].copy()
            member_obj['Cluster_ID'] = c_id
            member_obj['Representative_SMILES'] = centroid_obj['SMILES']
            member_obj['Is_Representative'] = (member_local_idx == cluster[0])
            all_members_map.append(member_obj)

    return representatives, all_members_map


def main():
    logger.info("Start Expanding of Ligands:")
    
    # Read seeds
    input_csv = config["ligand_expansion"].get('input_csv')
    if not input_csv or not os.path.exists(input_csv):
        if os.path.exists(os.path.join("..", input_csv)):
            input_csv = os.path.join("..", input_csv)
        else:
            logger.error(f"Input CSV not found: {input_csv}")
            return None

    df = pd.read_csv(input_csv)
    df.columns = [c.strip() for c in df.columns]
    col_map = {c.upper(): c for c in df.columns}
    smi_col = col_map.get("SMILES")
    if not smi_col:
        logger.error("CSV must contain a 'SMILES' column.")
        return

    logger.info(f"Loaded {len(df)} seeds. Use columns: Name, Role, Target.")
    client = get_smallworld_client()
    if not client:
        return None

    expanded_pool: Dict = {}

    for idx, row in df.iterrows():
        seed_smi = row[smi_col]
        seed_name = row.get(col_map.get('NAME', 'Name'), f"Seed_{idx}")
        seed_role = row.get(col_map.get('ROLE', 'Role'), "unknown")
        seed_target = row.get(col_map.get('TARGET', 'Target'), "unknown")
        
        logger.info(f"[Expanding] Finding new SMILES for {seed_name} ({seed_role})")
        analogs = query_smallworld_robust(seed_smi, client)
        count_new = 0
        for i, analog_smi in enumerate(analogs):
            if analog_smi not in expanded_pool:
                if passes_filters(analog_smi):
                    analog_name = f"Analog_{i+1}_of_{seed_name}"
                    
                    expanded_pool[analog_smi] = {
                        "SMILES": analog_smi,
                        "Name": analog_name,
                        "Role": seed_role, # From Seed
                        "Target": seed_target, # From Seed
                        "Parent_Seed": seed_name,
                        "Source": "EnamineREAL"
                    }
                    count_new += 1
        
        logger.info(f"  + Added {count_new}/{MAX_ANALOGS_PER_SEED} valid analogs.")
        time.sleep(API_DELAY)

    candidates_list = list(expanded_pool.values())
    logger.info(f"Total unique valid candidates: {len(candidates_list)}")

    if not candidates_list:
        logger.warning("No candidates generated.")
        return None

    # Clustering
    reps, members = cluster_and_map(candidates_list)
    logger.info(f"Clustering finished ====> Representatives: {len(reps)}; Total tracked: {len(members)}.")

    # File for further docking
    os.makedirs(DATA_DIR, exist_ok=True)
    df_reps = pd.DataFrame(reps)
    cols_order = ["Name", "SMILES", "Role", "Target", "Cluster_ID", "Cluster_Size", "Parent_Seed"]
    final_cols = [c for c in cols_order if c in df_reps.columns] +\
                 [c for c in df_reps.columns if c not in cols_order]
    
    df_reps = df_reps[final_cols]
    df_reps.to_csv(OUTPUT_MAIN, index=False)
    logger.info(f"Saved Representatives to: {OUTPUT_MAIN}")
    df_members = pd.DataFrame(members)
    df_members = df_members.sort_values(by=['Cluster_ID', 'Is_Representative'], ascending=[True, False]) 
    df_members.to_csv(OUTPUT_CLUSTERS, index=False)
    logger.info(f"Saved Cluster Map to: {OUTPUT_CLUSTERS}")


if __name__ == "__main__":
    main()

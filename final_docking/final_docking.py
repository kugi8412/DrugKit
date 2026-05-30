#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# final_docking/final_docking.py

import os
import sys
import json
import subprocess
import shutil
import pandas as pd
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy


CONFIG = {
    'input_csv': 'final_candidates_mixed_v2.csv',
    'output_csv': 'final_smina.csv',
    'structures_dir': '.', # Root dir
    
    'smina_exe': 'smina',
    'box_padding': 0.0,
    
    'exhaustiveness': 16, # Increased for final precision
    'num_modes': 1,
    'cpu_cores': 4,

    # File mapping: Key in JSON -> Actual filename on disk
    'pdb_map': {
        '8I91': '8I91.pdb',
        '8I92': '8I92.pdb',
        '8WBY': '8WBY.pdb',
        '8WM3': '8WM3.pdb',
        'SIT1_MODEL_00': 'SIT1_Model_OO.pdb',
        'SLC6A20': '8I91.pdb',
        'SLC6A19': '8I92.pdb' 
    }
}


def find_file_recursive(name, search_path='.'):
    for root, dirs, files in os.walk(search_path):
        if name in files:
            return os.path.join(root, name)
    return None


def load_all_grids():
    """Loads and merges all JSON grid files found in directory."""
    grids = {}
    json_files = [f for f in os.listdir('.') if f.endswith('.json')]
    for jf in json_files:
        try:
            with open(jf, 'r') as f:
                data = json.load(f)
                grids.update(data)
        except: pass
    return grids


def prepare_ligand(smiles, name, output_pdbqt):
    """Converts SMILES to PDBQT using Meeko."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol: return False
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        
        preparator = MoleculePreparation()
        preparator.prepare(mol)
        
        with open(output_pdbqt, 'w') as f:
            f.write(preparator.write_pdbqt_string())
        return True
    except Exception as e:
        print(f"Ligand error {name}: {e}")
        return False


def parse_affinity_from_stdout(stdout_text):
    """Extracts binding affinity from Smina console output."""
    try:
        lines = stdout_text.split('\n')
        for i, line in enumerate(lines):
            if "Affinity" in line and "kcal/mol" in line:
                data_line = lines[i+2].strip() 
                parts = data_line.split()
                if len(parts) >= 2:
                    return float(parts[1])
    except:
        pass
    return np.nan


def parse_affinity_from_file(file_path):
    """Extracts binding affinity from PDBQT remarks."""
    try:
        if not os.path.exists(file_path): return np.nan
        with open(file_path, 'r') as f:
            for line in f:
                if "minimizedAffinity" in line:
                    return float(line.split()[1])
                if "REMARK VINA RESULT" in line:
                    return float(line.split()[1])
    except:
        pass

    return np.nan


def main():
    print("=== FINAL SMINA DOCKING ===")
    
    # 1. Load Data
    if not os.path.exists(CONFIG['input_csv']):
        print(f"Error: {CONFIG['input_csv']} not found.")
        return
    df = pd.read_csv(CONFIG['input_csv'])
    print(f"Loaded {len(df)} candidates.")

    # 2. Load Grids
    grids_data = load_all_grids()
    if not grids_data:
        print("Error: No docking grids (JSON) found.")
        return

    # 3. Setup Folders
    temp_dir = "temp_docking"
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    
    # 4. Docking Loop
    for target_key, pockets in grids_data.items():
        
        # Resolve PDB file
        pdb_name = CONFIG['pdb_map'].get(target_key, f"{target_key}.pdb")
        pdb_path = find_file_recursive(pdb_name, CONFIG['structures_dir'])
        
        if not pdb_path:
            print(f"[SKIP] Structure not found for: {target_key}")
            continue
            
        print(f"\n<== Processing Target: {target_key} ({pdb_path}) ==>")
        
        for pocket in pockets:
            pocket_id = pocket['id']
         
            print(f" > Pocket: {pocket_id}")
            
            # Column name for results
            col_name = f"{target_key}_{pocket_id}_Smina"
            if col_name not in df.columns:
                df[col_name] = np.nan
            
            # Grid params
            cx, cy, cz = pocket['center']
            sx, sy, sz = pocket['size']
            
            # Dock each ligand
            for idx, row in tqdm(df.iterrows(), total=len(df), leave=False):
                # Skip if already done
                if pd.notnull(row[col_name]): continue
                
                name = str(row['Name']).replace(" ", "_")
                smiles = row['SMILES']
                
                # Paths
                lig_pdbqt = os.path.join(temp_dir, f"{name}.pdbqt")
                out_docked = os.path.join(temp_dir, f"{name}_docked.pdbqt")
                
                # Prep Ligand
                if not os.path.exists(lig_pdbqt):
                    if not prepare_ligand(smiles, name, lig_pdbqt): continue
                
                # Run Smina
                cmd = [
                    CONFIG['smina_exe'],
                    '--receptor', pdb_path,
                    '--ligand', lig_pdbqt,
                    '--center_x', str(cx), '--center_y', str(cy), '--center_z', str(cz),
                    '--size_x', str(sx), '--size_y', str(sy), '--size_z', str(sz),
                    '--exhaustiveness', str(CONFIG['exhaustiveness']),
                    '--num_modes', str(CONFIG['num_modes']),
                    '--cpu', str(CONFIG['cpu_cores']),
                    '--out', out_docked
                ]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    # Parse Score
                    val = parse_affinity_from_file(out_docked)
                    if pd.isna(val):
                        val = parse_affinity_from_stdout(result.stdout)
                        
                    df.at[idx, col_name] = val
                    
                except Exception as e:
                    pass
            
            # Save intermediate results
            df.to_csv(CONFIG['output_csv'], index=False, float_format='%.3f')

    # Cleanup
    if os.path.exists(temp_dir): shutil.rmtree(temp_dir)
    print(f"\nDone. Results saved to {CONFIG['output_csv']}")


if __name__ == "__main__":
    main()

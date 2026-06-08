#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# final_docking/virtual_screenig.py

import os
import sys
import torch
import pandas as pd
import torch.nn as nn
import pubchempy as pcp
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GINEConv, global_add_pool
from rdkit import Chem

# Config
CONFIG = {
    'model_path': 'results/siamese_v15_model.pth',
    'train_data': 'smina_results_exact.csv', 
    'output_file': 'final_candidates_mixed_v2.csv',
    
    # Seed Selection Criteria
    'top_n_affinity': 5,     
    'top_n_selectivity': 5,  
    'min_binding_energy': -4.0, # Filter out weak binders and errors (0.0)
    
    # PubChem Evolution
    'similarity_threshold': 80,   
    'candidates_pool_size': 100,  
    'keep_top_analogs': 2,       
    
    # GINE Model Parameters
    'hidden_dim': 128,
    'dropout': 0.0,
    'batch_size': 32,
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    
    'target_key': 'SLC6A20',
    'offtarget_key': 'SLC6A19'
}

# MODEL
PERMITTED_ATOMS = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'I', 'B', 'H']

def one_hot_encoding(value, choices):
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding


def get_atom_features(atom):
    features = one_hot_encoding(atom.GetSymbol(), PERMITTED_ATOMS)
    features += one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4])
    features += one_hot_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])
    features += one_hot_encoding(atom.GetFormalCharge(), [-1, -2, 1, 2, 0])
    features += one_hot_encoding(str(atom.GetHybridization()), ['SP', 'SP2', 'SP3', 'SP3D', 'SP3D2'])
    features += [1 if atom.GetIsAromatic() else 0]
    features += [atom.GetMass() * 0.01]
    try:
        chiral = str(atom.GetChiralTag())
        features += one_hot_encoding(chiral, ['CHI_TETRAHEDRAL_CW', 'CHI_TETRAHEDRAL_CCW'])

    except: features += [0, 0, 1]
    return features

def get_bond_features(bond):
    bt = bond.GetBondType()
    features = [
        1 if bt == Chem.rdchem.BondType.SINGLE else 0,
        1 if bt == Chem.rdchem.BondType.DOUBLE else 0,
        1 if bt == Chem.rdchem.BondType.TRIPLE else 0,
        1 if bt == Chem.rdchem.BondType.AROMATIC else 0,
    ]
    features += [1 if bond.GetIsConjugated() else 0]
    features += [1 if bond.IsInRing() else 0]
    stereo = str(bond.GetStereo())
    features += one_hot_encoding(stereo, ['STEREOZ', 'STEREOE', 'STEREOCIS', 'STEREOTRANS'])
    return features

def smiles_to_graph_gine(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None

        atom_feats = [get_atom_features(atom) for atom in mol.GetAtoms()]
        x = torch.tensor(atom_feats, dtype=torch.float)
        rows, cols, edge_feats = [], [], []
        for bond in mol.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            b_feat = get_bond_features(bond)
            rows += [start, end]; cols += [end, start]
            edge_feats += [b_feat, b_feat]
        if not rows:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, len(get_bond_features(Chem.MolFromSmiles("CC").GetBondWithIdx(0)))), dtype=torch.float)
        else:
            edge_index = torch.tensor([rows, cols], dtype=torch.long)
            edge_attr = torch.tensor(edge_feats, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=smiles)
    except:
        return None


class GINEEncoder(nn.Module):
    def __init__(self, node_in, edge_in, hidden_dim, dropout=0.3):
        super(GINEEncoder, self).__init__()
        self.node_lin = nn.Linear(node_in, hidden_dim)
        self.edge_lin = nn.Linear(edge_in, hidden_dim)
        def make_gine():
            return GINEConv(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)))
        self.conv1 = make_gine(); self.conv2 = make_gine(); self.conv3 = make_gine()
        self.dropout = dropout
    def forward(self, x, edge_index, edge_attr, batch):
        x = self.node_lin(x)
        if edge_attr.numel() > 0: edge_attr = self.edge_lin(edge_attr)
        else: edge_attr = torch.zeros((edge_index.size(1), x.size(1)), device=x.device)
        x = self.conv1(x, edge_index, edge_attr); x = F.relu(x)
        x = self.conv2(x, edge_index, edge_attr); x = F.relu(x)
        x = self.conv3(x, edge_index, edge_attr); x = F.relu(x)
        return global_add_pool(x, batch)


class SiameseRankNet(nn.Module):
    def __init__(self, node_in, edge_in, hidden_dim, dropout=0.3):
        super(SiameseRankNet, self).__init__()
        self.encoder = GINEEncoder(node_in, edge_in, hidden_dim, dropout)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 1)
    def forward_one(self, data):
        x = self.encoder(data.x, data.edge_index, data.edge_attr, data.batch)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def robust_min(row, cols):
    vals = []
    for c in cols:
        if pd.notnull(row[c]) and isinstance(row[c], (int, float)):
            vals.append(row[c])
    return min(vals) if vals else 999.0


def get_seeds(df):
    """Selects seeds with error filtering."""
    
    # Identify relevant columns
    target_cols = [c for c in df.columns if CONFIG['target_key'] in c]
    offtarget_cols = [c for c in df.columns if CONFIG['offtarget_key'] in c]
    
    print(f"Found {len(target_cols)} Target columns and {len(offtarget_cols)} Off-Target columns.")
    
    # Compute metrics
    df['Best_Target'] = df.apply(lambda r: robust_min(r, target_cols), axis=1)
    df['Best_OffTarget'] = df.apply(lambda r: robust_min(r, offtarget_cols), axis=1)
    
    # Filter weak/invalid results
    valid_df = df[
        (df['Best_Target'] < CONFIG['min_binding_energy']) & 
        (df['Best_OffTarget'] < CONFIG['min_binding_energy'])
    ].copy()
    
    print(f"After filtering (Energy < {CONFIG['min_binding_energy']}): {len(valid_df)} compounds remaining.")
    
    if len(valid_df) == 0:
        print("[CRITICAL] No compounds meet the energy criteria! Check input file.")
        sys.exit()

    # Selectivity = OffTarget - Target
    valid_df['Selectivity'] = valid_df['Best_OffTarget'] - valid_df['Best_Target']
    
    # Select Top Affinity
    top_affinity = valid_df.sort_values('Best_Target', ascending=True).head(CONFIG['top_n_affinity'])
    top_affinity['Seed_Type'] = 'Affinity_Elite'
    
    # Select Top Selectivity
    top_selectivity = valid_df.sort_values('Selectivity', ascending=False).head(CONFIG['top_n_selectivity'])
    top_selectivity['Seed_Type'] = 'Selectivity_Elite'
    
    # Merge
    combined = pd.concat([top_affinity, top_selectivity]).drop_duplicates(subset='SMILES')
    return combined


# Helper
def fetch_similar_compounds(smiles, threshold=80, limit=100):
    try:
        compounds = pcp.get_compounds(smiles, namespace='smiles', searchtype='similarity', threshold=threshold, listkey_count=limit)
        fetched_smiles = [c.isomeric_smiles for c in compounds if c.isomeric_smiles]
        return list(set(fetched_smiles))
    except Exception as e:
        print(f"   [PubChem Error]: {e}")
        return []

def predict_score_single(model, smiles):
    model.eval()
    g = smiles_to_graph_gine(smiles)
    if not g:
        return -999.0

    loader = DataLoader([g], batch_size=1, shuffle=False)
    with torch.no_grad():
        for batch in loader:
            return model.forward_one(batch.to(CONFIG['device'])).item()

    return -999.0

def predict_scores_batch(model, smiles_list, batch_size):
    model.eval()
    data_list = []
    valid_smiles = []
    for s in smiles_list:
        g = smiles_to_graph_gine(s)
        if g:
            data_list.append(g)
            valid_smiles.append(s)
    if not data_list:
        return []

    loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)
    scores = []
    with torch.no_grad():
        for batch in loader:
            out = model.forward_one(batch.to(CONFIG['device']))
            scores.extend(out.cpu().numpy().flatten())

    return list(zip(valid_smiles, scores))


def main():
    print("<== SCREENING ONLINE v2: FILTERED & BALANCED ==>")
    
    # 1. Load Data
    if not os.path.exists(CONFIG['train_data']):
        print("Docking results file not found.")
        return
    df = pd.read_csv(CONFIG['train_data'])
    
    # Select Seeds
    seeds_df = get_seeds(df)
    print(f"\nSelected {len(seeds_df)} seeds for evolution:")
    for _, row in seeds_df.iterrows():
        print(f" - {row['Name'][:25]}... ({row['Seed_Type']}) | Target: {row['Best_Target']:.2f}, Selectivity: {row['Selectivity']:.2f}")

    # Load Model
    dummy = smiles_to_graph_gine('C')
    model = SiameseRankNet(dummy.x.shape[1], dummy.edge_attr.shape[1], CONFIG['hidden_dim'], CONFIG['dropout']).to(CONFIG['device'])
    model.load_state_dict(torch.load(CONFIG['model_path'], map_location=CONFIG['device']))
    
    final_results = []
    known_smiles = set(df['SMILES'].tolist())
    
    # Evolution Loop
    print("\n--- Starting analog retrieval ---")
    
    for i, row in seeds_df.iterrows():
        seed_smi = row['SMILES']
        seed_name = row['Name']
        seed_type = row['Seed_Type']
        
        # Score the seed with GNN
        seed_gnn_score = predict_score_single(model, seed_smi)
        
        final_results.append({
            'SMILES': seed_smi,
            'Name': seed_name,
            'GNN_Score': seed_gnn_score,
            'Source': f"Seed_{seed_type}",
            'Parent_Score': seed_gnn_score
        })
        
        print(f"\nProcessing Seed: {seed_name[:20]}... | GNN: {seed_gnn_score:.4f}")
        
        analogs = fetch_similar_compounds(seed_smi, threshold=CONFIG['similarity_threshold'], limit=CONFIG['candidates_pool_size'])
        analogs = [s for s in analogs if s not in known_smiles]
        
        if not analogs:
            print("   -> No new analogs found.")
            continue
            
        scored_analogs = predict_scores_batch(model, analogs, CONFIG['batch_size'])
        
        # Select best performers relative to seed
        better = [(s, sc) for s, sc in scored_analogs if sc > seed_gnn_score]
        better.sort(key=lambda x: x[1], reverse=True)
        
        selected = better[:CONFIG['keep_top_analogs']]
        print(f"   -> Found {len(better)} better candidates. Selected {len(selected)}.")
        
        for s, sc in selected:
            final_results.append({
                'SMILES': s,
                'Name': f"Analog_of_{seed_name}",
                'GNN_Score': sc,
                'Source': f"Analog_of_{seed_type}",
                'Parent_Score': seed_gnn_score
            })
            known_smiles.add(s)

    # Save Results
    df_out = pd.DataFrame(final_results)
    df_out.to_csv(CONFIG['output_file'], index=False)
    
    print("="*40)
    print(f"Finished. Collected {len(df_out)} structures.")
    print(f"Results saved to: {CONFIG['output_file']}")

 
if __name__ == "__main__":
    main()

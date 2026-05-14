# -*- coding: utf-8 -*-

from typing import Optional, Tuple, Any

from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy


def prepare_ligand(smiles: str, name: str) -> Optional[Tuple[str, Any]]:
    """
    Zwraca (pdbqt_string, rdkit_mol) albo None
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None

        mol = Chem.AddHs(mol)
        mol.SetProp("_Name", name)

        try:
            params = AllChem.ETKDGv3()
        except Exception:
            params = AllChem.ETKDG()

        params.randomSeed = 42
        params.useRandomCoords = True

        if AllChem.EmbedMolecule(mol, params) == -1:
            params_fb = AllChem.ETKDG()
            params_fb.useRandomCoords = True
            if AllChem.EmbedMolecule(mol, params_fb) == -1:
                return None

        try:
            Chem.SanitizeMol(mol)
            AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            pass

        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)
        if not mol_setups:
            return None

        output = PDBQTWriterLegacy.write_string(mol_setups[0])
        if isinstance(output, tuple):
            return output[0], mol
        return output, mol
    except Exception:
        return None

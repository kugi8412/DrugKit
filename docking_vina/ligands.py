# src/docking_vina/ligands.py
# -*- coding: utf-8 -*-

from typing import Any, Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem
from meeko import MoleculePreparation, PDBQTWriterLegacy


def prepare_ligand(smiles: str, name: str) -> Optional[Tuple[str, Any]]:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None

        mol = Chem.AddHs(mol)
        mol.SetProp("_Name", name)

        try:
            params = AllChem.ETKDGv3()
        except AttributeError:
            params = AllChem.ETKDG()

        params.randomSeed = 42
        params.useRandomCoords = True

        res = AllChem.EmbedMolecule(mol, params)

        if res == -1:
            params_fb = AllChem.ETKDG()
            params_fb.useRandomCoords = True
            res = AllChem.EmbedMolecule(mol, params_fb)

        if res == -1:
            return None

        try:
            Chem.SanitizeMol(mol)
            AllChem.UFFOptimizeMolecule(mol)
        except Exception:
            pass

        preparator = MoleculePreparation()
        mol_setups = preparator.prepare(mol)

        if mol_setups and len(mol_setups) > 0:
            output = PDBQTWriterLegacy.write_string(mol_setups[0])

            if isinstance(output, tuple):
                return output[0], mol
            if isinstance(output, str):
                return output, mol

    except Exception:
        return None

    return None

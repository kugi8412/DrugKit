# src/docking_vina/vina_engine.py
# -*- coding: utf-8 -*-

from typing import List, Tuple
import numpy as np
from vina import Vina


def run_vina_scoring(
    pdbqt_ligand: str,
    receptor_path: str,
    center: List[float],
    size: List[float],
    base_exhaustiveness: int,
) -> Tuple[float, str]:
    try:
        if not pdbqt_ligand or len(pdbqt_ligand) < 10:
            return np.nan, ""

        volume = size[0] * size[1] * size[2]
        exhaustiveness = base_exhaustiveness
        if volume > 27000:
            exhaustiveness = max(base_exhaustiveness, 32)

        v = Vina(sf_name="vina", cpu=1, verbosity=0)
        v.set_receptor(receptor_path)
        v.set_ligand_from_string(pdbqt_ligand)
        v.compute_vina_maps(center=center, box_size=size)
        v.dock(exhaustiveness=exhaustiveness, n_poses=1)

        energies = v.energies(n_poses=1)
        if energies is not None and len(energies) > 0:
            return energies[0][0], v.poses(n_poses=1)

    except Exception:
        return np.nan, ""

    return np.nan, ""

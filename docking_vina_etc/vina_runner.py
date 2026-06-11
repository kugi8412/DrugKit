#!/usr/bin/env python3
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
    """
    Return (energy, pose_pdbqt_string)
    """
    try:
        if not pdbqt_ligand:
            return float("nan"), ""

        volume = float(size[0]) * float(size[1]) * float(size[2])
        exhaustiveness = max(base_exhaustiveness, 128) if volume > 27000 else base_exhaustiveness

        v = Vina(sf_name="vina", cpu=1, verbosity=0)
        v.set_receptor(receptor_path)
        v.set_ligand_from_string(pdbqt_ligand)
        v.compute_vina_maps(center=center, box_size=size)
        v.dock(exhaustiveness=exhaustiveness, n_poses=1)

        energies = v.energies(n_poses=1)
        if energies is not None and len(energies) > 0:
            return float(energies[0][0]), v.poses(n_poses=1)
    except Exception:
        pass

    return float("nan"), ""

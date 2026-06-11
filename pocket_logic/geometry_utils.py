#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pocket_logic/geometry_utils.py

from typing import List, Tuple, Set
import numpy as np


def get_residue_coords(structure,
                       target_chain: str,
                       res_ids: Set[int]) -> np.ndarray:
    atoms = []
    for model in structure:
        for chain in model:
            if chain.id != target_chain:
                continue
            for res in chain:
                if res.id[1] in res_ids:
                    if "CA" in res:
                        atoms.append(res["CA"].get_coord())
                    elif len(res) > 0:
                        atoms.append(res.child_list[0].get_coord())
    return np.array(atoms) if atoms else np.empty((0, 3))


def calculate_centered_box(coords: np.ndarray,
                           buffer: float,
                           margin: float = 2.0) -> Tuple[List[float], List[float]]:
    if coords.size == 0:
        return [0.0] * 3, [30.0] * 3

    center = np.mean(coords, axis=0)
    diffs = np.abs(coords - center)
    max_dist = np.max(diffs, axis=0)
    size = (max_dist * margin) + buffer
    size = [min(float(s), 30.0 + buffer) for s in size]
    return center.tolist(), size


def calculate_box_from_radius(center: List[float],
                              radius: float,
                              buffer: float,
                              margin: float = 2.0) -> List[float]:
    side = min((radius * margin) + buffer, 30.0 * margin)
    return [side, side, side]


def check_proximity_to_chain(center: List[float],
                             structure,
                             target_chain: str,
                             threshold: float = 10.0) -> bool:
    c_vec = np.array(center)
    for model in structure:
        for chain in model:
            if chain.id != target_chain:
                continue
            for res in list(chain.get_residues()):
                if "CA" in res:
                    if np.linalg.norm(c_vec - res["CA"].get_coord()) < threshold:
                        return True
    return False

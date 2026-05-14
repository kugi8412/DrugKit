# -*- coding: utf-8 -*-

from typing import Dict
from ipz_core.config_loader import StructureConfig


def build_target_to_protein_map(structures: Dict[str, StructureConfig]) -> Dict[str, str]:
    """
    Flattened structures keys are like "PROT/CONF".
    We map target_id (conformation_key) -> protein_key
    """
    out: Dict[str, str] = {}
    for _, scfg in structures.items():
        out[str(scfg.conformation_key)] = str(scfg.protein_key)
    return out


def compute_selectivity_min_for_target(
    energies_by_target: Dict[str, float],
    target_id: str,
    target_to_protein: Dict[str, str],
) -> float:
    """
    Selectivity(target_id) = min_{other targets from other proteins} (E_other - E_target)

    - higher value => better selectivity for target_id (because E_target is more negative)
    - if no other proteins => 0.0
    - if uncountable => NaN
    """
    if target_id not in energies_by_target:
        return float("nan")

    my_prot = target_to_protein.get(str(target_id))
    if not my_prot:
        return float("nan")

    e_t = energies_by_target[target_id]
    candidates = []
    for other_t, e_o in energies_by_target.items():
        if other_t == target_id:
            continue
        if target_to_protein.get(str(other_t)) == my_prot:
            continue
        candidates.append(e_o - e_t)

    if not candidates:
        return 0.0

    return float(min(candidates))

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Acquisition functions for active learning.
"""

from typing import List, Sequence, Set

import numpy as np


def select_top_uncertain(names: Sequence[str], uncertainties: np.ndarray,
                         k: int, exclude: Set[str]) -> List[str]:
    """Return up to k names with the highest uncertainty, skipping `exclude`."""
    order = np.argsort(-np.asarray(uncertainties), kind="stable")
    picked: List[str] = []
    for idx in order:
        name = names[idx]
        if name in exclude:
            continue
        picked.append(name)
        if len(picked) >= k:
            break
    return picked

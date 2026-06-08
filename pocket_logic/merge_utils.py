#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pocket_logic/merge_utils.py

from typing import List
import numpy as np


def is_duplicate(center: List[float],
                 existing_pockets: List[dict],
                 overlap_threshold: float) -> bool:
    for ex in existing_pockets:
        if np.linalg.norm(np.array(center) - np.array(ex["center"])) < overlap_threshold:
            return True
    return False

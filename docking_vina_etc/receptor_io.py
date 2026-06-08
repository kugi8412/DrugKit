# -*- coding: utf-8 -*-

import os
from typing import Optional


def get_receptor_path(target_id: str, data_dir: str) -> Optional[str]:
    """
    target_id = conformation_key (np. 8WM3, SIT1_MODEL_00)
    Szuka {target_id}.pdbqt w kilku typowych miejscach.
    """
    possibilities = [
        f"output/{target_id}.pdbqt",
        f"data/{target_id}.pdbqt",
        f"{target_id}.pdbqt",
        os.path.join(data_dir, f"{target_id}.pdbqt"),
    ]
    for p in possibilities:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return os.path.abspath(p)
    return None

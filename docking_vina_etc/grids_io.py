# -*- coding: utf-8 -*-

import json
from typing import Dict, List


def load_grids_json(grids_file: str) -> Dict[str, List[dict]]:
    with open(grids_file, "r", encoding="utf-8") as f:
        return json.load(f)

# -*- coding: utf-8 -*-

import os
import subprocess
import tempfile
from typing import List, Tuple

import numpy as np


def _first_float_in_line(line: str) -> float:
    for token in line.split():
        cleaned = token.rstrip(":")
        try:
            return float(cleaned)
        except ValueError:
            continue
    return float("nan")


def parse_affinity_from_stdout(stdout_text: str) -> float:
    try:
        for line in stdout_text.splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                return float(parts[1])
    except Exception:
        pass
    return float("nan")


def parse_affinity_from_file(file_path: str) -> float:
    try:
        if not os.path.exists(file_path):
            return float("nan")
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                if "minimizedAffinity" in line or "REMARK VINA RESULT" in line:
                    value = _first_float_in_line(line)
                    if not np.isnan(value):
                        return value
    except Exception:
        pass
    return float("nan")


def _effective_exhaustiveness(base_exhaustiveness: int, size: List[float]) -> int:
    volume = float(size[0]) * float(size[1]) * float(size[2])
    if volume > 27000:
        return max(base_exhaustiveness, 32)
    return base_exhaustiveness


def run_smina_scoring(
    receptor_path: str,
    pdbqt_ligand: str,
    center: List[float],
    size: List[float],
    base_exhaustiveness: int,
    smina_exe: str = "smina",
    num_modes: int = 1,
    cpu: int = 1,
) -> Tuple[float, str]:
    try:
        if not pdbqt_ligand or len(pdbqt_ligand) < 10:
            return np.nan, ""
        if not receptor_path or not os.path.exists(receptor_path):
            return np.nan, ""

        exhaustiveness = _effective_exhaustiveness(base_exhaustiveness, size)

        with tempfile.TemporaryDirectory(prefix="smina_") as tmp_dir:
            lig_path = os.path.join(tmp_dir, "ligand.pdbqt")
            out_path = os.path.join(tmp_dir, "docked.pdbqt")

            with open(lig_path, "w", encoding="utf-8") as f:
                f.write(pdbqt_ligand)

            cmd = [
                smina_exe,
                "--receptor",
                receptor_path,
                "--ligand",
                lig_path,
                "--center_x",
                str(center[0]),
                "--center_y",
                str(center[1]),
                "--center_z",
                str(center[2]),
                "--size_x",
                str(size[0]),
                "--size_y",
                str(size[1]),
                "--size_z",
                str(size[2]),
                "--exhaustiveness",
                str(exhaustiveness),
                "--num_modes",
                str(num_modes),
                "--cpu",
                str(cpu),
                "--out",
                out_path,
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=False)

            energy = parse_affinity_from_file(out_path)
            if np.isnan(energy):
                energy = parse_affinity_from_stdout(result.stdout)

            pose = ""
            if os.path.exists(out_path):
                with open(out_path, "r", encoding="utf-8") as f:
                    pose = f.read()

            if not np.isnan(energy) and pose:
                return float(energy), pose

    except Exception:
        return np.nan, ""

    return np.nan, ""

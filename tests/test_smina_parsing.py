#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pytest

from docking_smina.smina_engine import (
    parse_affinity_from_file,
    parse_affinity_from_stdout,
)


def test_parse_affinity_from_file_minimized_affinity(tmp_path):
    pdbqt = tmp_path / "out.pdbqt"
    pdbqt.write_text(
        "REMARK minimizedAffinity -8.123\n"
        "ATOM\n",
        encoding="utf-8",
    )
    assert parse_affinity_from_file(str(pdbqt)) == pytest.approx(-8.123)


def test_parse_affinity_from_file_vina_result_remark(tmp_path):
    pdbqt = tmp_path / "out.pdbqt"
    pdbqt.write_text(
        "REMARK VINA RESULT:    -7.500      0.000      0.000\n"
        "ATOM\n",
        encoding="utf-8",
    )
    assert parse_affinity_from_file(str(pdbqt)) == pytest.approx(-7.5)


def test_parse_affinity_from_file_missing_returns_nan(tmp_path):
    assert np.isnan(parse_affinity_from_file(str(tmp_path / "missing.pdbqt")))


def test_parse_affinity_from_stdout():
    stdout = (
        "   -----+------------+----------+----------\n"
        "   mode |   affinity | dist from|  closest |\n"
        "        | (kcal/mol) | best mode|  rmsd l.b.|\n"
        "   -----+------------+----------+----------\n"
        "     1       -9.210          0          0\n"
    )
    assert parse_affinity_from_stdout(stdout) == pytest.approx(-9.21)


def test_parse_affinity_from_stdout_missing_returns_nan():
    assert np.isnan(parse_affinity_from_stdout("no scores here"))

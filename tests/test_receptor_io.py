import os

import pytest

from docking_common.receptor_io import get_receptor_path, sanitize_key


def test_sanitize_key_replaces_slashes():
    assert sanitize_key("foo/bar") == "foo_bar"
    assert sanitize_key("foo\\bar") == "foo_bar"


def test_get_receptor_path_prefers_pdbqt_over_pdb(tmp_path):
    target = "8I91"
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)

    pdb_path = tmp_path / "output" / f"{target}.pdb"
    pdbqt_path = tmp_path / "output" / f"{target}.pdbqt"
    os.makedirs(pdb_path.parent, exist_ok=True)
    pdb_path.write_text("ATOM", encoding="utf-8")
    pdbqt_path.write_text("ATOM", encoding="utf-8")

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = get_receptor_path(target, data_dir)
    finally:
        os.chdir(cwd)

    assert result is not None
    assert result.endswith(".pdbqt")


def test_get_receptor_path_falls_back_to_pdb(tmp_path):
    target = "8I92"
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)

    pdb_path = tmp_path / "data" / f"{target}.pdb"
    pdb_path.write_text("ATOM", encoding="utf-8")

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = get_receptor_path(target, data_dir)
    finally:
        os.chdir(cwd)

    assert result is not None
    assert result.endswith(".pdb")


def test_get_receptor_path_pdbqt_only(tmp_path):
    target = "8WM3"
    data_dir = str(tmp_path / "data")
    os.makedirs(data_dir, exist_ok=True)

    pdb_path = tmp_path / "data" / f"{target}.pdb"
    pdb_path.write_text("ATOM", encoding="utf-8")

    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = get_receptor_path(target, data_dir, extensions=(".pdbqt",))
    finally:
        os.chdir(cwd)

    assert result is None

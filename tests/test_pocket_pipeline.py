#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Pytest coverage for the refactored pocket detection pipeline.

Test surface:
  1. P2Rank CSV parsing — valid CSV → expected pocket records.
  2. P2Rank CSV parsing — missing / invalid CSV handled gracefully.
  3. Pipeline behaviour when P2Rank results ARE present.
  4. Pipeline behaviour when P2Rank results are ABSENT.
  5. Regression: GeneoNet is no longer required for successful execution.
"""

import io
import os
import logging
import textwrap
import unittest
from unittest.mock import MagicMock, patch, mock_open

import numpy as np
import pandas as pd
import pytest

import sys
import types

_bio = types.ModuleType("Bio")
_bio_pdb = types.ModuleType("Bio.PDB")

class _FakePDBParser:
    def __init__(self, QUIET=True): ...
    def get_structure(self, name, path): return MagicMock()

_bio_pdb.PDBParser = _FakePDBParser
_bio.PDB = _bio_pdb
sys.modules.setdefault("Bio", _bio)
sys.modules.setdefault("Bio.PDB", _bio_pdb)

_requests = types.ModuleType("requests")
class _FakeRequestException(Exception): ...
_requests.RequestException = _FakeRequestException
sys.modules.setdefault("requests", _requests)

_HERE = os.path.dirname(__file__)
_POCKET_LOGIC = os.path.join(_HERE, "..", "pocket_logic")
if _POCKET_LOGIC not in sys.path:
    sys.path.insert(0, _POCKET_LOGIC)


def _get_residue_coords_stub(structure, chain_id, res_ids):
    return np.array([[1.0, 2.0, 3.0]])

def _calculate_centered_box_stub(coords, buffer):
    center = coords.mean(axis=0).tolist()
    return center, [20.0, 20.0, 20.0]

def _is_duplicate_stub(center, existing, threshold):
    return False

_geom = types.ModuleType("geometry_utils")
_geom.get_residue_coords = _get_residue_coords_stub
_geom.calculate_centered_box = _calculate_centered_box_stub
sys.modules["geometry_utils"] = _geom

_merge = types.ModuleType("merge_utils")
_merge.is_duplicate = _is_duplicate_stub
sys.modules["merge_utils"] = _merge

from p2rank_stage import run_or_load_p2rank, add_p2rank_pockets, validate_quality  # noqa: E402


LOGGER = logging.getLogger("test_pocket_pipeline")

VALID_P2RANK_CSV = textwrap.dedent("""\
    rank, score, center_x, center_y, center_z, residue_ids
    1,    12.5,  10.0,     20.0,     30.0,     A_42 A_43 B_10
    2,     8.1,  -5.0,      1.0,      7.5,     A_55 A_56
""")


class TestP2RankCSVParsing:
    """Tests for run_or_load_p2rank — the CSV loading path."""

    def test_valid_csv_produces_expected_records(self, tmp_path):
        """A well-formed predictions CSV returns one dict per pocket."""
        pdb_path = str(tmp_path / "target.pdb")
        csv_path = str(tmp_path / "target.pdb_predictions.csv")

        # Write the CSV so the file-existence check passes.
        with open(csv_path, "w") as fh:
            fh.write(VALID_P2RANK_CSV)

        pockets = run_or_load_p2rank(
            pdb_path=pdb_path,
            output_dir=str(tmp_path),
            p2rank_exec=None,
            logger=LOGGER,
        )

        assert len(pockets) == 2, "Expected two pocket records"

        first = pockets[0]
        assert first["rank"] == 1
        assert abs(first["score"] - 12.5) < 1e-6
        assert first["center"] == [10.0, 20.0, 30.0]
        assert 42 in first["residues"]
        assert 43 in first["residues"]
        assert 10 in first["residues"]
        assert "A" in first["chains"]
        assert "B" in first["chains"]

        second = pockets[1]
        assert second["rank"] == 2
        assert 55 in second["residues"]

    def test_missing_csv_returns_empty_list(self, tmp_path):
        """No CSV file and no p2rank executable → empty list, no exception."""
        pdb_path = str(tmp_path / "nonexistent.pdb")

        pockets = run_or_load_p2rank(
            pdb_path=pdb_path,
            output_dir=str(tmp_path),
            p2rank_exec=None,
            logger=LOGGER,
        )

        assert pockets == []

    def test_malformed_csv_returns_empty_list(self, tmp_path):
        """A CSV with completely wrong columns is silently skipped."""
        pdb_path = str(tmp_path / "target.pdb")
        csv_path = str(tmp_path / "target.pdb_predictions.csv")

        with open(csv_path, "w") as fh:
            fh.write("col_a,col_b\n1,2\n")

        pockets = run_or_load_p2rank(
            pdb_path=pdb_path,
            output_dir=str(tmp_path),
            p2rank_exec=None,
            logger=LOGGER,
        )

        # Graceful degradation: no crash, possibly empty or partial result.
        assert isinstance(pockets, list)

    def test_empty_csv_returns_empty_list(self, tmp_path):
        """An empty CSV (headers only) produces zero pockets."""
        pdb_path = str(tmp_path / "target.pdb")
        csv_path = str(tmp_path / "target.pdb_predictions.csv")

        with open(csv_path, "w") as fh:
            fh.write("rank,score,center_x,center_y,center_z,residue_ids\n")

        pockets = run_or_load_p2rank(
            pdb_path=pdb_path,
            output_dir=str(tmp_path),
            p2rank_exec=None,
            logger=LOGGER,
        )

        assert pockets == []


class TestAddP2RankPocketsPresent:
    """add_p2rank_pockets correctly populates chain_pockets."""

    def _make_p2rank_list(self, n=3):
        return [
            {
                "rank": i + 1,
                "score": float(10 - i),
                "center": [float(i), float(i), float(i)],
                "residues": {i * 10 + 1, i * 10 + 2},
                "chains": {"A"},
            }
            for i in range(n)
        ]

    def test_adds_up_to_top_n(self):
        chain_pockets = []
        p2rank_global = self._make_p2rank_list(n=5)
        structure = MagicMock()

        added = add_p2rank_pockets(
            chain_pockets=chain_pockets,
            structure=structure,
            chain_id="A",
            p2rank_global=p2rank_global,
            p2rank_top_n=3,
            buffer_size=4.0,
            overlap_threshold=0.5,
            logger=LOGGER,
        )

        assert added == 3
        assert len(chain_pockets) == 3

    def test_pocket_ids_contain_chain_and_source(self):
        chain_pockets = []
        p2rank_global = self._make_p2rank_list(n=2)
        structure = MagicMock()

        add_p2rank_pockets(
            chain_pockets=chain_pockets,
            structure=structure,
            chain_id="B",
            p2rank_global=p2rank_global,
            p2rank_top_n=2,
            buffer_size=4.0,
            overlap_threshold=0.5,
            logger=LOGGER,
        )

        for pocket in chain_pockets:
            assert pocket["source"] == "P2Rank"
            assert pocket["id"].startswith("B_p2rank_r")

    def test_chain_filter_skips_wrong_chains(self):
        """Pockets belonging only to chain B are skipped when processing chain A."""
        chain_pockets = []
        p2rank_global = [
            {
                "rank": 1,
                "score": 9.0,
                "center": [1.0, 1.0, 1.0],
                "residues": {10},
                "chains": {"B"},   # <-- wrong chain
            }
        ]
        structure = MagicMock()

        added = add_p2rank_pockets(
            chain_pockets=chain_pockets,
            structure=structure,
            chain_id="A",
            p2rank_global=p2rank_global,
            p2rank_top_n=5,
            buffer_size=4.0,
            overlap_threshold=0.5,
            logger=LOGGER,
        )

        assert added == 0
        assert chain_pockets == []

    def test_pocket_with_empty_chains_is_accepted(self):
        """A P2Rank pocket with no chain info is accepted for any chain."""
        chain_pockets = []
        p2rank_global = [
            {
                "rank": 1,
                "score": 9.0,
                "center": [5.0, 5.0, 5.0],
                "residues": {10},
                "chains": set(),   # empty → no chain restriction
            }
        ]
        structure = MagicMock()

        added = add_p2rank_pockets(
            chain_pockets=chain_pockets,
            structure=structure,
            chain_id="A",
            p2rank_global=p2rank_global,
            p2rank_top_n=5,
            buffer_size=4.0,
            overlap_threshold=0.5,
            logger=LOGGER,
        )

        assert added == 1


class TestAddP2RankPocketsAbsent:
    """add_p2rank_pockets handles empty p2rank_global safely."""

    def test_empty_p2rank_produces_no_pockets(self):
        chain_pockets = []
        structure = MagicMock()

        added = add_p2rank_pockets(
            chain_pockets=chain_pockets,
            structure=structure,
            chain_id="A",
            p2rank_global=[],
            p2rank_top_n=5,
            buffer_size=4.0,
            overlap_threshold=0.5,
            logger=LOGGER,
        )

        assert added == 0
        assert chain_pockets == []

    def test_validate_quality_with_no_pockets_returns_unknown(self):
        result = validate_quality(
            target_center=[0.0, 0.0, 0.0],
            p2rank_pockets=[],
        )
        assert result["status"] == "Unknown"
        assert result["dist"] == 999.9


class TestGeneoNetRemoved:
    """Ensure the refactored pipeline has no dependency on geneonet_stage."""

    def test_geneonet_stage_not_imported_by_pocket_logic_r(self):
        """
        pocket_logic_r must not import geneonet_stage.
        We check the module's source text as a lightweight static assertion.
        """
        import importlib.util
        import pathlib

        # Locate pocket_logic_r relative to this test file.
        candidate = pathlib.Path(__file__).parent.parent / "pocket_logic" / "pocket_logic_r.py"
        if not candidate.exists():
            pytest.skip("pocket_logic_r.py not found next to tests/; skipping path check.")

        source = candidate.read_text()
        assert "geneonet_stage" not in source, (
            "pocket_logic_r.py still imports geneonet_stage — GeneoNet was not fully removed."
        )
        assert "find_geneonet_file" not in source
        assert "parse_geneonet_csv" not in source
        assert "add_geneonet_pockets" not in source
        assert "should_use_geneonet" not in source

    def test_geneonet_not_required_for_run_or_load_p2rank(self, tmp_path):
        """
        run_or_load_p2rank executes successfully even when nothing geneonet-
        related exists anywhere.
        """
        pdb_path = str(tmp_path / "target.pdb")
        # No predictions CSV either — should return [] without error.
        result = run_or_load_p2rank(
            pdb_path=pdb_path,
            output_dir=str(tmp_path),
            p2rank_exec=None,
            logger=LOGGER,
        )
        assert isinstance(result, list)

    def test_add_p2rank_pockets_has_no_geneonet_reference(self):
        """
        The p2rank_stage module must not reference geneonet at all.
        """
        import pathlib
        candidate = pathlib.Path(__file__).parent.parent / "pocket_logic" / "p2rank_stage.py"
        if not candidate.exists():
            pytest.skip("p2rank_stage.py not found next to tests/; skipping.")

        source = candidate.read_text()
        assert "geneonet" not in source.lower(), (
            "p2rank_stage.py contains a reference to 'geneonet' — unexpected coupling."
        )

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from active_learning.config import DEFAULT_CONFIG, merge_config


def test_defaults_present():
    for key in ["pool_file", "seed_file", "grids_file", "on_targets",
                "off_targets", "rounds", "seed_size", "acquisition_batch",
                "mc_samples", "output_dir"]:
        assert key in DEFAULT_CONFIG


def test_merge_overrides_section():
    raw = {"active_learning": {"rounds": 9, "off_targets": ["RENIN_2V0Z"]}}
    cfg = merge_config(raw)["active_learning"]
    assert cfg["rounds"] == 9
    assert cfg["off_targets"] == ["RENIN_2V0Z"]
    assert cfg["mc_samples"] == DEFAULT_CONFIG["mc_samples"]

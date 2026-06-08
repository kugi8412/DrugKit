import pandas as pd

import active_learning.docking_adapter as adapter


def test_dock_compounds_writes_temp_csv_and_returns_df(monkeypatch, tmp_path):
    captured = {}

    def fake_run_candidates(candidates_file, rec_map, grids, thresholds,
                            output_poses_dir, n_cpu, base_exhaustiveness,
                            smina_exe, num_modes, logger):
        df = pd.read_csv(candidates_file)
        captured["names"] = list(df["Name"])
        captured["thresholds"] = thresholds
        results = []
        for _, r in df.iterrows():
            results.append({"Name": r["Name"], "SMILES": r["SMILES"],
                            "Target": "HIVPRO_1HSG", "Pocket_ID": "p1", "Energy": -7.5})
        return results, 0

    monkeypatch.setattr(adapter, "run_candidates", fake_run_candidates)

    records = [{"Name": "A", "SMILES": "CCO"}, {"Name": "B", "SMILES": "CCN"}]
    cfg = {"default_baseline": -7.0, "exhaustiveness": 8,
           "smina_exe": "smina", "num_modes": 1, "n_cpu": 2}
    grids = {"HIVPRO_1HSG": [{"id": "p1"}], "RENIN_2V0Z": [{"id": "q1"}]}
    out = adapter.dock_compounds(records, rec_map={"HIVPRO_1HSG": "r.pdb"},
                                 grids=grids, cfg=cfg,
                                 output_poses_dir=str(tmp_path), logger=None)
    assert isinstance(out, pd.DataFrame)
    assert set(captured["names"]) == {"A", "B"}
    assert captured["thresholds"] == {"HIVPRO_1HSG": -7.0, "RENIN_2V0Z": -7.0}
    assert len(out) == 2

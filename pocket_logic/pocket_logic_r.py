# pocket_logic.py
# -*- coding: utf-8 -*-

import os
import json
import logging
from typing import List, Dict, Optional

import pandas as pd
from Bio.PDB import PDBParser

from ipz_core.logging_utils import setup_logging
from ipz_core.config_loader import (
    ConfigLoader,
    ProjectConfig,
    PocketAnalysisConfig,
    StructureConfig,
)

from p2rank_stage import run_or_load_p2rank, add_p2rank_pockets
from geneonet_stage import find_geneonet_file, should_use_geneonet, parse_geneonet_csv, add_geneonet_pockets
from expert_knowledge import load_expert_df, add_expert_pockets


class PocketPipeline:
    def __init__(self,
                 project_cfg: ProjectConfig,
                 pa_cfg: PocketAnalysisConfig,
                 logger: logging.Logger):
        self.project_cfg = project_cfg
        self.pa_cfg = pa_cfg
        self.logger = logger

        os.makedirs(self.project_cfg.data_dir, exist_ok=True)
        self.parser = PDBParser(QUIET=True)

    def load_structure(self, s_cfg: StructureConfig):
        pdb_path = s_cfg.resolve_structure_path(self.project_cfg.data_dir)
        if not os.path.exists(pdb_path):
            self.logger.error(f"[CRITICAL] PDB file not found for {s_cfg.conformation_key}: {pdb_path}")
            return None, pdb_path

        structure = self.parser.get_structure(s_cfg.conformation_key, pdb_path)
        return structure, pdb_path

    def load_global_predictors(self, s_cfg: StructureConfig, pdb_path: str):
        p2rank_global = run_or_load_p2rank(
            pdb_path=pdb_path,
            output_dir=self.project_cfg.data_dir,
            p2rank_exec=self.pa_cfg.p2rank_path,
            logger=self.logger
        )

        gn_file = find_geneonet_file(self.pa_cfg.geneonet_path, [s_cfg.pdb_id_lower, s_cfg.conformation_key])
        use_gn = should_use_geneonet(s_cfg.pdb_id, gn_file)
        gn_global = parse_geneonet_csv(gn_file) if use_gn else []
        if not use_gn:
            self.logger.info(f"{s_cfg.conformation_key}: GeneoNet disabled, using P2Rank only.")

        return p2rank_global, gn_global

    def process_structure(self,
                          s_cfg: StructureConfig,
                          expert_df: pd.DataFrame) -> Optional[List[dict]]:
        structure, pdb_path = self.load_structure(s_cfg)
        if structure is None:
            return None

        self.logger.info(f"Processing {s_cfg.conformation_key} (PDB: {s_cfg.pdb_id_lower}).")
        p2rank_global, gn_global = self.load_global_predictors(s_cfg, pdb_path)

        chain_pockets: List[dict] = []
        for chain_id in s_cfg.chains:
            self.logger.info(f"  > Analyzing Chain {chain_id}")

            add_expert_pockets(
                chain_pockets=chain_pockets,
                structure=structure,
                chain_id=chain_id,
                expert_df=expert_df,
                protein_key=s_cfg.protein_key,
                conformation=s_cfg.conformation,
                buffer_size=self.pa_cfg.buffer_size,
                p2rank_global=p2rank_global,
                logger=self.logger
            )

            add_geneonet_pockets(
                chain_pockets=chain_pockets,
                structure=structure,
                chain_id=chain_id,
                gn_global=gn_global,
                geneonet_top_n=self.pa_cfg.geneonet_top_n,
                buffer_size=self.pa_cfg.buffer_size,
                overlap_threshold=self.pa_cfg.overlap,
                p2rank_global=p2rank_global,
                logger=self.logger
            )

            add_p2rank_pockets(
                chain_pockets=chain_pockets,
                structure=structure,
                chain_id=chain_id,
                p2rank_global=p2rank_global,
                p2rank_top_n=self.pa_cfg.p2rank_top_n,
                buffer_size=self.pa_cfg.buffer_size,
                overlap_threshold=self.pa_cfg.overlap,
                logger=self.logger
            )

        return chain_pockets

    def run(self, structures: Dict[str, StructureConfig]) -> Dict[str, List[dict]]:
        self.logger.info("Start parsing of Pockets:")
        expert_df = load_expert_df(self.pa_cfg.pockets_csv)

        final_grids: Dict[str, List[dict]] = {}
        for _, s_cfg in structures.items():
            pockets = self.process_structure(s_cfg, expert_df)
            if pockets is None:
                continue
            final_grids[f"{s_cfg.conformation_key}"] = pockets

        return final_grids

    def save(self, final_grids: Dict[str, List[dict]]):
        with open(self.pa_cfg.grids_file, "w") as f:
            json.dump(final_grids, f, indent=4)
        self.logger.info(f"Saved grids to {self.pa_cfg.grids_file}")


def main():
    logger = setup_logging()

    loader = ConfigLoader("config.yaml")
    project_cfg = loader.project()
    pa_cfg = loader.pocket_analysis()
    structures = loader.structures()

    pipeline = PocketPipeline(project_cfg, pa_cfg, logger)
    final_grids = pipeline.run(structures)
    pipeline.save(final_grids)


if __name__ == "__main__":
    main()

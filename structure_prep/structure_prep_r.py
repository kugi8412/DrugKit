#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# structure_prep/structure_prep_r.py

import os
import shutil
import subprocess
import logging.handlers

from typing import List, Union, Set, Dict

from Bio.PDB import PDBIO, Select, NeighborSearch


from pocket_logic.logging_utils import setup_logging
from pocket_logic.config_loader import ConfigLoader, StructureConfig, ProjectConfig
from pocket_logic.structure_io import resolve_pdb_source, load_structure, check_alphaknot


# =========================
# Structure helpers
# =========================

def parse_target_chains(chain_conf: Union[str, List[str], None]) -> Set[str]:
    """Helper for parsing chain config field."""
    if not chain_conf:
        return set()
    if isinstance(chain_conf, str):
        return {chain_conf}
    if isinstance(chain_conf, list):
        return set(chain_conf)
    return set()


def annotate_bfactor(structure, target_chains: Set[str], logger: logging.Logger) -> None:
    """
    Target Chains -> B-factor = 100.0
    Other Chains  -> B-factor = 0.0
    """
    if not target_chains:
        logger.info("No target chains defined. Skipping B-factor annotation.")
        return

    logger.info(f"Annotating B-factor for chains: {target_chains}.")
    for model in structure:
        for chain in model:
            new_bfactor = 100.0 if chain.id in target_chains else 0.0
            for atom in chain.get_atoms():
                atom.set_bfactor(new_bfactor)


def find_bridging_waters(structure,
                         dist_cutoff: float,
                         logger: logging.Logger) -> Set:
    """
    Identifies waters that form hydrogen bonds between protein residues.
    Returns a set of residue full_id waters that should be retained.
    """
    logger.info("Analyzing structural waters (H-bonds).")

    protein_atoms = []
    water_atoms = []

    for atom in structure.get_atoms():
        res = atom.get_parent()
        if res.id[0] == " ":  # Protein
            protein_atoms.append(atom)
        elif res.id[0] == "W" or res.get_resname() in ["HOH", "WAT"]:  # Water
            if atom.element == "O":
                water_atoms.append(atom)

    if not water_atoms:
        return set()

    ns = NeighborSearch(protein_atoms)
    bridging_waters = set()

    for w_atom in water_atoms:
        water_res = w_atom.get_parent()
        neighbors = ns.search(w_atom.get_coord(), dist_cutoff)
        contact_residues = {n_atom.get_parent() for n_atom in neighbors}

        # "bridging" if contacts >= 2 residues
        if len(contact_residues) >= 2:
            bridging_waters.add(water_res.get_full_id())

    logger.info(f"Found {len(bridging_waters)} structural waters acting as bridges.")
    return bridging_waters


class CleanSelect(Select):
    """
    Selects:
      - standard residues
      - important ions
      - waters ONLY if they are bridging waters
    """
    important_ions = ["NA", "SOD", "CL", "CLA", "ZN", "MG", "CA", "MN", "K"]

    def __init__(self, bridging_waters_ids=None) -> None:
        self.bridging_waters_ids = bridging_waters_ids if bridging_waters_ids else set()

    def accept_residue(self, residue):
        resname = residue.get_resname().strip().upper()

        if resname in ("HOH", "WAT", "H2O"):
            return residue.get_full_id() in self.bridging_waters_ids

        # HETATM
        if residue.id[0].strip() != "":
            return resname in self.important_ions

        return True


def save_clean_pdb(structure,
                   output_pdb: str,
                   hydrogen_cutoff: float,
                   logger: logging.Logger) -> bool:
    """Clean structure and save PDB with CleanSelect (bridging waters + ions)."""
    io = PDBIO()
    io.set_structure(structure)

    structural_waters = find_bridging_waters(structure, dist_cutoff=hydrogen_cutoff, logger=logger)
    try:
        io.save(output_pdb, CleanSelect(bridging_waters_ids=structural_waters))
        logger.info(f"Cleaned structure saved to {output_pdb}")
        return True
    except Exception as e:
        logger.error(f"Error saving cleaned PDB: {e}")
        return False


def convert_pdb_to_pdbqt(output_pdb: str, logger: logging.Logger) -> None:
    """Convert PDB to PDBQT via OpenBabel (obabel)."""
    output_pdbqt = output_pdb.replace(".pdb", ".pdbqt")
    obabel_exec = shutil.which("obabel")
    if not obabel_exec:
        logger.critical("[Obabel] not found. Install via: conda install openbabel")
        return

    # Add polar hydrogens (-xp), remove non-polar Hs (-xn?) and ??? (kept as-is)
    cmd = [obabel_exec, output_pdb, "-O", output_pdbqt, "-xr", "-xn", "-xp"]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            logger.info(f"Successfully converted to {output_pdbqt}")
        else:
            logger.error(f"[OpenBabel] Error:\n{result.stderr}")
    except Exception as e:
        logger.error(f"[Subprocess] Failed: {e}")


# =========================
# Pipeline
# =========================

class StructurePrepPipeline:
    def __init__(self, project_cfg: ProjectConfig, logger: logging.Logger):
        self.project_cfg = project_cfg
        self.logger = logger
        os.makedirs(self.project_cfg.data_dir, exist_ok=True)

    def process_one(self, s_cfg: StructureConfig) -> None:
        target_name = s_cfg.conformation_key
        self.logger.info(f"Starting processing for {s_cfg.protein_key}/{s_cfg.conformation_key}")

        if s_cfg.uniprot_id:
            check_alphaknot(s_cfg.uniprot_id, logger=self.logger)

        pdb_file = resolve_pdb_source(
            pdb_id=s_cfg.pdb_id,
            pdb_path=s_cfg.pdb_path,
            data_dir=self.project_cfg.data_dir,
            logger=self.logger
        )
        if not pdb_file:
            return

        structure = load_structure(pdb_file, structure_id=target_name, logger=self.logger)
        if structure is None:
            return

        # Annotate (in-place)
        annotate_bfactor(structure, s_cfg.chains, logger=self.logger)

        # Output path: data_dir/{conformation_key}.pdb
        output_pdb = os.path.join(self.project_cfg.data_dir, f"{target_name}.pdb")

        ok = save_clean_pdb(
            structure=structure,
            output_pdb=output_pdb,
            hydrogen_cutoff=s_cfg.hydrogen_cutoff,
            logger=self.logger
        )
        if not ok:
            return

        convert_pdb_to_pdbqt(output_pdb, logger=self.logger)

    def run(self, structures: Dict[str, StructureConfig]) -> None:
        self.logger.info("Start Preparation of Structures:")
        if not structures:
            self.logger.error("No structures found in config.")
            return

        for _, s_cfg in structures.items():
            self.process_one(s_cfg)


# =========================
# Entry point
# =========================

def main():
    logger = setup_logging()

    loader = ConfigLoader("config.yaml")
    project_cfg = loader.project()
    structures = loader.structures()

    pipeline = StructurePrepPipeline(project_cfg, logger)
    pipeline.run(structures)


if __name__ == "__main__":
    main()

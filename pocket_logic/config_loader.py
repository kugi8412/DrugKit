# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class ProjectConfig:
    data_dir: str = "data"


@dataclass
class PocketAnalysisConfig:
    p2rank_path: str = "p2rank"
    geneonet_path: str = "geneonet"
    pockets_csv: str = "data/pockets.csv"
    buffer_size: float = 4.0
    overlap: float = 5.0
    p2rank_top_n: int = 5
    geneonet_top_n: int = 5
    grids_file: str = "docking_grids.json"


@dataclass
class StructureConfig:
    conformation_key: str
    protein_key: str = ""
    conformation: str = ""
    pdb_id: Optional[str] = None
    pdb_path: Optional[str] = None
    uniprot_id: Optional[str] = None
    hydrogen_cutoff: float = 3.5
    chains: List[str] = field(default_factory=list)

    @property
    def pdb_id_lower(self) -> str:
        return (self.pdb_id or "").lower()

    def resolve_structure_path(self, data_dir: str) -> str:
        if self.pdb_path and os.path.exists(self.pdb_path):
            return os.path.abspath(self.pdb_path)

        candidates = [
            os.path.join(data_dir, f"{self.conformation_key}.pdb"),
            os.path.join(data_dir, f"{self.pdb_id}.pdb") if self.pdb_id else "",
            f"data/{self.conformation_key}.pdb",
            f"output/{self.conformation_key}.pdb",
        ]
        for path in candidates:
            if path and os.path.exists(path) and os.path.getsize(path) > 0:
                return os.path.abspath(path)
        return os.path.abspath(os.path.join(data_dir, f"{self.conformation_key}.pdb"))


def _parse_chains(raw: Dict[str, Any]) -> List[str]:
    chain_val = raw.get("chains", raw.get("chain"))
    if not chain_val:
        return []
    if isinstance(chain_val, str):
        return [chain_val]
    return [str(c) for c in chain_val]


def _structure_from_entry(key: str, raw: Dict[str, Any]) -> StructureConfig:
    conformation_key = str(raw.get("conformation_key") or key)
    protein_key = str(raw.get("protein_key") or raw.get("protein") or "")
    if not protein_key and "_" in key:
        protein_key = key.split("_", 1)[0]

    return StructureConfig(
        conformation_key=conformation_key,
        protein_key=protein_key,
        conformation=str(raw.get("conformation") or ""),
        pdb_id=raw.get("pdb_id"),
        pdb_path=raw.get("pdb_path"),
        uniprot_id=raw.get("uniprot_id"),
        hydrogen_cutoff=float(raw.get("hydrogen_cutoff", 3.5)),
        chains=_parse_chains(raw),
    )


class ConfigLoader:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        if not os.path.exists(config_path):
            alt = os.path.join(os.path.dirname(__file__), "..", config_path)
            if os.path.exists(alt):
                config_path = os.path.abspath(alt)
        with open(config_path, "r", encoding="utf-8") as f:
            self.raw: Dict[str, Any] = yaml.safe_load(f) or {}

    def project(self) -> ProjectConfig:
        section = self.raw.get("project", {}) or {}
        return ProjectConfig(data_dir=str(section.get("data_dir", "data")))

    def pocket_analysis(self) -> PocketAnalysisConfig:
        section = self.raw.get("pocket_analysis", {}) or {}
        return PocketAnalysisConfig(
            p2rank_path=str(section.get("p2rank_path", "p2rank")),
            geneonet_path=str(section.get("geneonet_path", "geneonet")),
            pockets_csv=str(section.get("pockets_csv", "data/pockets.csv")),
            buffer_size=float(section.get("buffer_size", 4.0)),
            overlap=float(section.get("overlap", 5.0)),
            p2rank_top_n=int(section.get("p2rank_top_n", 5)),
            geneonet_top_n=int(section.get("geneonet_top_n", 5)),
            grids_file=str(section.get("grids_file", "docking_grids.json")),
        )

    def structures(self) -> Dict[str, StructureConfig]:
        section = self.raw.get("structures", {}) or {}
        return {key: _structure_from_entry(key, val or {}) for key, val in section.items()}

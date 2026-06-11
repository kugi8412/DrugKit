#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import gzip
import logging
import os
import shutil
import urllib.request
from typing import Optional

from Bio.PDB import PDBParser


def resolve_pdb_source(
    pdb_id: Optional[str],
    pdb_path: Optional[str],
    data_dir: str,
    logger: logging.Logger,
) -> Optional[str]:
    if pdb_path and os.path.exists(pdb_path):
        return os.path.abspath(pdb_path)

    if not pdb_id:
        logger.error("No pdb_id or pdb_path provided.")
        return None

    os.makedirs(data_dir, exist_ok=True)
    local_pdb = os.path.join(data_dir, f"{pdb_id.upper()}.pdb")
    if os.path.exists(local_pdb) and os.path.getsize(local_pdb) > 0:
        return local_pdb

    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        logger.info(f"Downloading PDB {pdb_id} from RCSB...")
        urllib.request.urlretrieve(url, local_pdb)
        return local_pdb
    except Exception as exc:
        logger.error(f"Failed to download PDB {pdb_id}: {exc}")
        return None


def _open_pdb(path: str):
    if path.endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return open(path, "r", encoding="utf-8")


def load_structure(
    pdb_file: str,
    structure_id: str,
    logger: logging.Logger,
):
    try:
        parser = PDBParser(QUIET=True)
        if pdb_file.endswith(".gz"):
            tmp_path = pdb_file[:-3]
            if not os.path.exists(tmp_path):
                with gzip.open(pdb_file, "rb") as src, open(tmp_path, "wb") as dst:
                    shutil.copyfileobj(src, dst)
            pdb_file = tmp_path
        return parser.get_structure(structure_id, pdb_file)
    except Exception as exc:
        logger.error(f"Failed to parse structure {pdb_file}: {exc}")
        return None


def check_alphaknot(uniprot_id: str, logger: logging.Logger) -> None:
    logger.info(f"AlphaKnot check skipped for UniProt {uniprot_id} (not configured).")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os


def ensure_dir(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def safe_filename(s: str) -> str:
    return "".join([c if c.isalnum() else "_" for c in str(s)])

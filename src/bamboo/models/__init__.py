# Copyright (c) 2025, Tom Ouellette
# Licensed under the GNU GPLv3 License

import os

CACHE_KEY: str = "BAMBOO_CACHE"

if CACHE_KEY not in os.environ:
    os.environ[CACHE_KEY] = os.path.expanduser(f"~/.{CACHE_KEY.lower()}")

CACHE_DIR = os.environ[CACHE_KEY]
os.makedirs(CACHE_DIR, exist_ok=True)

from ._semantic import Semantic, SemanticStats

__all__ = ["Semantic", "SemanticStats"]

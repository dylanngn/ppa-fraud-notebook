"""
Minimal Hydra utilities for path resolution and environment loading.

Uses built-in features from Hydra and python-dotenv:
- hydra.utils.to_absolute_path() for path resolution
- dotenv.find_dotenv() for .env discovery
"""
from pathlib import Path
from functools import lru_cache

import hydra.utils


@lru_cache(maxsize=32)
def resolve_path(relative_path: str) -> Path:
    """
    Resolve a relative path to absolute using Hydra's utility.
    
    Args:
        relative_path: Path relative to project root (e.g., "artifacts/graph.pt")
    
    Returns:
        Absolute Path object
    """
    return Path(hydra.utils.to_absolute_path(relative_path))


def load_env():
    """
    Load .env file, searching upward from current directory.
    
    Call at module level BEFORE @hydra.main for env var resolution.
    Uses python-dotenv's find_dotenv() to locate .env automatically.
    """
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())


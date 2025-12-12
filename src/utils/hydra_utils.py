from pathlib import Path

import hydra.utils


def resolve_path(relative_path: str) -> Path:
    return Path(hydra.utils.to_absolute_path(relative_path))


def load_env():
    from dotenv import load_dotenv, find_dotenv
    load_dotenv(find_dotenv())


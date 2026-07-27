"""
instruct_rl/utils/env_loader.py
================================
Utility for reading a .env file into os.environ.
Parses the file directly without the dotenv package.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path | None = None) -> dict[str, str]:
    """Read a .env file line by line and register entries in ``os.environ``.

    Parameters
    ----------
    path : str | Path | None
        Path to the .env file. When None, use .env at the project root
        (three levels above this file).

    Returns
    -------
    dict[str, str]
        Parsed and registered key-value pairs.

    Notes
    -----
    - Ignore blank lines and comment lines beginning with ``#``.
    - Strip matching single or double quotes around values.
    - Do not overwrite keys already in os.environ; system environment values win.
    """
    if path is None:
        # instruct_rl/utils/env_loader.py -> project root
        root = Path(__file__).resolve().parent.parent.parent
        path = root / ".env"
    else:
        path = Path(path)

    loaded: dict[str, str] = {}

    if not path.is_file():
        return loaded

    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            # Ignore blank lines and comments
            if not line or line.startswith("#"):
                continue

            # Support export KEY=VALUE syntax
            if line.startswith("export "):
                line = line[len("export "):].strip()

            if "=" not in line:
                continue

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()

            if not key:
                continue

            # Remove matching quotes
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]

            loaded[key] = value

            # Do not overwrite an existing system environment variable
            if key not in os.environ:
                os.environ[key] = value

    return loaded


def get_wandb_key() -> str | None:
    """Read WANDB_API_KEY from the environment.

    Call load_dotenv() again in case .env has not been loaded yet.
    """
    load_dotenv()
    return os.environ.get("WANDB_API_KEY")

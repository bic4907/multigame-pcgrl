"""
instruct_rl/utils/env_loader.py
================================
.env file  text os.environ  in  text  utility.
dotenv text text  direct parsingtext.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_dotenv(path: str | Path | None = None) -> dict[str, str]:
    """.env file  text text text ``os.environ``  in  text.

    Parameters
    ----------
    path : str | Path | None
        .env file path. None  text text to text text(  file basis 3text textabove) of
        .env   text for text.

    Returns
    -------
    dict[str, str]
        parsing·text key-value text.

    Notes
    -----
    - text text, ``#``  as  starttext  text text  text.
    - text text of  text texttable / texttable  automatic as  text.
    -  text os.environ  in  text  text  text text text(text text text).
    """
    if path is None:
        # instruct_rl/utils/env_loader.py → text to text text
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

            # text text·text text
            if not line or line.startswith("#"):
                continue

            # export KEY=VALUE form text
            if line.startswith("export "):
                line = line[len("export "):].strip()

            if "=" not in line:
                continue

            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()

            if not key:
                continue

            # texttable remove
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
                value = value[1:-1]

            loaded[key] = value

            # text text   text text text text
            if key not in os.environ:
                os.environ[key] = value

    return loaded


def get_wandb_key() -> str | None:
    """WANDB_API_KEY   text in   text.

    .env   text loadtext text  text text to  text text text load_dotenv()   calltext.
    """
    load_dotenv()
    return os.environ.get("WANDB_API_KEY")


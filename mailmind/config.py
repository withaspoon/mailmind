from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict


def _try_load_yaml(path: Path) -> Dict[str, Any] | None:
    try:
        import yaml  # type: ignore

        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return None


def _try_load_json(path: Path) -> Dict[str, Any] | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def load_config() -> Dict[str, Any]:
    """Load config from common locations.

    Priority:
      - ./mailmind.yaml (or .yml / .json)
      - ~/.config/mailmind/config.yaml (or .yml / .json)
      - env overrides: MAILMIND_ROOT, MAILMIND_DB_PATH
    Returns a dict; missing values are fine (defaults handled by callers).
    """

    candidates = [
        Path.cwd() / "mailmind.yaml",
        Path.cwd() / "mailmind.yml",
        Path.cwd() / "mailmind.json",
        Path.home() / ".config/mailmind/config.yaml",
        Path.home() / ".config/mailmind/config.yml",
        Path.home() / ".config/mailmind/config.json",
    ]
    cfg: Dict[str, Any] = {}

    for p in candidates:
        if not p.exists():
            continue
        if p.suffix in (".yaml", ".yml"):
            loaded = _try_load_yaml(p)
        elif p.suffix == ".json":
            loaded = _try_load_json(p)
        else:
            loaded = None
        if loaded is not None:
            cfg.update(loaded)
            break

    # Env overrides
    root = os.getenv("MAILMIND_ROOT")
    if root:
        cfg["root"] = root
    db_path = os.getenv("MAILMIND_DB_PATH")
    if db_path:
        cfg["db_path"] = db_path

    return cfg


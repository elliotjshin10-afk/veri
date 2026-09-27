"""Load `.env` so scripts pick up secrets without exporting them by hand.

Deliberately tiny and dependency-free. Values already set in the real
environment win, so CI or a shell export can override the file.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def load(path: Path | None = None) -> list[str]:
    """Read KEY=VALUE lines into os.environ. Returns the names it set."""
    path = path or (ROOT / ".env")
    if not path.exists():
        return []
    loaded = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
            loaded.append(key)
    return loaded

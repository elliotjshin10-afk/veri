"""Score the JSON tree export the browser ships, from Python.

The nightly ledger must predict with the model the PRODUCT uses, not a sibling
trained in the same run. On Tron those are the same file. On Ethereum the
shipped artefact is site/model_eth.json - LightGBM's trees exported to JSON so
a browser can walk them - and the LightGBM booster beside it carries a different
feature set, so loading that instead would mean the ledger and the site were
answering with different models while claiming to be one.

Hence this: the same walk the browser does, in Python. It is a re-implementation
and therefore a place train/serve skew can hide, which is why
tests/test_browser_model.py holds it to LightGBM's own scores on the parity
sample rather than trusting that it looks right.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


def load(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _walk(node: Mapping[str, Any], x: Sequence[float | None]) -> float:
    while "v" not in node:
        val = x[node["f"]]
        # LightGBM sends a missing value the default direction. None, and NaN,
        # are both missing - the JSON cannot carry NaN, so it arrives as null.
        if val is None or (isinstance(val, float) and math.isnan(val)):
            node = node["l"] if node["d"] else node["r"]
        else:
            node = node["l"] if val <= node["t"] else node["r"]
    return float(node["v"])


def score_row(model: Mapping[str, Any], x: Sequence[float | None]) -> float:
    raw = sum(_walk(t, x) for t in model["trees"])
    return 1.0 / (1.0 + math.exp(-raw))


def score(model: Mapping[str, Any], rows: Sequence[Sequence[float | None]]) -> list[float]:
    return [score_row(model, r) for r in rows]


def band(p: float, thresholds: Mapping[str, float]) -> str:
    if p >= thresholds["high"]:
        return "high"
    if p >= thresholds["elevated"]:
        return "elevated"
    return "ordinary"

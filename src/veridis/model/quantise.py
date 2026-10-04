"""Quantise features before scoring, so two languages take the same branch.

A feature like `dest_forward_ratio` is a ratio of two sums over thousands of
transfers. DuckDB and JavaScript accumulate those in different orders and the
results differ in the last few ulps - around 1e-14 relative. Tree splits are
exact comparisons, so a value landing on one goes left in Python and right in the
browser, and the summed score moves far enough to change the BAND. One address
came back 0.649936 from Python and 0.644155 from the browser, straddling the
0.644543 elevated threshold: the same model, the same features, two different
verdicts. The ledger scores in Python and the site scores in the browser, so that
is a real inconsistency a reader could catch.

Twelve significant digits is far more precision than any of these features carry
meaningfully, and `%.12g` here matches `Number.toPrecision(12)` there exactly -
verified on the values that actually diverged. After this both sides see the same
double.
"""
from __future__ import annotations

import math

import numpy as np

DIGITS = 12


def quantise(v: float) -> float:
    if v is None:
        return v
    f = float(v)
    if math.isnan(f) or math.isinf(f):
        return f
    return float(f"{f:.{DIGITS}g}")


def quantise_matrix(x: np.ndarray) -> np.ndarray:
    """Every finite entry rounded to DIGITS significant figures."""
    out = np.asarray(x, dtype=float).copy()
    finite = np.isfinite(out)
    flat = out[finite]
    out[finite] = np.array([float(f"{v:.{DIGITS}g}") for v in flat], dtype=float)
    return out

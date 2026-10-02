"""The Python walk of the shipped JSON model must equal LightGBM exactly.

The nightly ledger scores Ethereum candidates with site/model_eth.json, walked
in Python. That is a second implementation of someone else's tree format, and
the parity sample exists precisely because a second implementation is where the
serving path quietly diverges from the trained one. The browser is held to the
same fixture by scripts/check_eth_parity.py; this holds Python to it.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from veridis.config import PROCESSED, SITE
from veridis.model.browser_model import band, load, score


def _fixture(model_name: str, sample_name: str):
    m, s = SITE / model_name, PROCESSED / sample_name
    if not m.exists() or not s.exists():
        pytest.skip(f"{model_name} / {sample_name} not built in this environment")
    return load(m), json.loads(s.read_text())


@pytest.mark.parametrize("model_name,sample_name", [
    ("model_eth.json", "eth_parity_sample.json"),
    ("model_pair_eth.json", "eth_pair_parity_sample.json"),
])
def test_python_walk_matches_lightgbm(model_name, sample_name):
    model, sample = _fixture(model_name, sample_name)
    assert sample["features"] == model["features"]
    got = score(model, [r["x"] for r in sample["rows"]])
    worst = max(abs(a - r["p"]) for a, r in zip(got, sample["rows"]))
    assert worst < 1e-9, f"worst score difference {worst:.3e}"


@pytest.mark.parametrize("model_name,sample_name", [
    ("model_eth.json", "eth_parity_sample.json"),
    ("model_pair_eth.json", "eth_pair_parity_sample.json"),
])
def test_bands_agree_too(model_name, sample_name):
    """Numbers near each other are not enough - the band is what a person is
    told, so the labels have to match as labels."""
    model, sample = _fixture(model_name, sample_name)
    thr = model["thresholds"]
    got = score(model, [r["x"] for r in sample["rows"]])
    flips = [i for i, (a, r) in enumerate(zip(got, sample["rows"]))
             if band(a, thr) != band(r["p"], thr)]
    assert not flips, f"{len(flips)} rows land in a different band"


def test_a_missing_feature_takes_the_default_branch():
    """None must behave as LightGBM's missing, not as zero."""
    model = {"features": ["f0"], "thresholds": {"elevated": 0.5, "high": 0.9},
             "trees": [{"f": 0, "t": 1.0, "d": 1,
                        "l": {"v": 2.0}, "r": {"v": -2.0}}]}
    assert score(model, [[None]])[0] == pytest.approx(score(model, [[0.5]])[0])
    model["trees"][0]["d"] = 0
    assert score(model, [[None]])[0] == pytest.approx(score(model, [[5.0]])[0])

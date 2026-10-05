"""The decided-cases log must describe the model the site actually ships.

site/history.json is a frozen artefact: it records what a particular
model_pair.json said about particular held-out transfers. Retrain that model and
the file is no longer a record of anything the page serves - the verdicts stay,
the model behind them changes, and the page keeps printing a checksum that no
longer matches the file beside it. That is a quiet way to end up publishing
numbers nobody can reproduce, so it fails here instead.

Run `make history` when this test fails.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
BANDS = ("high", "elevated", "ordinary")


@pytest.fixture(scope="module")
def doc() -> dict:
    path = SITE / "history.json"
    if not path.exists():
        pytest.skip("site/history.json not built")
    return json.loads(path.read_text())


def test_cites_the_shipped_model(doc: dict) -> None:
    sha = hashlib.sha256((SITE / "model_pair.json").read_bytes()).hexdigest()[:16]
    assert doc["model"]["sha256_16"] == sha, (
        "history.json was built against a different model_pair.json - "
        "run `make history`")


def test_thresholds_match_the_shipped_model(doc: dict) -> None:
    shipped = json.loads((SITE / "model_pair.json").read_text())["thresholds"]
    assert doc["model"]["thresholds"] == shipped


def test_summary_is_over_the_whole_subset(doc: dict) -> None:
    s = doc["summary"]
    assert s["n"] == s["n_scam"] + s["n_control"]
    assert sum(s[f"scam_{b}"] for b in BANDS) == s["n_scam"]
    assert sum(s[f"control_{b}"] for b in BANDS) == s["n_control"]
    # The log is a sample of the subset, never the whole of it; a page that
    # computed its rates from eighteen rows would be quoting itself.
    assert len(doc["cases"]) < s["n"]


def test_every_case_carries_its_outcome(doc: dict) -> None:
    for c in doc["cases"]:
        assert c["band"] in BANDS
        assert c["why"] and not c["why"].endswith(".")
        if c["scam"]:
            # A positive is an address that was frozen LATER. A freeze at or
            # before the transfer would mean the label leaked into the score.
            assert c["froze"] is not None, c
            assert c["froze"] > c["t"], c
            assert c["lead_days"] > 0
        else:
            assert c["froze"] is None and c["unlisted_days"] > 0


def test_both_kinds_of_mistake_are_shown(doc: dict) -> None:
    """A log of only the wins is marketing, and this file is not that."""
    cases = doc["cases"]
    assert any(c["scam"] and c["band"] == "ordinary" for c in cases), "no misses shown"
    assert any(not c["scam"] and c["band"] != "ordinary" for c in cases), \
        "no false alarms shown"

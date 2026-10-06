"""Guards on the half of the ledger that can fail silently.

Appending predictions is loud: the chain breaks, or the entry count moves. The
resolution arm has no such tell. It reads a freeze list, finds no match, writes
nothing, and prints "0 resolved" - which is indistinguishable from "nothing we
flagged has been frozen yet". That is how it sat for nine days against lists
last fetched before the first prediction was ever made.

Two conditions produce that silence forever, and neither shows up in the
output, so they are asserted here instead.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
TRON = re.compile(r"^T[1-9A-HJ-NP-Za-km-z]{33}$")
EVM = re.compile(r"^0x[0-9a-f]{40}$")       # lower case: the lists store it so


def _load_resolver():
    spec = importlib.util.spec_from_file_location(
        "ledger_resolve", ROOT / "scripts" / "ledger_resolve.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def freezes() -> dict[str, int]:
    return _load_resolver().freeze_times()


@pytest.fixture(scope="module")
def predictions() -> list[dict]:
    path = ROOT / "data" / "ledger" / "predictions.jsonl"
    if not path.exists():
        pytest.skip("no prediction ledger")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_freeze_lists_carry_both_chains(freezes: dict[str, int]) -> None:
    """Ethereum entries used to be resolvable only through the labels parquet,
    which is rebuilt by hand - so in practice, never."""
    assert any(TRON.match(a) for a in freezes), "no Tron freezes loaded"
    assert any(EVM.match(a) for a in freezes), "no Ethereum freezes loaded"


def test_freeze_lists_reach_past_the_oldest_prediction(
        freezes: dict[str, int], predictions: list[dict]) -> None:
    """The condition that made resolution vacuous.

    If the newest freeze on record predates even the oldest prediction, then no
    entry in the ledger can resolve whatever happens on chain, and "0 resolved"
    is a statement about our fetching rather than about the world. `make ledger`
    refreshes the lists before resolving; this fails if that stops happening.
    """
    assert freezes, "no freeze times loaded at all"
    newest_freeze = max(freezes.values())
    oldest_prediction = min(p["predicted_at_ms"] for p in predictions)
    assert newest_freeze > oldest_prediction, (
        "the freeze lists end before the ledger begins - run "
        "scripts/refresh_freezes.py; resolution cannot find anything until then")


def test_addresses_are_spelled_the_same_on_both_sides(
        freezes: dict[str, int], predictions: list[dict]) -> None:
    """A checksummed 0x address never equals a lower-cased one, and the
    mismatch would look exactly like an address that was never frozen."""
    for p in predictions:
        a = p["address"]
        assert TRON.match(a) or EVM.match(a), f"ledger holds {a!r}"
    for a in freezes:
        assert TRON.match(a) or EVM.match(a), f"freeze list holds {a!r}"

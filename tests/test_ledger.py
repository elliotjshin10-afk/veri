"""The ledger's only job is to be un-editable after the fact."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from veridis.ledger import GENESIS, Ledger, entry_hash


@pytest.fixture
def led(tmp_path):
    L = Ledger(tmp_path / "p.jsonl")
    for i in range(5):
        L.append({"address": f"T{i}", "score": 0.9 + i / 100, "predicted_at_ms": 1000 + i})
    return L


def test_chain_links_each_entry_to_the_last(led):
    entries = list(led)
    assert entries[0]["prev"] == GENESIS
    for a, b in zip(entries, entries[1:]):
        assert b["prev"] == a["hash"]
    assert led.verify()[0]


def test_editing_a_value_breaks_the_chain(led):
    lines = led.path.read_text().splitlines()
    rec = json.loads(lines[2])
    rec["score"] = 0.99                      # the edit a motivated author would make
    lines[2] = json.dumps(rec, sort_keys=True, separators=(",", ":"))
    led.path.write_text("\n".join(lines) + "\n")
    ok, msg = Ledger(led.path).verify()
    assert not ok and "T2" in msg


def test_deleting_an_entry_breaks_the_chain(led):
    """Dropping a wrong call is the failure mode that matters most."""
    lines = led.path.read_text().splitlines()
    del lines[1]
    led.path.write_text("\n".join(lines) + "\n")
    assert not Ledger(led.path).verify()[0]


def test_reordering_breaks_the_chain(led):
    lines = led.path.read_text().splitlines()
    lines[1], lines[3] = lines[3], lines[1]
    led.path.write_text("\n".join(lines) + "\n")
    assert not Ledger(led.path).verify()[0]


def test_appending_after_a_reload_continues_the_same_chain(led):
    head = led.head
    again = Ledger(led.path)
    rec = again.append({"address": "T9", "score": 0.5})
    assert rec["prev"] == head
    assert Ledger(led.path).verify()[0]


def test_hash_is_independent_of_key_order():
    a = entry_hash(GENESIS, {"x": 1, "y": 2})
    b = entry_hash(GENESIS, {"y": 2, "x": 1})
    assert a == b

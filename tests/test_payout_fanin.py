"""The second hop, held to the browser's copy of it.

These two features are the first in the project that need a third address's
history, which is the reason they were excluded for years: computing them in
Python while defaulting them in the browser is the train/serve skew everything
else here guards against. So the definition is pinned from both ends. The cases
below are the ones where two reasonable implementations drift apart.
"""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys

import polars as pl
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from veridis.features.payout import batch, links, payout_fanin, top_payout  # noqa: E402
COLS = ["chain", "tx_hash", "from_address", "to_address", "amount_usd", "asset",
        "block_time"]


SCHEMA = {"chain": pl.Utf8, "tx_hash": pl.Utf8, "from_address": pl.Utf8,
          "to_address": pl.Utf8, "amount_usd": pl.Float64, "asset": pl.Utf8,
          "block_time": pl.Int64}


def tx(rows) -> pl.DataFrame:
    """An empty frame still needs the schema: a caller with no transfers is a
    real case here, and a frame with no columns fails differently from one with
    no rows."""
    return pl.DataFrame(
        [{"chain": "tron", "tx_hash": f"h{i}", "from_address": f, "to_address": t,
          "amount_usd": float(u), "asset": "USDT", "block_time": int(b)}
         for i, (f, t, u, b) in enumerate(rows)], schema=SCHEMA).select(COLS)


def test_top_payout_is_by_value_not_count():
    t = tx([("A", "P", 100, 10), ("A", "Q", 10, 20), ("A", "Q", 10, 30),
            ("A", "Q", 10, 40)])
    assert top_payout(t, "A", 1_000) == "P"


def test_top_payout_respects_the_cutoff():
    t = tx([("A", "P", 10, 10), ("A", "Q", 999, 500)])
    assert top_payout(t, "A", 100) == "P"      # Q has not happened yet
    assert top_payout(t, "A", 1_000) == "Q"


def test_top_payout_breaks_ties_on_the_address():
    """SQL's ROW_NUMBER picks arbitrarily among equal sums and a browser cannot
    reproduce an arbitrary pick, so the rule is stated in both places."""
    t = tx([("A", "zz", 50, 10), ("A", "aa", 50, 20)])
    assert top_payout(t, "A", 1_000) == "aa"


def test_top_payout_ignores_zero_value():
    t = tx([("A", "P", 0, 10), ("A", "Q", 1, 20)])
    assert top_payout(t, "A", 1_000) == "Q"


def test_fanin_excludes_the_address_that_sent_us_here():
    t = tx([("A", "P", 5, 10), ("X", "P", 5, 20)])
    assert payout_fanin(t, "P", 1_000, "A", False)["payout_fanin_pit"] == 1


def test_fanin_is_point_in_time():
    t = tx([("X", "P", 5, 10), ("Y", "P", 5, 500)])
    assert payout_fanin(t, "P", 100, "A", False)["payout_fanin_pit"] == 1
    assert payout_fanin(t, "P", 1_000, "A", False)["payout_fanin_pit"] == 2


def test_exact_is_false_only_when_the_fetch_stopped_short():
    t = tx([("X", "P", 5, 10), ("Y", "P", 5, 50)])
    # Not truncated: the count is the count, whatever the moment.
    assert payout_fanin(t, "P", 1_000, "A", False)["payout_fanin_exact"] == 1
    # Truncated but the fetch reached past the moment: still exact.
    assert payout_fanin(t, "P", 40, "A", True)["payout_fanin_exact"] == 1
    # Truncated and the moment is beyond what we fetched: a floor, not a count.
    assert payout_fanin(t, "P", 1_000, "A", True)["payout_fanin_exact"] == 0


def test_unknown_payout_reads_as_not_exact():
    """Zero with exact=1 would assert that nobody pays the wallet. Zero with
    exact=0 says we do not know, which is what the browser reports when the
    second fetch fails."""
    assert payout_fanin(tx([]), None, 100, "A", False) == {
        "payout_fanin_pit": 0, "payout_fanin_exact": 0}


def test_batch_matches_the_single_event_form():
    t = tx([("A", "P", 100, 10), ("A", "Q", 1, 20),
            ("X", "P", 5, 5), ("Y", "P", 5, 30), ("Z", "P", 5, 900)])
    probes = pl.DataFrame({"event_id": [0, 1], "destination": ["A", "A"],
                           "event_time": [50, 1_000]})
    got = batch(probes, t, t, {"P": False}).sort("event_id")
    for row in got.iter_rows(named=True):
        when = {0: 50, 1: 1_000}[row["event_id"]]
        one = payout_fanin(t, "P", when, "A", False)
        assert row["payout_fanin_pit"] == one["payout_fanin_pit"], row
        assert row["payout_fanin_exact"] == one["payout_fanin_exact"], row


@pytest.mark.skipif(shutil.which("node") is None, reason="node not installed")
def test_browser_agrees_with_python():
    """The whole reason this module exists. Same rows, both languages."""
    rows = [("A", "P", 100, 10), ("A", "Q", 10, 20), ("A", "zz", 100, 25),
            ("X", "P", 5, 5), ("Y", "P", 5, 30), ("A", "P", 7, 35),
            ("Z", "P", 5, 900)]
    t = tx(rows)
    js_rows = json.dumps([{"from": f, "to": to, "usd": float(u), "t": int(b)}
                          for f, to, u, b in rows])
    driver = f"""
import {{topPayout, payoutFanin}} from "{(ROOT / 'site' / 'scorer.js').as_uri()}";
const tx = {js_rows};
const out = [];
for (const asOf of [40, 100, 1000]) {{
  for (const trunc of [false, true]) {{
    const p = topPayout(tx, "A", asOf);
    out.push([asOf, trunc, p, payoutFanin(tx, p, asOf, "A", trunc)]);
  }}
}}
console.log(JSON.stringify(out));
"""
    res = subprocess.run(["node", "--input-type=module", "-e", driver],
                         capture_output=True, text=True, check=True)
    for as_of, trunc, js_payout, js_fanin in json.loads(res.stdout):
        py_payout = top_payout(t, "A", as_of)
        assert js_payout == py_payout, (as_of, js_payout, py_payout)
        py = payout_fanin(t, py_payout, as_of, "A", trunc)
        assert js_fanin == py, (as_of, trunc, js_fanin, py)

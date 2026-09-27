"""Adversarial stress test mechanics.

The failure that matters here is silent: an attack that rewrites transfers can
break the code that identifies which transfers were attacked, and the result
reads as "detection fell to zero" when nothing was scored at all. That happened
during development with the fresh-address attack, so it is pinned.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from veridis.model.evasion import (  # noqa: E402
    VICTIM_FLAG, deround, fresh_destination, slow_cadence, split_transfers,
    tag_victim_transfers,
)

DAY = 86_400_000
T0 = 1_700_000_000_000


def _tx():
    rows = [("victim", "scam", 1000.0, T0), ("victim", "scam", 5000.0, T0 + DAY),
            ("other", "shop", 40.0, T0 + 2 * DAY)]
    return pl.DataFrame(
        [{"chain": "tron", "tx_hash": f"h{i}", "from_address": f, "to_address": t,
          "amount_usd": a, "asset": "USDT", "block_time": b}
         for i, (f, t, a, b) in enumerate(rows)])


def _pairs():
    return pl.DataFrame({"from_address": ["victim"], "to_address": ["scam"]})


def test_tag_marks_exactly_the_attacked_transfers():
    out = tag_victim_transfers(_tx(), _pairs())
    assert out[VICTIM_FLAG].to_list() == [True, True, False]


def test_fresh_address_attack_keeps_transfers_identifiable():
    """The bug this exists to prevent: renaming the destination used to break
    the join that found victim transfers, emptying the positive set."""
    tagged = tag_victim_transfers(_tx(), _pairs())
    out = fresh_destination(tagged, _pairs())
    victims = out.filter(pl.col(VICTIM_FLAG))
    assert victims.height == 2, "attacked transfers must still be findable"
    assert all(d != "scam" for d in victims["to_address"].to_list())
    # untouched traffic is untouched
    assert out.filter(~pl.col(VICTIM_FLAG))["to_address"].to_list() == ["shop"]


def test_split_preserves_total_value_and_multiplies_transfers():
    tagged = tag_victim_transfers(_tx(), _pairs())
    out = split_transfers(tagged, _pairs(), n_parts=4)
    v = out.filter(pl.col(VICTIM_FLAG))
    assert v.height == 8
    assert abs(v["amount_usd"].sum() - 6000.0) < 1e-6
    assert out.filter(~pl.col(VICTIM_FLAG)).height == 1


def test_slow_cadence_stretches_gaps_but_keeps_the_start():
    tagged = tag_victim_transfers(_tx(), _pairs())
    out = slow_cadence(tagged, _pairs(), factor=5.0).filter(pl.col(VICTIM_FLAG)).sort("block_time")
    times = out["block_time"].to_list()
    assert times[0] == T0
    assert times[1] - times[0] == 5 * DAY


def test_deround_moves_amounts_off_round_numbers():
    tagged = tag_victim_transfers(_tx(), _pairs())
    out = deround(tagged, _pairs()).filter(pl.col(VICTIM_FLAG))
    assert all(a % 1000 != 0 for a in out["amount_usd"].to_list())


def test_attacks_never_touch_traffic_they_do_not_own():
    """A scammer cannot change how legitimate users behave."""
    tagged = tag_victim_transfers(_tx(), _pairs())
    for fn in (split_transfers, slow_cadence, deround, fresh_destination):
        out = fn(tagged, _pairs())
        clean = out.filter(~pl.col(VICTIM_FLAG)).sort("tx_hash")
        assert clean.height == 1
        assert clean["amount_usd"].to_list() == [40.0]
        assert clean["block_time"].to_list() == [T0 + 2 * DAY]

"""The behavioural fraud-modelling features, and the ablation's separation.

These features exist to make the sender-side half of the product real, so the
things worth testing are that they compute the quantity they claim and that the
ablation arms genuinely cannot see the destination.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from veridis.features.asof import FEATURE_COLUMNS, FEATURE_FAMILY, compute_features  # noqa: E402
from veridis.model.ablation import ARMS, _cols  # noqa: E402

DAY = 86_400_000
T0 = 1_700_000_000_000


def _tx(rows):
    return pl.DataFrame(
        [{"chain": "tron", "tx_hash": f"t{i}", "from_address": f, "to_address": t,
          "amount_usd": float(a), "asset": "USDT", "block_time": int(bt)}
         for i, (f, t, a, bt) in enumerate(rows)],
        schema={"chain": pl.Utf8, "tx_hash": pl.Utf8, "from_address": pl.Utf8,
                "to_address": pl.Utf8, "amount_usd": pl.Float64, "asset": pl.Utf8,
                "block_time": pl.Int64})


def _ev(sender, dest, amount, t):
    return pl.DataFrame(
        [{"event_id": 0, "chain": "tron", "sender": sender, "destination": dest,
          "amount_usd": float(amount), "event_time": int(t)}],
        schema={"event_id": pl.Int64, "chain": pl.Utf8, "sender": pl.Utf8,
                "destination": pl.Utf8, "amount_usd": pl.Float64,
                "event_time": pl.Int64})


def _row(transfers, sender="v", dest="d", amount=1000, t=T0):
    return compute_features(_ev(sender, dest, amount, t), _tx(transfers)).row(0, named=True)


def test_dormancy_measures_gap_since_last_outbound():
    r = _row([("v", "shop", 100, T0 - 30 * DAY), ("v", "shop", 100, T0 - 10 * DAY)])
    assert r["sender_dormancy_days"] == pytest.approx(10.0, abs=0.01)


def test_dormancy_is_sentinel_when_sender_never_sent_before():
    r = _row([("exchange", "v", 5000, T0 - 5 * DAY)])
    assert r["sender_dormancy_days"] == -1.0


def test_amount_zscore_is_against_the_senders_own_baseline():
    """A $5,000 transfer is unremarkable for a $5,000-a-week sender."""
    steady = [("v", f"p{i}", 5000, T0 - (i + 1) * DAY) for i in range(8)]
    modest = [("v", f"p{i}", 50, T0 - (i + 1) * DAY) for i in range(8)]
    big_for_steady = _row(steady, amount=5000)["amount_zscore_own"]
    big_for_modest = _row(modest, amount=5000)["amount_zscore_own"]
    assert big_for_modest > big_for_steady
    assert big_for_steady < 3.0


def test_pair_share_of_outflow_detects_converging_on_one_counterparty():
    """All recent outflow to a single payee is the classic APP shape."""
    spread = [("v", f"p{i}", 100, T0 - (i + 1) * DAY) for i in range(5)]
    converged = [("v", "d", 100, T0 - (i + 1) * DAY) for i in range(5)]
    assert _row(spread)["pair_share_of_outflow_7d"] < 0.3
    assert _row(converged)["pair_share_of_outflow_7d"] > 0.9


def test_round_number_flags():
    base = [("v", "d", 100, T0 - 5 * DAY)]
    assert _row(base, amount=5000)["amount_round_1k"] == 1
    assert _row(base, amount=5000)["amount_round_100"] == 1
    assert _row(base, amount=4317.22)["amount_round_1k"] == 0
    assert _row(base, amount=4317.22)["amount_round_100"] == 0


def test_passthrough_detects_money_arriving_then_leaving():
    held = [("exchange", "v", 10_000, T0 - 3 * DAY)]
    passed = [("exchange", "v", 10_000, T0 - 3 * DAY), ("v", "x", 9_800, T0 - 2 * DAY)]
    assert _row(held)["sender_passthrough_7d"] < 0.1
    assert _row(passed)["sender_passthrough_7d"] > 0.9


def test_new_features_are_all_registered_with_a_family():
    for f in ["amount_zscore_own", "sender_dormancy_days", "pair_share_of_outflow_7d",
              "sender_drawdown_7d", "sender_passthrough_7d", "amount_round_1k"]:
        assert f in FEATURE_COLUMNS
        assert f in FEATURE_FAMILY


def test_sender_side_arms_cannot_see_the_destination():
    """The differentiation claim rests on this, so assert it rather than trust it."""
    for arm in ("sender_side_only", "sender_behaviour_only"):
        cols = _cols(ARMS[arm])
        assert cols, f"{arm} has no features"
        assert not any(FEATURE_FAMILY[c] == "destination" for c in cols)
        assert not any(c.startswith("dest_") for c in cols)


def test_destination_only_arm_sees_no_sender_behaviour():
    cols = _cols(ARMS["destination_only"])
    assert not any(FEATURE_FAMILY[c] in ("sender", "relationship") for c in cols)
    assert any(c.startswith("dest_") for c in cols)


def test_every_arm_is_a_strict_subset_of_full():
    full = set(_cols(ARMS["full"]))
    for arm, fams in ARMS.items():
        assert set(_cols(fams)) <= full


def test_acceleration_detects_a_change_in_cadence():
    """A wallet moving at its usual pace differs from one speeding up."""
    steady = [("v", f"p{i}", 100, T0 - (i + 1) * 3 * DAY) for i in range(10)]
    # same wallet history, then everything crammed into the last few days
    spiking = ([("v", f"q{i}", 100, T0 - (20 + i) * DAY) for i in range(6)]
               + [("v", f"r{i}", 100, T0 - (i + 1) * DAY) for i in range(6)])
    assert _row(spiking)["sender_outflow_accel"] > _row(steady)["sender_outflow_accel"]


def test_rate_vs_lifetime_is_near_one_for_a_steady_wallet():
    steady = [("v", f"p{i}", 100, T0 - (i + 1) * DAY) for i in range(30)]
    r = _row(steady)["sender_rate_vs_lifetime"]
    assert 0.5 < r < 2.5, f"steady wallet should sit near its own baseline, got {r}"


def test_inflow_cover_spots_topping_up_to_keep_sending():
    """Victims are often told to deposit more so they can keep transferring."""
    no_topup = [("v", "d", 5000, T0 - 2 * DAY)]
    topped_up = [("exchange", "v", 20_000, T0 - 3 * DAY), ("v", "d", 5000, T0 - 2 * DAY)]
    assert _row(topped_up)["sender_inflow_cover_7d"] > _row(no_topup)["sender_inflow_cover_7d"]


def test_graph_features_are_computed_but_excluded_from_the_model():
    """Standard AML mule-network features, deliberately kept out.

    They cannot work on this warehouse: the second hop is mostly unfetched, so
    `payout_fanin` counts how much of the payout wallet we happened to collect.
    And the artefact aligns with the label, because the control population was
    built as an interconnected two-hop subgraph. A model given these would
    partly learn how an address entered our dataset.
    """
    from veridis.features.asof import GRAPH_FEATURES

    assert GRAPH_FEATURES, "the features should still be computed"
    assert not set(GRAPH_FEATURES) & set(FEATURE_COLUMNS), (
        "graph features must not reach the model while the second hop is unfetched")
    for f in GRAPH_FEATURES:
        assert f not in FEATURE_FAMILY


def test_graph_features_are_still_produced_for_inspection():
    """Excluded from the model, but kept available so the decision is revisitable."""
    from veridis.features.asof import GRAPH_FEATURES

    tx = [("a", "d", 100, T0 - 10 * DAY), ("d", "payout", 95, T0 - 9 * DAY),
          ("b", "payout", 400, T0 - 8 * DAY), ("c", "payout", 300, T0 - 7 * DAY),
          ("v", "d", 500, T0 - 5 * DAY)]
    row = _row(tx, sender="v", dest="d", amount=1000)
    for f in GRAPH_FEATURES:
        assert f in row, f"{f} should still be computed"
    # three wallets fed the payout address before the event
    assert row["payout_fanin"] == 3

"""The M3 gate: prove the feature layer cannot see the future.

The decisive test is not reading the SQL, it is behavioural: compute features,
then append transfers dated at or after each event - chosen so they would move
every aggregate a lot if they were visible - recompute, and require the feature
matrix to be bit-identical.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from veridis.features.asof import FEATURE_COLUMNS, compute_features  # noqa: E402

DAY = 86_400_000
T0 = 1_700_000_000_000


def _transfers(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "chain": "tron",
                "tx_hash": f"tx{i}",
                "from_address": f,
                "to_address": t,
                "amount_usd": float(a),
                "asset": "USDT",
                "block_time": int(bt),
            }
            for i, (f, t, a, bt) in enumerate(rows)
        ],
        schema={
            "chain": pl.Utf8, "tx_hash": pl.Utf8, "from_address": pl.Utf8,
            "to_address": pl.Utf8, "amount_usd": pl.Float64, "asset": pl.Utf8,
            "block_time": pl.Int64,
        },
    )


def _events(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "event_id": i,
                "chain": "tron",
                "sender": s,
                "destination": d,
                "amount_usd": float(a),
                "event_time": int(bt),
            }
            for i, (s, d, a, bt) in enumerate(rows)
        ],
        schema={
            "event_id": pl.Int64, "chain": pl.Utf8, "sender": pl.Utf8,
            "destination": pl.Utf8, "amount_usd": pl.Float64,
            "event_time": pl.Int64,
        },
    )


BASE_TRANSFERS = [
    ("victim1", "scam", 100, T0 - 10 * DAY),
    ("victim1", "scam", 500, T0 - 5 * DAY),
    ("other1", "scam", 300, T0 - 4 * DAY),
    ("scam", "collector", 900, T0 - 3 * DAY),
    ("exchange", "victim1", 9000, T0 - 20 * DAY),
    ("victim1", "friend", 50, T0 - 30 * DAY),
]
EVENTS = [("victim1", "scam", 5000, T0)]

# Transfers that would massively inflate destination stats if visible: 40 new
# senders, a huge inbound, and a same-millisecond tie.
FUTURE_TRANSFERS = (
    [(f"latevictim{i}", "scam", 2000, T0 + (i + 1) * DAY) for i in range(40)]
    + [("victim1", "scam", 99_999, T0 + DAY)]
    + [("tievictim", "scam", 7777, T0)]          # exact tie: must be excluded
    + [("scam", "collector", 80_000, T0 + 2 * DAY)]
)


def _feature_matrix(transfers: pl.DataFrame) -> pl.DataFrame:
    feats = compute_features(_events(EVENTS), transfers)
    return feats.select(["event_id"] + FEATURE_COLUMNS).sort("event_id")


def test_future_transfers_do_not_change_features():
    past_only = _feature_matrix(_transfers(BASE_TRANSFERS))
    with_future = _feature_matrix(_transfers(BASE_TRANSFERS + FUTURE_TRANSFERS))
    assert past_only.equals(with_future), (
        "feature values changed when future transfers were added - the feature "
        "layer is reading data it could not have known at scoring time"
    )


def test_same_timestamp_transfer_is_excluded():
    """A transfer in the same millisecond as the event is not knowable."""
    base = _feature_matrix(_transfers(BASE_TRANSFERS))
    tied = _feature_matrix(_transfers(BASE_TRANSFERS + [("tie2", "scam", 5000, T0)]))
    assert base.equals(tied)


def test_features_reflect_only_past_state():
    """Sanity-check the values themselves, not just their stability."""
    feats = compute_features(_events(EVENTS), _transfers(BASE_TRANSFERS))
    row = feats.row(0, named=True)
    assert row["sender_prior_sends_to_dest"] == 2          # 100 and 500
    assert row["pair_max_prior_usd"] == 500
    assert row["escalation_ratio"] == pytest.approx(10.0)  # 5000 / 500
    assert row["dest_senders_all"] == 2                    # victim1, other1
    assert row["dest_age_days"] == pytest.approx(10.0)
    assert row["shared_counterparties"] == 0


def test_no_feature_column_is_constant_across_time_shift():
    """Shifting the event later must let strictly more history in."""
    t_early = _feature_matrix(_transfers(BASE_TRANSFERS))
    later_events = _events([("victim1", "scam", 5000, T0 + 10 * DAY)])
    t_late = compute_features(
        later_events, _transfers(BASE_TRANSFERS + FUTURE_TRANSFERS)
    )
    assert t_late.row(0, named=True)["dest_senders_all"] > t_early.row(0, named=True)["dest_senders_all"]

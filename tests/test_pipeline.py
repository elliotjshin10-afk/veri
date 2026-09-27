"""Structural guarantees the pipeline must keep, independent of the data."""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from veridis.chain.address import hex_to_base58, base58_to_hex, is_tron_address  # noqa: E402
from veridis.dataset.matching import add_strata, match_controls  # noqa: E402
from veridis.features.asof import FEATURE_COLUMNS, FEATURE_FAMILY  # noqa: E402
from veridis.labels.collect import aggregate, finalise, is_collection_like  # noqa: E402
from veridis.model.evaluate import dollars_protected, early_detection, threshold_at_fpr  # noqa: E402
from veridis.model.reasons import render  # noqa: E402

USDT = "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t"


def test_tron_address_roundtrip():
    assert base58_to_hex(USDT) == "41a614f803b6fd780986a42c78ec9c7f77e6ded13c"
    assert hex_to_base58("0x" + base58_to_hex(USDT)[2:]) == USDT


def test_checksum_rejects_lookalike_addresses():
    """Regex-shaped strings that are not real addresses must be rejected.

    14 of 34 'Tron addresses' regexed out of CryptoScamDB were false matches.
    """
    assert is_tron_address(USDT)
    assert not is_tron_address("T3X6DwbzXvEG8yjJcMeau5Xb8Z3T67FcNe")
    assert not is_tron_address("0xdac17f958d2ee523a2206206994597c13d831ec7")


def test_no_feature_leaks_identity():
    """The model must not be able to memorise addresses or timestamps."""
    forbidden = {"sender", "destination", "tx_hash", "event_time", "frozen_at",
                 "label", "group_key", "transfer_k", "chain", "asset"}
    assert forbidden.isdisjoint(FEATURE_COLUMNS)


def test_every_feature_has_a_family():
    assert set(FEATURE_FAMILY) == set(FEATURE_COLUMNS)
    assert {"sender", "destination", "relationship"} <= set(FEATURE_FAMILY.values())


def test_corroboration_requires_two_sources_or_an_authoritative_one():
    reports = pl.DataFrame([
        # one weak community report: not enough on its own
        {"address": "Ta", "chain": "tron", "source": "cryptoscamdb",
         "reported_at": None, "category": "x", "weight": 0.5},
        # two independent weak reports: accepted
        {"address": "Tb", "chain": "tron", "source": "cryptoscamdb",
         "reported_at": None, "category": "x", "weight": 0.5},
        {"address": "Tb", "chain": "tron", "source": "mew_darklist",
         "reported_at": None, "category": "x", "weight": 0.5},
        # single authoritative source: accepted
        {"address": "Tc", "chain": "tron", "source": "tether_freeze",
         "reported_at": 1, "category": "x", "weight": 1.0},
    ])
    out = finalise(aggregate(reports)).sort("address")
    basis = dict(zip(out["address"], out["accepted"]))
    assert basis["Ta"] is False
    assert basis["Tb"] is True
    assert basis["Tc"] is True


def test_collection_profile_rejects_ordinary_wallets():
    assert is_collection_like({
        "distinct_inbound_senders": 40, "inbound_count": 80,
        "outbound_count": 5, "consolidation_ratio": 0.9})
    # a normal wallet: few payers, balanced flow
    assert not is_collection_like({
        "distinct_inbound_senders": 2, "inbound_count": 6,
        "outbound_count": 5, "consolidation_ratio": 0.9})
    # busy but pays many different people: not a collector
    assert not is_collection_like({
        "distinct_inbound_senders": 40, "inbound_count": 50,
        "outbound_count": 45, "consolidation_ratio": 0.2})


def test_threshold_respects_fpr_budget():
    import numpy as np
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(100), np.zeros(10_000)].astype(int)
    s = np.r_[rng.normal(0.8, 0.1, 100), rng.normal(0.2, 0.1, 10_000)]
    thr = threshold_at_fpr(y, s, 0.01)
    assert (s[y == 0] >= thr).mean() <= 0.011


def _seq(scores, amounts):
    return pl.DataFrame({
        "label": [1] * len(scores),
        "sender": ["v"] * len(scores),
        "destination": ["s"] * len(scores),
        "transfer_k": list(range(1, len(scores) + 1)),
        "score": scores,
        "amount_usd": amounts,
    })


def test_early_detection_counts_first_flag_position():
    df = _seq([0.1, 0.9, 0.95], [100.0, 500.0, 5000.0])
    ed = early_detection(df, threshold=0.5, ks=(1, 2, 3)).to_dicts()
    assert ed[0]["rate"] == 0.0   # not caught at k=1
    assert ed[1]["rate"] == 1.0   # caught by k=2
    assert ed[2]["rate"] == 1.0


def test_dollars_protected_excludes_money_already_gone():
    df = _seq([0.1, 0.9, 0.95], [100.0, 500.0, 5000.0])
    d = dollars_protected(df, threshold=0.5)
    # the first $100 was not saved; $500 + $5000 moved at/after the flag
    assert d["protected_usd"] == pytest.approx(5500.0)
    assert d["total_usd"] == pytest.approx(5600.0)


def test_reasons_are_suppressed_when_untrue():
    """A reason must never appear unless the statement it makes is true."""
    assert render("dest_age_days", 4, {}) == "Destination address first seen 4 days ago"
    assert render("dest_age_days", 900, {}) is None          # not a new address
    assert render("is_first_send_to_dest", 0, {}) is None    # they have sent before
    assert render("escalation_ratio", 1.0, {}) is None       # no escalation
    assert "15x" in render("escalation_ratio", 15.0, {})


def test_control_matching_holds_the_ratio_and_keeps_all_positives():
    rows = []
    for i in range(40):
        rows.append({"label": 1, "chain": "tron", "sender_age_days": 10.0,
                     "sender_outbound_count": 5.0, "sender_median_out_usd": 100.0,
                     "amount_usd": 900.0})
    for i in range(2000):
        rows.append({"label": 0, "chain": "tron", "sender_age_days": 10.0,
                     "sender_outbound_count": 5.0, "sender_median_out_usd": 100.0,
                     "amount_usd": 120.0})
    df = pl.DataFrame(rows)
    out = match_controls(df, ratio=10)
    assert int((out["label"] == 1).sum()) == 40
    assert int((out["label"] == 0).sum()) == 400


def test_strata_are_assigned_to_every_event():
    df = pl.DataFrame({"label": [1], "chain": ["tron"], "sender_age_days": [0.0],
                       "sender_outbound_count": [0.0], "sender_median_out_usd": [0.0],
                       "amount_usd": [10.0]})
    assert add_strata(df)["stratum"].null_count() == 0


def test_coverage_guard_drops_events_past_truncated_history():
    """An event after an address's truncation point must be dropped.

    Otherwise its as-of-time window covers data we never fetched, and since
    truncation correlates with the label, the model could learn fetch depth.
    """
    from veridis.dataset.coverage import coverage_table, filter_covered_events

    transfers = pl.DataFrame({
        "from_address": ["a", "a", "b"],
        "to_address": ["d", "d", "e"],
        "block_time": [100, 200, 300],
    })
    cov = coverage_table(transfers, {"d": True, "a": False, "b": False, "e": False})
    assert dict(zip(cov["address"], cov["truncated"]))["d"] is True

    events = pl.DataFrame({
        "sender": ["a", "a"],
        "destination": ["d", "d"],
        "event_time": [150, 500],   # 150 is covered, 500 is past truncation
        "label": [1, 1],
    })
    out = filter_covered_events(events, cov)
    assert out["event_time"].to_list() == [150]


def test_coverage_guard_keeps_everything_when_nothing_truncated():
    from veridis.dataset.coverage import coverage_table, filter_covered_events

    transfers = pl.DataFrame({
        "from_address": ["a"], "to_address": ["d"], "block_time": [100],
    })
    cov = coverage_table(transfers, {"a": False, "d": False})
    events = pl.DataFrame({"sender": ["a"], "destination": ["d"],
                           "event_time": [10**12], "label": [0]})
    assert filter_covered_events(events, cov).height == 1


def test_forward_ratio_reason_never_reads_as_nonsense():
    """A ratio slightly above 1 must not render as '120% forwarded out'."""
    assert render("dest_forward_ratio", 0.95, {}).endswith("95% is forwarded out")
    assert render("dest_forward_ratio", 1.2, {}).endswith("100% is forwarded out")
    assert render("dest_forward_ratio", 4.0, {}) is None   # out of plausible range
    assert render("dest_forward_ratio", 0.2, {}) is None   # not a sweeping address


def test_unhealthy_base_rate_is_flagged_not_silently_reported():
    """A thin control pool inflates PR-AUC; the run must say so.

    Observed in practice: a larger victim fetch landed before its matching
    control fetch, giving a 75% base rate, and the pipeline happily reported
    "PR-AUC 0.998" for what was a 1.3x lift.
    """
    import numpy as np
    from veridis.model.train import UNHEALTHY_BASE_RATE
    from veridis.model.evaluate import core_metrics

    y = np.r_[np.ones(75), np.zeros(25)].astype(int)
    s = np.r_[np.full(75, 0.9), np.full(25, 0.1)]
    m = core_metrics(y, s, 0.5)
    assert m["base_rate"] > UNHEALTHY_BASE_RATE
    # PR-AUC looks superb while the lift over base rate is tiny - which is
    # exactly why base rate is reported beside it everywhere.
    assert m["pr_auc"] > 0.95
    assert m["pr_auc"] / m["base_rate"] < 1.5

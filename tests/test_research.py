"""The address-research verdict.

This is what someone sees when they look up a wallet, so the failure that
matters is calling ordinary infrastructure a scam.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from veridis.api.live import collection_verdict  # noqa: E402


def _profile(**kw):
    base = dict(senders_all=3, inbound_count=5, outbound_count=4, inbound_usd=5_000.0,
                outbound_usd=4_000.0, age_days=200.0, forward_ratio=0.5,
                consolidation_ratio=0.2, median_hold_secs=None, usd_per_sender=1_600.0)
    base.update(kw)
    return base


def test_exchange_hot_wallet_is_not_called_a_collector():
    """Observed failure: a real Binance hot wallet tripped four shape rules.

    It has the same shape as a collector - many payers, few payees, everything
    forwarded - and is separated only by scale.
    """
    hot = _profile(senders_all=33, inbound_count=75, outbound_count=17,
                   inbound_usd=20_376_531_398.0, outbound_usd=20_376_531_394.0,
                   age_days=2367.8, forward_ratio=1.0, consolidation_ratio=0.79,
                   usd_per_sender=617_470_648.0)
    verdict, notes = collection_verdict(hot)
    assert "institutional" in verdict
    assert "retail collection address" not in verdict
    assert any("institutional scale" in n for n in notes)


def test_retail_collection_address_is_still_caught():
    scam = _profile(senders_all=47, inbound_count=94, outbound_count=6,
                    inbound_usd=180_000.0, outbound_usd=178_000.0, age_days=12.0,
                    forward_ratio=0.99, consolidation_ratio=0.95,
                    median_hold_secs=180.0, usd_per_sender=3_829.0)
    verdict, _ = collection_verdict(scam)
    assert verdict == "behaves like a collection address"


def test_ordinary_wallet_is_not_flagged():
    verdict, _ = collection_verdict(_profile())
    assert "no collection" in verdict


def test_long_lived_address_needs_more_than_shape():
    """A years-old address with collector shape is usually a service."""
    old = _profile(senders_all=90, inbound_count=200, outbound_count=60,
                   inbound_usd=400_000.0, age_days=800.0, forward_ratio=0.9,
                   consolidation_ratio=0.3, usd_per_sender=4_400.0)
    verdict, notes = collection_verdict(old)
    assert verdict != "behaves like a collection address"
    assert any("years" in n for n in notes)


def test_every_observation_is_a_checkable_statement():
    """Each note must cite a number, not a judgement."""
    scam = _profile(senders_all=47, inbound_count=94, outbound_count=6,
                    inbound_usd=180_000.0, age_days=12.0, forward_ratio=0.99,
                    consolidation_ratio=0.95, median_hold_secs=180.0,
                    usd_per_sender=3_829.0)
    _, notes = collection_verdict(scam)
    assert notes
    assert all(any(ch.isdigit() for ch in n) for n in notes)


def test_counterparty_only_addresses_are_not_treated_as_indexed():
    """Appearing in the warehouse is not the same as having been fetched.

    Observed: an address we had never fetched showed up once as somebody
    else's counterparty, and profiling it from that single transfer reported
    "100% of outflow to a single address" with total confidence. Fetched
    properly, the same address turned out to be $30bn of exchange
    infrastructure.
    """
    import polars as pl
    from veridis.api.live import LiveWarehouse

    transfers = pl.DataFrame(
        [{"chain": "tron", "tx_hash": "t1", "from_address": "counterparty",
          "to_address": "indexed_addr", "amount_usd": 1000.0, "asset": "USDT",
          "block_time": 1_700_000_000_000}],
        schema={"chain": pl.Utf8, "tx_hash": pl.Utf8, "from_address": pl.Utf8,
                "to_address": pl.Utf8, "amount_usd": pl.Float64, "asset": pl.Utf8,
                "block_time": pl.Int64})
    wh = LiveWarehouse(transfers, indexed={"indexed_addr"})
    try:
        assert wh.knows("indexed_addr")
        assert not wh.knows("counterparty"), (
            "a counterparty we never fetched must not count as indexed")
    finally:
        wh.engine.close()

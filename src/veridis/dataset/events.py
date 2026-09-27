"""M1/M2 - victims, controls, and the scoring-event table.

A "scoring event" is one transfer we could have been asked to score before it
executed. Positives are transfers from a victim to a confirmed scam address;
each transfer is its own event, which is what makes early detection (flagged at
transfer k?) measurable. Negatives are transfers by matched non-victim senders.
"""
from __future__ import annotations

import logging

import polars as pl

from veridis.config import MIN_VICTIM_SEND_USD, SEED

log = logging.getLogger(__name__)

DAY_MS = 86_400_000

# A retail scam collection address, as opposed to laundering or exchange
# infrastructure that Tether also freezes. Without this filter the "victims" of
# a $99M address are other criminal wallets, which corrupts the behavioural
# signature we are trying to learn.
RETAIL_MIN_SENDERS = 5
RETAIL_MAX_SENDERS = 5_000
RETAIL_MAX_USD_PER_SENDER = 250_000.0
RETAIL_MIN_INBOUND_USD = 500.0

# Cap how many events one victim relationship may contribute.
#
# Without it, a single relationship with 30 small recurring transfers supplies
# 30 events while a textbook pig-butchering case ($9 test transfer, then
# $3,959) supplies 4 - so event-level learning is dominated by the long,
# non-escalating relationships and the escalation signal is averaged away.
# Early detection only ever asks about k = 1..5, so nothing the metric needs
# is lost.
MAX_EVENTS_PER_PAIR = 10


def profile_destinations(transfers: pl.DataFrame, addresses: pl.Series) -> pl.DataFrame:
    """Whole-history profile of candidate scam addresses.

    Used only for *label vetting* (deciding which addresses are retail
    collection points), never as a model feature - model features go through
    the as-of-time layer instead.
    """
    inb = (
        transfers.filter(pl.col("to_address").is_in(addresses.implode()))
        .group_by("to_address")
        .agg(
            pl.len().alias("inbound_count"),
            pl.col("from_address").n_unique().alias("distinct_inbound_senders"),
            pl.col("amount_usd").sum().alias("inbound_usd"),
            pl.col("amount_usd").median().alias("median_inbound_usd"),
            pl.col("block_time").min().alias("first_seen"),
            pl.col("block_time").max().alias("last_inbound"),
        )
        .rename({"to_address": "address"})
    )
    outb = (
        transfers.filter(pl.col("from_address").is_in(addresses.implode()))
        .group_by("from_address")
        .agg(
            pl.len().alias("outbound_count"),
            pl.col("to_address").n_unique().alias("distinct_outbound_dests"),
            pl.col("amount_usd").sum().alias("outbound_usd"),
        )
        .rename({"from_address": "address"})
    )
    top = (
        transfers.filter(pl.col("from_address").is_in(addresses.implode()))
        .group_by(["from_address", "to_address"])
        .agg(pl.col("amount_usd").sum().alias("v"))
        .group_by("from_address")
        .agg((pl.col("v").max() / pl.col("v").sum()).alias("consolidation_ratio"))
        .rename({"from_address": "address"})
    )
    return (
        inb.join(outb, on="address", how="left")
        .join(top, on="address", how="left")
        .with_columns(
            pl.col("outbound_count").fill_null(0),
            pl.col("outbound_usd").fill_null(0.0),
            pl.col("distinct_outbound_dests").fill_null(0),
            pl.col("consolidation_ratio").fill_null(0.0),
            (pl.col("inbound_usd") / pl.col("distinct_inbound_senders"))
                .alias("usd_per_sender"),
        )
    )


def select_retail_collection(profiles: pl.DataFrame) -> pl.DataFrame:
    return profiles.filter(
        (pl.col("distinct_inbound_senders") >= RETAIL_MIN_SENDERS)
        & (pl.col("distinct_inbound_senders") <= RETAIL_MAX_SENDERS)
        & (pl.col("usd_per_sender") <= RETAIL_MAX_USD_PER_SENDER)
        & (pl.col("inbound_usd") >= RETAIL_MIN_INBOUND_USD)
    )


def extract_victims(
    transfers: pl.DataFrame, scam_addresses: pl.DataFrame
) -> pl.DataFrame:
    """Victims = senders into a confirmed scam address, before it was frozen.

    Transfers at or after the freeze are dropped: once Tether freezes the
    address the scam is over, and those flows are not victim behaviour.
    """
    scam = scam_addresses.select(
        pl.col("address").alias("scam_address"),
        pl.col("first_reported_at").alias("frozen_at"),
    )
    sends = (
        transfers.join(
            scam, left_on="to_address", right_on="scam_address", how="inner"
        )
        .filter(pl.col("block_time") < pl.col("frozen_at"))
        .rename({"from_address": "victim_address", "to_address": "scam_address"})
    )
    # A sender that is itself a labelled scam address is infrastructure, not a victim.
    sends = sends.filter(
        ~pl.col("victim_address").is_in(scam_addresses["address"].implode())
    )
    agg = sends.group_by(["victim_address", "scam_address"]).agg(
        pl.col("block_time").min().alias("first_send_at"),
        pl.col("block_time").max().alias("last_send_at"),
        pl.col("amount_usd").sum().alias("total_sent_usd"),
        pl.len().alias("n_sends"),
        pl.col("frozen_at").first(),
    )
    return agg.filter(pl.col("total_sent_usd") >= MIN_VICTIM_SEND_USD)


def build_positive_events(
    transfers: pl.DataFrame, victims: pl.DataFrame
) -> pl.DataFrame:
    """One event per victim->scam transfer, indexed by position in the sequence."""
    pairs = victims.select("victim_address", "scam_address", "frozen_at")
    ev = (
        transfers.join(
            pairs,
            left_on=["from_address", "to_address"],
            right_on=["victim_address", "scam_address"],
            how="inner",
        )
        .filter(pl.col("block_time") < pl.col("frozen_at"))
        .rename({"from_address": "sender", "to_address": "destination"})
    )
    ev = ev.sort(["sender", "destination", "block_time"]).with_columns(
        (pl.col("block_time")).alias("event_time"),
        (pl.int_range(pl.len()).over(["sender", "destination"]) + 1).alias("transfer_k"),
    )
    ev = ev.filter(pl.col("transfer_k") <= MAX_EVENTS_PER_PAIR)
    return ev.select(
        "chain", "sender", "destination", "amount_usd", "asset", "event_time",
        "tx_hash", "transfer_k", "frozen_at",
    ).with_columns(
        pl.lit(1).cast(pl.Int8).alias("label"),
        pl.col("destination").alias("group_key"),  # group = scam operation
    )


def build_control_events(
    transfers: pl.DataFrame,
    exclude_senders: pl.Series,
    scam_addresses: pl.Series,
    n_events: int,
    seed: int = SEED,
) -> pl.DataFrame:
    """Negative events: transfers by senders with no link to any scam address."""
    ev = (
        transfers.filter(
            ~pl.col("from_address").is_in(exclude_senders.implode())
            & ~pl.col("to_address").is_in(scam_addresses.implode())
            & ~pl.col("from_address").is_in(scam_addresses.implode())
        )
        .rename({"from_address": "sender", "to_address": "destination"})
        .with_columns(pl.col("block_time").alias("event_time"))
    )
    if ev.height > n_events:
        ev = ev.sample(n=n_events, seed=seed, shuffle=True)
    return ev.with_columns(
        pl.lit(0).cast(pl.Int8).alias("label"),
        pl.lit(0).cast(pl.Int64).alias("transfer_k"),
        pl.lit(None).cast(pl.Int64).alias("frozen_at"),
        pl.col("sender").alias("group_key"),
    ).select(
        "chain", "sender", "destination", "amount_usd", "asset", "event_time",
        "tx_hash", "transfer_k", "frozen_at", "label", "group_key",
    )

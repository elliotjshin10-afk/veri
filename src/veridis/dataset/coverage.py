"""History-coverage guards.

Histories are fetched oldest-first and capped at a page limit, so a busy
address can be *truncated*: we hold its early life but not its recent activity.
Scoring an event that falls after an address's truncation point would compute
as-of-time features over a window we never fetched - the destination would look
younger and quieter than it was.

That error is not random. Scam destinations are short-lived and fully covered,
while ordinary destinations are long-lived and more often truncated, so the
artefact lines up with the label and the model can learn fetch depth instead of
behaviour. This module drops events whose sender or destination history is not
complete through the event time, for both classes alike.
"""
from __future__ import annotations

import logging

import polars as pl

log = logging.getLogger(__name__)


def coverage_table(transfers: pl.DataFrame, truncated: dict[str, bool]) -> pl.DataFrame:
    """Per address: the time through which its fetched history is complete."""
    last_seen = (
        pl.concat(
            [
                transfers.select(pl.col("from_address").alias("address"), "block_time"),
                transfers.select(pl.col("to_address").alias("address"), "block_time"),
            ],
            how="vertical_relaxed",
        )
        .group_by("address")
        .agg(pl.col("block_time").max().alias("last_fetched"))
    )
    trunc = pl.DataFrame(
        {
            "address": list(truncated.keys()),
            "truncated": list(truncated.values()),
        },
        schema={"address": pl.Utf8, "truncated": pl.Boolean},
    )
    out = last_seen.join(trunc, on="address", how="left").with_columns(
        pl.col("truncated").fill_null(False)
    )
    # A complete history is valid through "now"; a truncated one only through
    # the last transfer we actually hold.
    return out.with_columns(
        pl.when(pl.col("truncated"))
        .then(pl.col("last_fetched"))
        .otherwise(pl.lit(2**62, dtype=pl.Int64))
        .alias("complete_until")
    ).select("address", "truncated", "last_fetched", "complete_until")


def filter_covered_events(events: pl.DataFrame, coverage: pl.DataFrame) -> pl.DataFrame:
    """Keep only events whose sender and destination history covers event_time."""
    before = events.height
    cov_s = coverage.select(
        pl.col("address").alias("sender"),
        pl.col("complete_until").alias("_s_until"),
    )
    cov_d = coverage.select(
        pl.col("address").alias("destination"),
        pl.col("complete_until").alias("_d_until"),
    )
    out = (
        events.join(cov_s, on="sender", how="left")
        .join(cov_d, on="destination", how="left")
        .with_columns(
            pl.col("_s_until").fill_null(2**62),
            pl.col("_d_until").fill_null(2**62),
        )
        .filter(
            (pl.col("event_time") <= pl.col("_s_until"))
            & (pl.col("event_time") <= pl.col("_d_until"))
        )
        .drop("_s_until", "_d_until")
    )
    dropped = before - out.height
    if dropped:
        log.info(
            "coverage guard dropped %d/%d events (%.1f%%) beyond fetched history",
            dropped, before, 100 * dropped / max(before, 1),
        )
        by_label = (
            events.join(cov_d, on="destination", how="left")
            .with_columns(pl.col("_d_until").fill_null(2**62))
            .with_columns((pl.col("event_time") > pl.col("_d_until")).alias("dropped"))
            .group_by(["label", "dropped"]).len().sort(["label", "dropped"])
        )
        log.info("coverage drops by label: %s", by_label.to_dicts())
    return out

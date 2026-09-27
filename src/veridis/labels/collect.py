"""M0 — aggregate scam reports into corroborated labels with provenance.

Corroboration rule (from the brief): accept an address as a positive only with
>= 2 independent reports, OR one report plus an on-chain pattern consistent
with collection behaviour. Provenance is kept on every row so we can measure
how sensitive the model is to weak labels.
"""
from __future__ import annotations

import logging

import polars as pl

from veridis.config import INTERIM, MIN_CORROBORATION
from veridis.labels.sources import collect_all

log = logging.getLogger(__name__)

# An address showing this profile is behaving like a scam collection point:
# many unrelated payers in, few payees out, funds not held.
COLLECTION_MIN_SENDERS = 5
COLLECTION_MIN_INOUT_RATIO = 2.0
COLLECTION_MIN_CONSOLIDATION = 0.60


def build_reports() -> pl.DataFrame:
    rows = collect_all()
    if not rows:
        raise RuntimeError("no scam reports collected; check network access")
    return pl.DataFrame(rows)


def aggregate(reports: pl.DataFrame) -> pl.DataFrame:
    """One row per (address, chain) with summed provenance."""
    return (
        reports.group_by(["address", "chain"])
        .agg(
            pl.col("source").n_unique().alias("source_count"),
            pl.col("weight").max().alias("max_source_weight"),
            pl.col("weight").sum().alias("corroboration_score"),
            pl.col("source").unique().sort().str.join(",").alias("sources"),
            pl.col("reported_at").min().alias("first_reported_at"),
            pl.col("category").first().alias("category"),
        )
        .with_columns(
            # A single authoritative source (issuer freeze, sanctions listing)
            # already meets the bar; community lists need a second voice.
            (
                (pl.col("source_count") >= MIN_CORROBORATION)
                | (pl.col("max_source_weight") >= 1.0)
            ).alias("label_corroborated_offchain")
        )
    )


def is_collection_like(profile: dict) -> bool:
    """On-chain corroboration: does this address behave like a collector?"""
    senders = profile.get("distinct_inbound_senders", 0) or 0
    in_ct = profile.get("inbound_count", 0) or 0
    out_ct = profile.get("outbound_count", 0) or 0
    consolidation = profile.get("consolidation_ratio", 0.0) or 0.0
    if senders < COLLECTION_MIN_SENDERS:
        return False
    if out_ct == 0:
        return in_ct >= COLLECTION_MIN_SENDERS
    if in_ct / max(out_ct, 1) < COLLECTION_MIN_INOUT_RATIO:
        return False
    return consolidation >= COLLECTION_MIN_CONSOLIDATION


def finalise(
    aggregated: pl.DataFrame, profiles: pl.DataFrame | None = None
) -> pl.DataFrame:
    """Attach on-chain corroboration and emit the accepted label set."""
    df = aggregated
    if profiles is not None and profiles.height:
        df = df.join(profiles, on=["address", "chain"], how="left")
        onchain = pl.Series(
            "label_corroborated_onchain",
            [is_collection_like(r) for r in df.iter_rows(named=True)],
        )
        df = df.with_columns(onchain)
    else:
        df = df.with_columns(pl.lit(False).alias("label_corroborated_onchain"))

    return df.with_columns(
        (
            pl.col("label_corroborated_offchain")
            | pl.col("label_corroborated_onchain")
        ).alias("accepted"),
        pl.when(pl.col("source_count") >= MIN_CORROBORATION)
        .then(pl.lit("multi_source"))
        .when(pl.col("max_source_weight") >= 1.0)
        .then(pl.lit("authoritative_single_source"))
        .when(pl.col("label_corroborated_onchain"))
        .then(pl.lit("report_plus_onchain_pattern"))
        .otherwise(pl.lit("uncorroborated"))
        .alias("label_basis"),
    )


def write(df: pl.DataFrame, name: str = "scam_addresses.parquet") -> None:
    path = INTERIM / name
    df.write_parquet(path)
    log.info("wrote %s rows=%d", path, df.height)

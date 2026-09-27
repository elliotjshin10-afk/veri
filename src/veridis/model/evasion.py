"""Adversarial stress test: what happens when the scammer adapts?

Every number elsewhere assumes the attacker keeps behaving the way the training
data says they behave. That assumption expires the moment the product works.
A pig-butchering operation controls three things - what it tells the victim to
send, when to send it, and which address receives it - so those are the levers
to test.

The test is done at the transfer level, not the feature level. Perturbing
features directly would break the relationships between them (halving an amount
without changing the velocity features produces a wallet that cannot exist), so
instead the underlying transfers are rewritten, the real as-of-time feature
layer recomputes everything from scratch, and the unchanged model scores the
result.

The decision threshold is held fixed at its pre-attack value throughout. A
scammer cannot move a wallet's threshold, and letting it float would hide the
damage.
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

log = logging.getLogger(__name__)

HOUR_MS = 3_600_000
DAY_MS = 86_400_000


VICTIM_FLAG = "_is_victim_tx"


def tag_victim_transfers(tx: pl.DataFrame, pairs: pl.DataFrame) -> pl.DataFrame:
    """Mark the transfers under attack before any of them are rewritten.

    Identifying them afterwards by (sender, destination) does not work: the
    fresh-address attack renames the destination, so the original pair list
    matches nothing and the positive set comes back empty - which reads as
    "detection fell to zero" when in fact nothing was scored at all.
    """
    return tx.with_columns(
        pl.struct("from_address", "to_address")
        .is_in(pairs.select(pl.struct("from_address", "to_address"))
               .to_series().implode())
        .alias(VICTIM_FLAG))


def _targets(tx: pl.DataFrame):
    return (tx.filter(pl.col(VICTIM_FLAG)), tx.filter(~pl.col(VICTIM_FLAG)))


def split_transfers(tx: pl.DataFrame, pairs: pl.DataFrame, n_parts: int = 4,
                    gap_hours: float = 6.0) -> pl.DataFrame:
    """Coach the victim to send in instalments instead of one transfer.

    Attacks amount-based signals (deviation from the sender's own baseline,
    share of balance) at the cost of raising velocity ones.
    """
    target, rest = _targets(tx)
    if target.height == 0:
        return tx
    parts = []
    for i in range(n_parts):
        parts.append(target.with_columns(
            (pl.col("amount_usd") / n_parts).alias("amount_usd"),
            (pl.col("block_time") + int(i * gap_hours * HOUR_MS)).alias("block_time"),
            (pl.col("tx_hash") + f"_s{i}").alias("tx_hash"),
        ))
    return pl.concat([rest, *parts], how="vertical_relaxed")


def slow_cadence(tx: pl.DataFrame, pairs: pl.DataFrame, factor: float = 6.0) -> pl.DataFrame:
    """Stretch the sequence out so bursts stop looking like bursts."""
    target, rest = _targets(tx)
    if target.height == 0:
        return tx
    target = target.sort(["from_address", "to_address", "block_time"])
    first = target.group_by(["from_address", "to_address"]).agg(
        pl.col("block_time").min().alias("_t0"))
    target = (target.join(first, on=["from_address", "to_address"])
              .with_columns((pl.col("_t0")
                             + ((pl.col("block_time") - pl.col("_t0")) * factor)
                             .cast(pl.Int64)).alias("block_time"))
              .drop("_t0"))
    return pl.concat([rest, target], how="vertical_relaxed")


def fresh_destination(tx: pl.DataFrame, pairs: pl.DataFrame) -> pl.DataFrame:
    """Give every victim their own collection address.

    This is the strongest attack available to the operator, and it is cheap:
    addresses are free. It erases every destination-side signal that depends on
    unrelated payers converging - which is most of them.
    """
    target, rest = _targets(tx)
    if target.height == 0:
        return tx
    target = target.with_columns(
        (pl.col("to_address") + "__" + pl.col("from_address").str.slice(0, 10))
        .alias("to_address"))
    return pl.concat([rest, target], how="vertical_relaxed")


def deround(tx: pl.DataFrame, pairs: pl.DataFrame, seed: int = 17) -> pl.DataFrame:
    """Tell the victim to send 4,983 instead of 5,000."""
    target, rest = _targets(tx)
    if target.height == 0:
        return tx
    rng = np.random.default_rng(seed)
    jitter = 1.0 + rng.uniform(-0.04, 0.04, target.height)
    target = target.with_columns(
        (pl.col("amount_usd") * pl.Series(jitter)).alias("amount_usd"))
    return pl.concat([rest, target], how="vertical_relaxed")


ATTACKS = {
    "none": lambda tx, pairs: tx,
    "split_4x": lambda tx, pairs: split_transfers(tx, pairs, n_parts=4),
    "slow_6x": lambda tx, pairs: slow_cadence(tx, pairs, factor=6.0),
    "avoid_round_numbers": lambda tx, pairs: deround(tx, pairs),
    "fresh_address_per_victim": lambda tx, pairs: fresh_destination(tx, pairs),
    "split_and_slow": lambda tx, pairs: slow_cadence(
        split_transfers(tx, pairs, n_parts=4), pairs, factor=6.0),
    "all_combined": lambda tx, pairs: fresh_destination(
        deround(slow_cadence(split_transfers(tx, pairs, n_parts=4), pairs, 6.0), pairs),
        pairs),
}


def evaluate_attack(
    warehouse: pl.DataFrame, victim_pairs: pl.DataFrame, controls: pl.DataFrame,
    booster, threshold: float, attack: str, feature_columns: list[str],
) -> dict:
    """Rebuild the world under an attack and rescore it with the same model."""
    from veridis.features.asof import FeatureEngine

    tagged = tag_victim_transfers(warehouse, victim_pairs)
    tx = ATTACKS[attack](tagged, victim_pairs)

    # Identify positives by the tag, which survives a destination rename.
    pos = (tx.filter(pl.col(VICTIM_FLAG))
           .rename({"from_address": "sender", "to_address": "destination"})
           .with_columns(pl.col("block_time").alias("event_time"))
           .sort(["sender", "destination", "event_time"])
           .with_columns(
               (pl.int_range(pl.len()).over(["sender", "destination"]) + 1)
               .alias("transfer_k"),
               pl.lit(1).cast(pl.Int8).alias("label")))
    cols = ["chain", "sender", "destination", "amount_usd", "asset", "event_time",
            "tx_hash", "transfer_k", "label"]
    events = pl.concat([pos.select(cols), controls.select(cols)],
                       how="vertical_relaxed").with_row_index("event_id")

    engine = FeatureEngine(tx.drop(VICTIM_FLAG))
    feats = engine.compute(events)
    engine.close()

    raw = booster.predict(feats.select(feature_columns).to_numpy())
    feats = feats.with_columns(pl.Series("raw_score", raw))
    y = feats["label"].to_numpy()

    p = feats.filter(pl.col("label") == 1).with_columns(
        (pl.col("raw_score") >= threshold).alias("hit"))
    rel = (p.group_by(["sender", "destination"])
           .agg(pl.col("hit").any().alias("ever"),
                (pl.col("hit") & (pl.col("transfer_k") <= 3)).any().alias("by_k3")))
    first = (p.filter(pl.col("hit")).group_by(["sender", "destination"])
             .agg(pl.col("transfer_k").min().alias("fk")))
    joined = p.join(first, on=["sender", "destination"], how="left")
    protected = float(joined.filter(
        pl.col("fk").is_not_null() & (pl.col("transfer_k") >= pl.col("fk"))
    )["amount_usd"].sum() or 0.0)
    total = float(p["amount_usd"].sum())

    return {
        "attack": attack,
        "n_events": int(feats.height),
        "n_victim_transfers": int(p.height),
        "relationships": int(rel.height),
        "detected_ever": float(rel["ever"].mean()) if rel.height else 0.0,
        "detected_by_k3": float(rel["by_k3"].mean()) if rel.height else 0.0,
        "dollars_protected": protected,
        "dollars_total": total,
        "dollars_share": protected / max(total, 1e-9),
        "control_fpr": float((feats.filter(pl.col("label") == 0)["raw_score"]
                              .to_numpy() >= threshold).mean()),
    }

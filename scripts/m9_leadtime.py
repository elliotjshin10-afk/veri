"""M9b - how early did we already look wrong, before anyone froze the address?

The product claim is that blocklists are late. This measures the gap.

For every Tron address Tether froze in the last 180 days, roll the clock back
to N days before the freeze, compute the destination features using only
transfers that existed at that moment, and score with the destination-only
model. The output is a lead-time distribution: how far ahead of the official
list our score already said "collection address".

Why the setup is clean:

  * The model's training destinations and this population do not intersect at
    all (0 of 2,231 overlap), and training events end 2026-03-13 while the
    freeze window opens 2026-03-31. Out of sample by address and by time.
  * It is the WHOLE recent-freeze population, not a sample. The earlier M1
    ingest had history for 161 of them, but that 161 was selected for having
    traceable victims - which is correlated with being detectable, so a rate
    measured on it would flatter the model.
  * Point-in-time features come from the same FeatureEngine the API serves,
    under the same `block_time < event_time` guard the leakage test covers.

Two honest limits, both reported rather than hidden:

  * Coverage. A truncated history is complete from genesis up to the last
    transfer fetched, so a horizon is only scored when it falls inside that
    window. Horizons we cannot see are excluded, never scored as quiet.
  * dest_funder_fanout needs the funder's own history, which we hold for only
    part of this population. Missing funder history pushes that feature down,
    so it can only cost detections, not invent them. The bias is conservative.
"""
from __future__ import annotations

import datetime as dt
import json
import sys

sys.path.insert(0, "src")

import lightgbm as lgb
import numpy as np
import polars as pl

from veridis.config import INTERIM, PROCESSED
from veridis.features.asof import FeatureEngine
from veridis.model.address_risk import DEST_FEATURES

DAY_MS = 86_400_000
# Geometric-ish grid: dense where the action is (a collector's whole life can be
# under a fortnight), sparse out at the tail where a day of resolution is noise.
# It runs to a year because the first pass stopped at 90 days and the detection
# rate was still flat there - 90d was the grid's ceiling, not the model's, and a
# lead time quoted off a truncated grid understates itself.
HORIZONS = [int(x) for x in sys.argv[1].split(",")] if len(sys.argv) > 1 else [
    1, 3, 7, 14, 30, 60, 90, 120, 180, 270, 365]


def load_warehouse() -> pl.DataFrame:
    """Recent-freeze histories plus everything already ingested.

    The union matters for the funder features: a funder we happen to hold from
    an earlier milestone makes dest_funder_fanout real rather than zero.
    """
    parts = [pl.read_parquet(INTERIM / "leadtime_transfers.parquet")]
    wh = PROCESSED / "warehouse_transfers.parquet"
    if wh.exists():
        parts.append(pl.read_parquet(wh).select(parts[0].columns))
    out = pl.concat(parts, how="vertical_relaxed").unique(subset=["tx_hash", "to_address"])
    print(f"warehouse: {out.height:,} transfers "
          f"({parts[0].height:,} from the recent-freeze fetch)")
    return out


def main() -> None:
    labels = pl.read_parquet(INTERIM / "leadtime_labels.parquet")
    trunc = pl.read_parquet(INTERIM / "leadtime_truncated.parquet")
    transfers = load_warehouse()

    # Per address: when we first and last have visibility.
    span = (pl.concat([
        transfers.select(pl.col("to_address").alias("address"), "block_time"),
        transfers.select(pl.col("from_address").alias("address"), "block_time"),
    ], how="vertical_relaxed")
        .group_by("address")
        .agg(pl.col("block_time").min().alias("first_seen"),
             pl.col("block_time").max().alias("last_seen")))

    pop = (labels.select("address", pl.col("first_reported_at").alias("frozen_at"))
           .join(trunc, on="address", how="left")
           .join(span, on="address", how="left")
           .with_columns(pl.col("truncated").fill_null(False)))

    no_history = pop.filter(pl.col("first_seen").is_null()).height
    pop = pop.filter(pl.col("first_seen").is_not_null())
    print(f"population: {labels.height:,} frozen addresses; "
          f"{no_history:,} with no transfers fetched, {pop.height:,} scoreable")

    # One probe event per (address, horizon): "a transfer arriving here now,
    # judged only on what the destination looks like".
    rows = []
    for r in pop.iter_rows(named=True):
        for h in HORIZONS:
            t = r["frozen_at"] - h * DAY_MS
            if t <= r["first_seen"]:
                continue                      # address did not exist yet
            if r["truncated"] and t > r["last_seen"]:
                continue                      # beyond what we fetched: unseen, not quiet
            rows.append({"address": r["address"], "horizon": h, "event_time": t,
                         "frozen_at": r["frozen_at"]})
    probes = pl.DataFrame(rows)
    print(f"probes: {probes.height:,} (address, horizon) pairs inside coverage")

    ev = (probes.select(pl.col("address").alias("destination"), "event_time")
          .with_columns(pl.lit("tron").alias("chain"),
                        pl.lit("__probe__").alias("sender"),
                        pl.lit(1000.0).alias("amount_usd"))
          .with_row_index("event_id"))

    engine = FeatureEngine(transfers)
    feats = engine.compute(ev)
    engine.close()

    booster = lgb.Booster(model_file=str(PROCESSED / "model_dest_latest.txt"))
    scores = booster.predict(feats.select(DEST_FEATURES).to_numpy())

    thresholds = json.loads((PROCESSED / "address_model_report.json").read_text())["thresholds"]
    hi, el = thresholds["high"], thresholds["elevated"]

    scored = probes.with_columns(pl.Series("score", scores)).with_columns(
        (pl.col("score") >= hi).alias("is_high"),
        (pl.col("score") >= el).alias("is_elevated"))
    scored.write_parquet(PROCESSED / "leadtime_scores.parquet")

    # ---- per-horizon detection ----
    print(f"\nthresholds: high >= {hi:.4f}, elevated >= {el:.4f}\n")
    print("days before      addresses     already      already")
    print("the freeze         scored        HIGH     ELEVATED")
    per_h = {}
    for h in HORIZONS:
        g = scored.filter(pl.col("horizon") == h)
        if not g.height:
            continue
        rh, re_ = g["is_high"].mean(), g["is_elevated"].mean()
        per_h[h] = {"n": g.height, "high": float(rh), "elevated": float(re_)}
        print(f"  {h:3d} days        {g.height:7,}     {rh:6.1%}       {re_:6.1%}")

    # ---- lead time: furthest horizon at which we already said HIGH ----
    lead = (scored.filter(pl.col("is_high"))
            .group_by("address").agg(pl.col("horizon").max().alias("lead_days")))
    detected_any = (scored.group_by("address")
                    .agg(pl.col("is_high").any().alias("ever_high")))
    n_addr = detected_any.height
    n_hit = int(detected_any["ever_high"].sum())
    print(f"\naddresses with at least one scoreable horizon: {n_addr:,}")
    print(f"  already HIGH at some point before the freeze: {n_hit:,} ({n_hit/n_addr:.1%})")
    if n_hit:
        ld = lead["lead_days"]
        print(f"  lead time among those - median {ld.median():.0f}d, "
              f"p25 {ld.quantile(.25):.0f}d, p75 {ld.quantile(.75):.0f}d, max {ld.max():.0f}d")

    out = {"population": labels.height, "no_history": no_history,
           "scoreable_addresses": n_addr, "ever_high": n_hit,
           "ever_high_rate": n_hit / n_addr if n_addr else None,
           "thresholds": thresholds, "per_horizon": per_h,
           "lead_days_median": float(lead["lead_days"].median()) if n_hit else None,
           "horizons": HORIZONS,
           "generated_at": dt.datetime.utcnow().isoformat() + "Z"}
    (PROCESSED / "leadtime_report.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {PROCESSED / 'leadtime_report.json'}")


if __name__ == "__main__":
    main()

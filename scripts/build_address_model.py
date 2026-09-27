"""Train the destination-only address model and measure it at address level."""
import json, sys
sys.path.insert(0, "src")
import numpy as np
import polars as pl
from veridis.features.asof import FeatureEngine
from veridis.model.address_risk import (
    DEST_FEATURES, address_thresholds, evaluate_at_address_level,
    train_destination_model,
)
from veridis.model.train import temporal_split
from veridis.dataset.holdings import all_indexed, all_transfers
from veridis.config import INTERIM, PROCESSED

events = pl.read_parquet(PROCESSED / "events_features.parquet")
split = temporal_split(events)
booster = train_destination_model(split.train)
booster.save_model(str(PROCESSED / "model_dest_latest.txt"))
print(f"trained destination-only model on {split.train.height:,} events, "
      f"{len(DEST_FEATURES)} features")

# --- score every indexed address, as of its own last activity ---
# Training is unchanged - it reads events_features.parquet above. This is the
# scoring pass, and it should cover every address whose history we hold.
warehouse = all_transfers()
indexed = all_indexed()

last = (pl.concat([
    warehouse.select(pl.col("to_address").alias("address"), "block_time"),
    warehouse.select(pl.col("from_address").alias("address"), "block_time"),
], how="vertical_relaxed")
    .filter(pl.col("address").is_in(list(indexed)))
    .group_by("address").agg(pl.col("block_time").max().alias("last_seen")))

# One synthetic scoring event per address, at its own last activity plus a
# second: "what would a transfer into this address have looked like, judged
# only on the destination".
ev = last.select(
    pl.col("address").alias("destination"),
    (pl.col("last_seen") + 1000).alias("event_time"),
).with_columns(
    pl.lit("tron").alias("chain"),
    pl.lit("__probe__").alias("sender"),
    pl.lit(1000.0).alias("amount_usd"),
).with_row_index("event_id")

engine = FeatureEngine(warehouse)
feats = engine.compute(ev)
engine.close()
feats = feats.with_columns(pl.col("destination").alias("address"))
print(f"scored {feats.height:,} indexed addresses")

scam = set(pl.read_parquet(INTERIM / "scam_addresses.parquet")
           .filter(pl.col("accepted"))["address"].to_list())
seeds = set(pl.read_parquet(INTERIM / "control_seeds.parquet")["address"].to_list())
peers = set(pl.read_parquet(INTERIM / "control_peers.parquet")["address"].to_list())
victims = set(pl.read_parquet(INTERIM / "victims.parquet")["victim_address"].to_list())
control = (seeds | peers) - scam - victims

report = evaluate_at_address_level(booster, feats, scam, control)
print("\naddress-level discrimination (scam-listed vs control addresses):")
for k, v in report.items():
    print(f"  {k:<22} {v:.4f}" if isinstance(v, float) else f"  {k:<22} {v}")

labelled = feats.filter(pl.col("address").is_in(list(scam | control)))
y = np.array([1 if a in scam else 0 for a in labelled["address"].to_list()])
s = booster.predict(labelled.select(DEST_FEATURES).to_numpy())
thr = address_thresholds(s, y)
print(f"\nthresholds (on control distribution): {thr}")

all_scores = booster.predict(feats.select(DEST_FEATURES).to_numpy())
out = feats.select("address").with_columns(pl.Series("dest_score", all_scores))
out.write_parquet(PROCESSED / "address_scores.parquet")
(PROCESSED / "address_model_report.json").write_text(json.dumps(
    {"evaluation": report, "thresholds": thr, "features": DEST_FEATURES}, indent=2))
print(f"\nwrote address_scores.parquet ({out.height:,} rows)")

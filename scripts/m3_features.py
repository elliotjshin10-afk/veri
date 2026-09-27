"""M3 - assemble scoring events and compute the as-of-time feature matrix."""
import logging, sys
sys.path.insert(0, "src")
import polars as pl
from veridis.dataset.ingest import dedupe_transfers
from veridis.dataset.events import build_positive_events, build_control_events
from veridis.dataset.matching import (
    match_controls, match_controls_nn, balance_report, standardised_mean_difference,
)
from veridis.dataset.coverage import coverage_table, filter_covered_events
from veridis.features.asof import FeatureEngine, FEATURE_COLUMNS
from veridis.config import INTERIM, PROCESSED, CONTROL_RATIO

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

def load(name):
    p = INTERIM / name
    return pl.read_parquet(p) if p.exists() else None

scam_tx = load("scam_transfers.parquet")
vic_tx = load("victim_transfers.parquet")
ctl_tx = load("control_transfers.parquet")
warehouse = dedupe_transfers([x for x in (scam_tx, vic_tx, ctl_tx) if x is not None])
print(f"warehouse transfers: {warehouse.height:,}")
warehouse.write_parquet(PROCESSED / "warehouse_transfers.parquet")

victims = pl.read_parquet(INTERIM / "victims.parquet")
pos = build_positive_events(warehouse, victims)
print(f"positive events: {pos.height:,} over {pos['destination'].n_unique()} scam addresses")

# Control senders: seeds+peers we fetched, minus anything victim/scam linked.
scam_addrs = pl.read_parquet(INTERIM / "scam_addresses.parquet").filter(
    pl.col("chain") == "tron")["address"]
all_victims = pl.read_parquet(INTERIM / "victims_all.parquet")["victim_address"]

if ctl_tx is not None:
    seeds = set(pl.read_parquet(INTERIM / "control_seeds.parquet")["address"])
    peers = set(pl.read_parquet(INTERIM / "control_peers.parquet")["address"])
    fetched = seeds | peers
    # Both ends must have fetched history, matching the positives' completeness.
    cand = warehouse.filter(
        pl.col("from_address").is_in(list(fetched))
        & pl.col("to_address").is_in(list(fetched))
    )
    neg = build_control_events(cand, all_victims, scam_addrs,
                               n_events=pos.height * CONTROL_RATIO * 3)
else:
    neg = pos.head(0)
print(f"candidate control events: {neg.height:,}")

events = pl.concat([pos, neg], how="vertical_relaxed")

# Drop events whose sender or destination history we did not fetch through the
# event time, so neither class benefits from deeper coverage than the other.
trunc = {}
for f in ("scam_truncated.parquet", "victim_truncated.parquet", "control_truncated.parquet"):
    t = load(f)
    if t is not None:
        trunc.update(dict(zip(t["address"].to_list(), t["truncated"].to_list())))
cov = coverage_table(warehouse, trunc)
print(f"truncated addresses: {int(cov['truncated'].sum())}/{cov.height}")
events = filter_covered_events(events, cov)
print(f"events after coverage guard: {events.height:,} "
      f"(pos={int((events['label']==1).sum()):,})")

events = events.with_row_index("event_id")
engine = FeatureEngine(warehouse)
feats = engine.compute(events)
engine.close()
print(f"features computed: {feats.height:,} rows x {len(FEATURE_COLUMNS)} features")

# Coarse-bin matching first, for comparison, then nearest-neighbour matching
# which is what the model actually trains on. The firehose control pool is
# size-biased toward high-volume senders, so bins alone leave the classes
# separable on sender activity rather than behaviour.
binned = match_controls(feats, ratio=CONTROL_RATIO)
print("\nbin-matched balance (SMD, |SMD|<0.1 is balanced):")
print(standardised_mean_difference(binned))

matched = match_controls_nn(feats, ratio=CONTROL_RATIO)
print("\nnearest-neighbour balance:")
print(standardised_mean_difference(matched))
matched.write_parquet(PROCESSED / "events_features.parquet")
print(f"\nmatched dataset: {matched.height:,} events")
print(matched.group_by("label").len().sort("label"))
print("\ncovariate balance (medians):")
print(balance_report(matched))

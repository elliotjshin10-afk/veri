"""Train and export a model the browser can run on any address.

TronGrid allows cross-origin requests, so a static page can fetch any address's
TRC-20 history itself. That makes a serverless checker possible - but only for
features derivable from *that address's own* transfers.

`dest_funder_fanout` needs a second address's history (whoever first funded
it). Computing it in Python and defaulting it in the browser would be train /
serve skew, the exact failure this project has guarded against elsewhere, so
it is dropped and the cost of dropping it is measured rather than assumed.
"""
import json, sys
sys.path.insert(0, "src")
import lightgbm as lgb
import numpy as np
import polars as pl

from veridis.features.asof import FEATURE_COLUMNS, FEATURE_FAMILY
from veridis.model.address_risk import (
    DEST_FEATURES, address_thresholds, evaluate_at_address_level,
    train_destination_model,
)
from veridis.model.train import temporal_split
from veridis.config import INTERIM, PROCESSED

# Everything computable from one address's own transfer history.
NEEDS_OTHER_ADDRESS = {"dest_funder_fanout"}
BROWSER_FEATURES = [f for f in DEST_FEATURES if f not in NEEDS_OTHER_ADDRESS]
print(f"destination features: {len(DEST_FEATURES)} -> browser-computable: "
      f"{len(BROWSER_FEATURES)} (dropped: {sorted(NEEDS_OTHER_ADDRESS)})")

events = pl.read_parquet(PROCESSED / "events_features.parquet")
split = temporal_split(events)

scam = set(pl.read_parquet(INTERIM / "scam_addresses.parquet")
           .filter(pl.col("accepted"))["address"].to_list())
seeds = set(pl.read_parquet(INTERIM / "control_seeds.parquet")["address"].to_list())
peers = set(pl.read_parquet(INTERIM / "control_peers.parquet")["address"].to_list())
victims = set(pl.read_parquet(INTERIM / "victims.parquet")["victim_address"].to_list())
control = (seeds | peers) - scam - victims

addr_feats = pl.read_parquet(PROCESSED / "address_scores.parquet")  # for the address list
# Recompute address-level features exactly as build_address_model does.
from veridis.features.asof import FeatureEngine
warehouse = pl.read_parquet(PROCESSED / "warehouse_transfers.parquet")
indexed: set[str] = set()
for name in ("scam_truncated.parquet", "victim_truncated.parquet",
             "control_truncated.parquet"):
    p = INTERIM / name
    if p.exists():
        indexed |= set(pl.read_parquet(p)["address"].to_list())
last = (pl.concat([
    warehouse.select(pl.col("to_address").alias("address"), "block_time"),
    warehouse.select(pl.col("from_address").alias("address"), "block_time"),
], how="vertical_relaxed")
    .filter(pl.col("address").is_in(list(indexed)))
    .group_by("address").agg(pl.col("block_time").max().alias("last_seen")))
probe = last.select(
    pl.col("address").alias("destination"),
    (pl.col("last_seen") + 1000).alias("event_time"),
).with_columns(pl.lit("tron").alias("chain"), pl.lit("__probe__").alias("sender"),
               pl.lit(1000.0).alias("amount_usd")).with_row_index("event_id")
engine = FeatureEngine(warehouse)
feats = engine.compute(probe).with_columns(pl.col("destination").alias("address"))
engine.close()


def _score_arm(cols):
    y = split.train["label"].to_numpy()
    ds = lgb.Dataset(split.train.select(cols).to_numpy(), label=y, feature_name=cols)
    b = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 31,
                   "min_data_in_leaf": 30, "feature_fraction": 0.8,
                   "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
                   "scale_pos_weight": (y == 0).sum() / max((y == 1).sum(), 1),
                   "verbose": -1, "seed": 17, "num_threads": 4}, ds, num_boost_round=300)
    return b


full = train_destination_model(split.train)
lite = _score_arm(BROWSER_FEATURES)

ev_full = evaluate_at_address_level(full, feats, scam, control)
import veridis.model.address_risk as ar
ar.DEST_FEATURES = BROWSER_FEATURES      # evaluate the lite arm on its own columns
ev_lite = evaluate_at_address_level(lite, feats, scam, control)
ar.DEST_FEATURES = DEST_FEATURES

print(f"\naddress-level ROC-AUC")
print(f"  full destination model ({len(DEST_FEATURES)} features): {ev_full['roc_auc']:.4f}")
print(f"  browser model ({len(BROWSER_FEATURES)} features):        {ev_lite['roc_auc']:.4f}")
print(f"  cost of dropping the funder feature: {ev_full['roc_auc']-ev_lite['roc_auc']:+.4f}")

labelled = feats.filter(pl.col("address").is_in(list(scam | control)))
y = np.array([1 if a in scam else 0 for a in labelled["address"].to_list()])
s = lite.predict(labelled.select(BROWSER_FEATURES).to_numpy())
thr = address_thresholds(s, y)

def slim(t):
    """Export at full precision.

    Rounding thresholds to six decimals is tempting for file size and wrong:
    a feature value sitting just either side of a rounded threshold takes the
    other branch, and across 300 trees that moved scores by up to 0.19 against
    the Python model. Cheap to store, expensive to debug.
    """
    def node(n):
        if "leaf_value" in n:
            return {"v": n["leaf_value"]}
        return {"f": n["split_feature"], "t": n["threshold"],
                "d": 1 if n.get("default_left") else 0,
                "l": node(n["left_child"]), "r": node(n["right_child"])}
    return node(t["tree_structure"])

dump = lite.dump_model()
out = {"features": BROWSER_FEATURES,
       "trees": [slim(t) for t in dump["tree_info"]],
       "thresholds": thr,
       "evaluation": ev_lite}
path = __import__("pathlib").Path("site/model_dest.json")
path.write_text(json.dumps(out, separators=(",", ":")))
lite.save_model(str(PROCESSED / "model_browser.txt"))
print(f"\nwrote {path} ({path.stat().st_size/1024:.0f} KB, "
      f"{len(out['trees'])} trees)")
print(f"thresholds: {thr}")

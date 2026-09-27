"""M9e - recall in the unit the product actually works in: payments warned on.

Address-level recall answers "what share of scam collectors do we catch". That
is not what the product does. It sits in a send flow and warns on individual
payments, so the number that matters is what share of the payments INTO those
addresses would have carried a warning.

The two differ, and in our favour: the addresses we catch are the busy ones, so
payment-weighted recall runs well above address-weighted recall. The collectors
we miss are the quiet ones, which by definition took money from fewer people.

Deliberately counted in payments, not dollars. The holdings contain a single
$212.9bn USDT transfer - larger than Tether's entire supply - and a handful of
other treasury-scale movements, so any dollar total is set by a few outliers
rather than by the victims. Median transfer here is $837.
"""
from __future__ import annotations
import json, sys
sys.path.insert(0, "src")
import lightgbm as lgb
import polars as pl
from veridis.config import INTERIM, PROCESSED
from veridis.features.asof import FeatureEngine
from veridis.model.address_risk import DEST_FEATURES
from veridis.model.train import temporal_split
from veridis.dataset.holdings import all_transfers

DAY = 86_400_000
HORIZON = int(sys.argv[1]) if len(sys.argv) > 1 else 90


def main() -> None:
    tf = all_transfers()
    trained = set(temporal_split(
        pl.read_parquet(PROCESSED / "events_features.parquet")).train["destination"].unique())
    span = (pl.concat([
        tf.select(pl.col("to_address").alias("address"), "block_time"),
        tf.select(pl.col("from_address").alias("address"), "block_time")], how="vertical_relaxed")
        .group_by("address").agg(pl.col("block_time").min().alias("first_seen"),
                                 pl.col("block_time").max().alias("last_seen")))

    pos = (pl.read_parquet(INTERIM / "leadtime_labels.parquet")
           .select("address", pl.col("first_reported_at").alias("frozen_at"))
           .join(pl.read_parquet(INTERIM / "leadtime_truncated.parquet"), on="address", how="left")
           .join(span, on="address", how="left")
           .with_columns(pl.col("truncated").fill_null(False))
           .filter(pl.col("first_seen").is_not_null() & ~pl.col("address").is_in(list(trained)))
           .with_columns((pl.col("frozen_at") - HORIZON * DAY).alias("t")))
    pos = pos.filter((pl.col("t") > pl.col("first_seen"))
                     & (~pl.col("truncated") | (pl.col("t") <= pl.col("last_seen"))))

    ev = (pos.select(pl.col("address").alias("destination"), pl.col("t").alias("event_time"))
          .with_columns(pl.lit("tron").alias("chain"), pl.lit("__probe__").alias("sender"),
                        pl.lit(1000.0).alias("amount_usd")).with_row_index("event_id"))
    engine = FeatureEngine(tf)
    feats = engine.compute(ev)
    engine.close()
    booster = lgb.Booster(model_file=str(PROCESSED / "model_dest_latest.txt"))
    pos = pos.with_columns(pl.Series("score", booster.predict(
        feats.select(DEST_FEATURES).to_numpy())))

    # Payments arriving between the probe and the freeze - the window in which
    # the address was already flaggable and still entirely unlisted.
    pay = (tf.select(pl.col("to_address").alias("address"), "block_time", "amount_usd")
           .join(pos.select("address", "t", "frozen_at", "score"), on="address", how="inner")
           .filter((pl.col("block_time") > pl.col("t"))
                   & (pl.col("block_time") <= pl.col("frozen_at"))))

    roc = json.loads((PROCESSED / "leadtime_roc.json").read_text())
    total = pay.height
    print(f"horizon {HORIZON}d - {pos.height:,} addresses probed, "
          f"{pay['address'].n_unique():,} of them took payments in the window")
    print(f"payments into them while still unlisted: {total:,}")
    print(f"  median payment ${pay['amount_usd'].median():,.0f}, "
          f"p90 ${pay['amount_usd'].quantile(0.9):,.0f}\n")

    out = {}
    print("false-alarm budget   addresses caught   PAYMENTS warned on")
    for key in ("0.01", "0.02", "0.05", "0.10"):
        thr = roc["operating_points"][key]["threshold"]
        addr = pos.filter(pl.col("score") >= thr).height / max(pos.height, 1)
        pays = pay.filter(pl.col("score") >= thr).height / max(total, 1)
        out[key] = {"threshold": thr, "address_recall": addr, "payment_recall": pays}
        print(f"       {float(key):4.0%}              {addr:5.1%}              {pays:5.1%}")

    (PROCESSED / "leadtime_payments.json").write_text(json.dumps(
        {"horizon_days": HORIZON, "payments_in_window": total,
         "median_payment_usd": float(pay["amount_usd"].median()),
         "operating_points": out}, indent=2))
    print(f"\nwrote {PROCESSED / 'leadtime_payments.json'}")


if __name__ == "__main__":
    main()

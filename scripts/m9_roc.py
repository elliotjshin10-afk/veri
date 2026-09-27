"""M9d - one point-in-time experiment with both arms, so recall and FPR come
from the same protocol.

The lead-time backtest scored only addresses that were later frozen. That makes
it a recall figure and nothing else: every address in it was a positive, so it
cannot say how often the model cries wolf. Pairing it with the 5% FPR from the
address-level evaluation was an approximation across two different populations
scored under two different rules - the positives at a horizon before a freeze,
the negatives at their own last activity.

This runs both arms through the identical protocol:

  * Positives: every Tron address Tether froze in the window, probed N days
    before its freeze.
  * Negatives: ordinary wallets that were never frozen, probed N days before a
    PSEUDO-freeze date drawn from the positives' own freeze-date distribution,
    so both arms face the same calendar and the same market conditions. A
    control scored at today's date and a positive scored a year ago would differ
    by epoch as much as by behaviour.

Both arms honour the same coverage rule and the same as-of guard, and every
address in both arms is excluded from the model's training data.

Output is a real ROC point per horizon: TPR and FPR measured together.
"""
from __future__ import annotations

import datetime as dt
import json
import sys

sys.path.insert(0, "src")

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score, average_precision_score

from veridis.config import INTERIM, PROCESSED
from veridis.features.asof import FeatureEngine
from veridis.model.address_risk import DEST_FEATURES
from veridis.model.train import temporal_split

DAY_MS = 86_400_000
HORIZONS = [1, 7, 14, 30, 60, 90, 180]
SEED = 17


def spans(transfers: pl.DataFrame) -> pl.DataFrame:
    return (pl.concat([
        transfers.select(pl.col("to_address").alias("address"), "block_time"),
        transfers.select(pl.col("from_address").alias("address"), "block_time"),
    ], how="vertical_relaxed")
        .group_by("address")
        .agg(pl.col("block_time").min().alias("first_seen"),
             pl.col("block_time").max().alias("last_seen")))


def probes(pop: pl.DataFrame, label: int) -> list[dict]:
    """(address, horizon) pairs that fall inside what we can actually see."""
    rows = []
    for r in pop.iter_rows(named=True):
        for h in HORIZONS:
            t = r["mark_at"] - h * DAY_MS
            if t <= r["first_seen"]:
                continue                      # address did not exist yet
            if r["truncated"] and t > r["last_seen"]:
                continue                      # past our coverage: unseen, not quiet
            rows.append({"address": r["address"], "horizon": h,
                         "event_time": t, "label": label})
    return rows


def main() -> None:
    rng = np.random.default_rng(SEED)

    # ---- everything we hold, for features ----
    parts = [pl.read_parquet(INTERIM / "leadtime_transfers.parquet")]
    cols = parts[0].columns
    for extra in (PROCESSED / "warehouse_transfers.parquet",
                  INTERIM / "control_transfers.parquet"):
        if extra.exists():
            parts.append(pl.read_parquet(extra).select(cols))
    transfers = pl.concat(parts, how="vertical_relaxed").unique(subset=["tx_hash", "to_address"])
    span = spans(transfers)
    print(f"warehouse: {transfers.height:,} transfers")

    train_dest = set(temporal_split(
        pl.read_parquet(PROCESSED / "events_features.parquet")).train["destination"].unique())

    # ---- positives ----
    pos = (pl.read_parquet(INTERIM / "leadtime_labels.parquet")
           .select("address", pl.col("first_reported_at").alias("mark_at"))
           .join(pl.read_parquet(INTERIM / "leadtime_truncated.parquet"), on="address", how="left")
           .join(span, on="address", how="left")
           .with_columns(pl.col("truncated").fill_null(False))
           .filter(pl.col("first_seen").is_not_null()
                   & ~pl.col("address").is_in(list(train_dest))))

    # ---- negatives, with pseudo-freeze dates drawn from the positives ----
    scam = set(pl.read_parquet(INTERIM / "scam_addresses.parquet")
               .filter(pl.col("accepted"))["address"].to_list())
    victims = set(pl.read_parquet(INTERIM / "victims.parquet")["victim_address"].to_list())
    ctrl = ((set(pl.read_parquet(INTERIM / "control_seeds.parquet")["address"].to_list())
             | set(pl.read_parquet(INTERIM / "control_peers.parquet")["address"].to_list()))
            - scam - victims - train_dest)

    marks = pos["mark_at"].to_numpy()
    neg = (pl.DataFrame({"address": sorted(ctrl)})
           .join(pl.read_parquet(INTERIM / "control_truncated.parquet"), on="address", how="left")
           .join(span, on="address", how="left")
           .with_columns(pl.col("truncated").fill_null(False))
           .filter(pl.col("first_seen").is_not_null()))
    neg = neg.with_columns(
        pl.Series("mark_at", rng.choice(marks, size=neg.height, replace=True).astype("int64")))

    print(f"positives {pos.height:,} (never trained on)   negatives {neg.height:,}")

    rows = probes(pos, 1) + probes(neg, 0)
    pr = pl.DataFrame(rows)
    print(f"probes: {pr.height:,} "
          f"({int((pr['label'] == 1).sum()):,} pos / {int((pr['label'] == 0).sum()):,} neg)")

    ev = (pr.select(pl.col("address").alias("destination"), "event_time")
          .with_columns(pl.lit("tron").alias("chain"), pl.lit("__probe__").alias("sender"),
                        pl.lit(1000.0).alias("amount_usd"))
          .with_row_index("event_id"))
    engine = FeatureEngine(transfers)
    feats = engine.compute(ev)
    engine.close()

    booster = lgb.Booster(model_file=str(PROCESSED / "model_dest_latest.txt"))
    pr = pr.with_columns(pl.Series("score", booster.predict(
        feats.select(DEST_FEATURES).to_numpy())))

    thr = json.loads((PROCESSED / "address_model_report.json").read_text())["thresholds"]
    hi = thr["high"]

    print(f"\nboth arms scored at the same horizon, threshold high >= {hi:.4f}\n")
    print("days before   positives  negatives      TPR      FPR   ROC-AUC")
    out = {}
    for h in HORIZONS:
        g = pr.filter(pl.col("horizon") == h)
        p, n = g.filter(pl.col("label") == 1), g.filter(pl.col("label") == 0)
        if not p.height or not n.height:
            continue
        tpr = float((p["score"] >= hi).mean())
        fpr = float((n["score"] >= hi).mean())
        auc = roc_auc_score(g["label"].to_numpy(), g["score"].to_numpy())
        ap = average_precision_score(g["label"].to_numpy(), g["score"].to_numpy())
        out[h] = {"n_pos": p.height, "n_neg": n.height, "tpr": tpr, "fpr": fpr,
                  "roc_auc": float(auc), "pr_auc": float(ap)}
        print(f"  {h:3d} days      {p.height:6,}     {n.height:6,}   {tpr:6.1%}   {fpr:6.1%}     {auc:.3f}")

    # ---- what a chosen false-alarm budget actually buys, at the headline
    # horizon. One threshold is a point; a product decision needs the curve.
    head = 90
    g = pr.filter(pl.col("horizon") == head)
    ps = np.sort(g.filter(pl.col("label") == 1)["score"].to_numpy())
    ns = np.sort(g.filter(pl.col("label") == 0)["score"].to_numpy())
    ops = {}
    print(f"\noperating points at {head} days, both arms point-in-time")
    print("  FPR budget   threshold      TPR")
    for target in (0.01, 0.02, 0.05, 0.10):
        t = float(np.quantile(ns, 1 - target))
        ops[f"{target:.2f}"] = {"threshold": t,
                                "tpr": float((ps >= t).mean()),
                                "fpr": float((ns >= t).mean())}
        print(f"    {target:6.0%}     {t:9.4f}   {float((ps >= t).mean()):6.1%}")

    (PROCESSED / "leadtime_roc.json").write_text(json.dumps(
        {"horizons": out, "operating_points": ops, "headline_horizon": head,
         "threshold_high": hi, "seed": SEED,
         "generated_at": dt.datetime.utcnow().isoformat() + "Z"}, indent=2))
    print(f"\nwrote {PROCESSED / 'leadtime_roc.json'}")


if __name__ == "__main__":
    main()

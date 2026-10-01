"""Does the method work on Ethereum, or only on Tron?

The earlier cross-chain test took the Tron model, changed nothing, and scored
Ethereum: ROC-AUC 0.682 against 0.947. That measured TRANSFER, not the method -
and the same experiment on Tron would have looked just as bad, because the
model was trained on the wrong task there too.

This retrains instead. Same point-in-time protocol as Tron: frozen addresses
probed at several horizons before the freeze that listed them, ordinary wallets
probed before a pseudo-freeze drawn from the positives' own dates, split
temporally on the mark date so an address sits on one side only.

ONE DIFFERENCE, FORCED BY THE DATA SOURCE. Blockscout pages newest-first, so a
truncated history is missing an address's EARLIEST activity - age, lifetime
counts and every "since first seen" feature are wrong at every point in time,
not merely late. There is no as-of cutoff that repairs that, so only complete
histories are used. It costs more than a third of the addresses and it is not
optional.
"""
from __future__ import annotations

import json, sys
sys.path.insert(0, "src")
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from veridis.config import INTERIM, PROCESSED
from veridis.features.asof import FeatureEngine
from veridis.model.address_risk import DEST_FEATURES

DAY = 86_400_000
HORIZONS = [7, 30, 90, 180]
TEST_FRAC = 0.30
SEED = 17
BROWSER = [f for f in DEST_FEATURES if f != "dest_funder_fanout"]

TX = ("eth_scam_transfers", "eth_victim_transfers", "eth_control_transfers",
      "eth_coverage_transfers", "eth_control2_transfers")
TR = ("eth_truncated", "eth_coverage_truncated", "eth_control2_truncated")


def main() -> None:
    rng = np.random.default_rng(SEED)
    tf = pl.concat([pl.read_parquet(INTERIM / f"{n}.parquet") for n in TX
                    if (INTERIM / f"{n}.parquet").exists()],
                   how="vertical_relaxed").unique(subset=["tx_hash", "to_address"])
    trunc = pl.concat([pl.read_parquet(INTERIM / f"{n}.parquet") for n in TR
                       if (INTERIM / f"{n}.parquet").exists()],
                      how="vertical_relaxed").unique("address")
    complete = set(trunc.filter(~pl.col("truncated"))["address"].to_list())

    span = (pl.concat([
        tf.select(pl.col("to_address").alias("address"), "block_time"),
        tf.select(pl.col("from_address").alias("address"), "block_time")],
        how="vertical_relaxed").group_by("address")
        .agg(pl.col("block_time").min().alias("first_seen"),
             pl.col("block_time").max().alias("last_seen")))

    sa = pl.read_parquet(INTERIM / "scam_addresses.parquet")
    scam = set(sa.filter(pl.col("accepted"))["address"].to_list())
    pos = (sa.filter((pl.col("chain") == "ethereum") & pl.col("accepted")
                     & pl.col("first_reported_at").is_not_null())
           .select("address", pl.col("first_reported_at").alias("mark"))
           .filter(pl.col("address").is_in(list(complete))))
    ctrl: set[str] = set()
    for n in ("eth_control_seeds", "eth_control_peers", "eth_control2_truncated"):
        p = INTERIM / f"{n}.parquet"
        if p.exists():
            ctrl |= set(pl.read_parquet(p)["address"].to_list())
    ctrl = (ctrl & complete) - scam
    print(f"complete histories: {len(pos):,} frozen, {len(ctrl):,} ordinary "
          f"({tf.height:,} transfers)")
    # A pseudo-freeze must sit inside the control's own life. Drawing one blind
    # from the positives put it before the address existed, every probe was
    # dropped by the coverage guard, and the negative arm came back empty - a
    # silent failure that reads as a LightGBM error about scale_pos_weight.
    marks = np.sort(pos["mark"].to_numpy())
    first = dict(zip(span["address"].to_list(), span["first_seen"].to_list()))
    rowsn, skipped = [], 0
    for a in sorted(ctrl):
        f = first.get(a)
        if f is None:
            skipped += 1
            continue
        usable = marks[marks - max(HORIZONS) * DAY > f]
        if not len(usable):
            skipped += 1
            continue
        rowsn.append({"address": a, "mark": int(rng.choice(usable))})
    neg = pl.DataFrame(rowsn)
    print(f"ordinary wallets usable: {neg.height:,} "
          f"({skipped:,} too young for any pseudo-freeze date)")
    if neg.height < 150:
        sys.exit("too few usable ordinary wallets - run scripts/m7_controls_more.py")

    rows = []
    for df, label in ((pos, 1), (neg, 0)):
        d = df.join(span, on="address", how="left").filter(pl.col("first_seen").is_not_null())
        for r in d.iter_rows(named=True):
            for h in HORIZONS:
                t = r["mark"] - h * DAY
                if t <= r["first_seen"]:
                    continue
                rows.append({"address": r["address"], "label": label, "event_time": t,
                             "mark": r["mark"]})
    pr = pl.DataFrame(rows)
    cut = float(np.quantile(pr["mark"].to_numpy(), 1 - TEST_FRAC))
    is_test = pr["mark"].to_numpy() >= cut
    y = pr["label"].to_numpy()
    print(f"probes {pr.height:,} over {pr['address'].n_unique():,} addresses; "
          f"test {int(is_test.sum()):,} "
          f"({int((y[is_test]==1).sum()):,} frozen / {int((y[is_test]==0).sum()):,} ordinary)")

    ev = (pr.select(pl.col("address").alias("destination"), "event_time")
          .with_columns(pl.lit("ethereum").alias("chain"), pl.lit("__probe__").alias("sender"),
                        pl.lit(1000.0).alias("amount_usd")).with_row_index("event_id"))
    engine = FeatureEngine(tf)
    feats = engine.compute(ev)
    engine.close()

    out = {}
    print("\n  arm        ROC-AUC  PR-AUC   TPR@5%FPR")
    for name, cols in (("full   ", DEST_FEATURES), ("browser", BROWSER)):
        X = feats.select(cols).to_numpy()
        ytr = y[~is_test]
        b = lgb.train({
            "objective": "binary", "learning_rate": 0.05, "num_leaves": 31,
            "min_data_in_leaf": 30, "feature_fraction": 0.8, "bagging_fraction": 0.8,
            "bagging_freq": 1, "lambda_l2": 1.0,
            "scale_pos_weight": (ytr == 0).sum() / max((ytr == 1).sum(), 1),
            "verbose": -1, "seed": SEED, "num_threads": 4,
        }, lgb.Dataset(X[~is_test], label=ytr, feature_name=cols), num_boost_round=400)
        s, yt = b.predict(X[is_test]), y[is_test]
        thr = float(np.quantile(np.sort(s[yt == 0]), 0.95))
        res = {"roc_auc": float(roc_auc_score(yt, s)),
               "pr_auc": float(average_precision_score(yt, s)),
               "tpr_at_5pct_fpr": float((s[yt == 1] >= thr).mean()),
               "n_pos": int((yt == 1).sum()), "n_neg": int((yt == 0).sum())}
        out[name.strip()] = res
        print(f"  {name}   {res['roc_auc']:.4f}  {res['pr_auc']:.4f}   {res['tpr_at_5pct_fpr']:.1%}")
        if name.strip() == "full":
            b.save_model(str(PROCESSED / "model_eth_pit.txt"))

    (PROCESSED / "eth_pit_report.json").write_text(json.dumps(
        {"results": out, "horizons": HORIZONS, "complete_only": True,
         "n_pos_addresses": len(pos), "n_ctrl_addresses": len(ctrl)}, indent=2))
    print(f"\nwrote model_eth_pit.txt and eth_pit_report.json")


if __name__ == "__main__":
    main()

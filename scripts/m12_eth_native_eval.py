"""Does reading native ETH make the Ethereum model better?

The shipped model reads stablecoins only. 80% of frozen Ethereum addresses also
receive ether, and we have never looked at it. This answers whether that is a
gap worth closing or a detail.

The comparison is deliberately narrow. One cohort, one set of probes, two
warehouses:

    A  stablecoin transfers only          - what ships today
    B  stablecoin transfers plus ether    - the same addresses, read fully

Everything else is held fixed. The probes are built ONCE, from the stablecoin
warehouse, and both arms score exactly those - so a probe cannot become valid in
one arm and not the other, and the difference cannot be an artifact of which
events each model was asked about. That choice is conservative: ether makes
addresses look older, which would have let arm B probe further back.

Run: python scripts/m12_eth_native_eval.py
"""
from __future__ import annotations

import json, sys
sys.path.insert(0, "src")

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from veridis.config import INTERIM, PROCESSED
from veridis.dataset.holdings import eth_transfers
from veridis.features.asof import FeatureEngine
from veridis.model.address_risk import DEST_FEATURES

DAY = 86_400_000
HORIZONS = [7, 30, 90, 180]
TEST_FRAC = 0.30
SEED = 17
BROWSER = [f for f in DEST_FEATURES if f != "dest_funder_fanout"]


def spans(tf: pl.DataFrame) -> pl.DataFrame:
    return (pl.concat([
        tf.select(pl.col("to_address").alias("address"), "block_time"),
        tf.select(pl.col("from_address").alias("address"), "block_time")],
        how="vertical_relaxed").group_by("address")
        .agg(pl.col("block_time").min().alias("first_seen")))


def main() -> None:
    cohort = json.loads((PROCESSED / "eth_native_cohort.json").read_text())
    pos_addrs, ctrl_addrs = set(cohort["frozen"]), set(cohort["ordinary"])

    stable = eth_transfers()
    nat_p = INTERIM / "eth_native_transfers.parquet"
    if not nat_p.exists():
        sys.exit("no eth_native_transfers.parquet - run scripts/m12_eth_native.py")
    native = pl.read_parquet(nat_p).select(stable.columns)
    both = pl.concat([stable, native], how="vertical_relaxed").unique(
        subset=["tx_hash", "to_address", "asset"])

    # Only the cohort's own rows matter, but the warehouse must keep every
    # counterparty row too - the features count distinct payers, and trimming
    # the warehouse to the cohort would silently shrink them.
    fetched = set(pl.read_parquet(INTERIM / "eth_native_truncated.parquet")
                  ["address"].to_list())
    pos_addrs &= fetched
    ctrl_addrs &= fetched
    nat_in_cohort = native.filter(
        pl.col("to_address").is_in(list(pos_addrs | ctrl_addrs))
        | pl.col("from_address").is_in(list(pos_addrs | ctrl_addrs)))
    print(f"cohort with ether fetched: {len(pos_addrs):,} frozen, "
          f"{len(ctrl_addrs):,} ordinary")
    print(f"stablecoin warehouse {stable.height:,} rows; "
          f"+ {native.height:,} native ETH rows "
          f"({nat_in_cohort.height:,} touching the cohort)")

    sa = pl.read_parquet(INTERIM / "scam_addresses.parquet")
    pos = (sa.filter(pl.col("address").is_in(list(pos_addrs))
                     & pl.col("first_reported_at").is_not_null())
           .select("address", pl.col("first_reported_at").alias("mark"))
           .unique("address"))

    # Probes from the STABLECOIN span, so both arms answer the same questions.
    span = spans(stable)
    rng = np.random.default_rng(SEED)
    marks = np.sort(pos["mark"].to_numpy())
    first = dict(zip(span["address"].to_list(), span["first_seen"].to_list()))
    rowsn = []
    for a in sorted(ctrl_addrs):
        f = first.get(a)
        if f is None:
            continue
        usable = marks[marks - max(HORIZONS) * DAY > f]
        if not len(usable):
            continue
        rowsn.append({"address": a, "mark": int(rng.choice(usable))})
    neg = pl.DataFrame(rowsn)
    if neg.height < 150 or pos.height < 150:
        sys.exit(f"cohort too small: {pos.height} frozen, {neg.height} ordinary")

    rows = []
    for df, label in ((pos, 1), (neg, 0)):
        d = df.join(span, on="address", how="left").filter(
            pl.col("first_seen").is_not_null())
        for r in d.iter_rows(named=True):
            for h in HORIZONS:
                t = r["mark"] - h * DAY
                if t <= r["first_seen"]:
                    continue
                rows.append({"address": r["address"], "label": label,
                             "event_time": t, "mark": r["mark"]})
    pr = pl.DataFrame(rows)
    cut = float(np.quantile(pr["mark"].to_numpy(), 1 - TEST_FRAC))
    is_test = pr["mark"].to_numpy() >= cut
    y = pr["label"].to_numpy()
    print(f"probes {pr.height:,} over {pr['address'].n_unique():,} addresses; "
          f"test {int(is_test.sum()):,} "
          f"({int((y[is_test]==1).sum()):,} frozen / "
          f"{int((y[is_test]==0).sum()):,} ordinary)")

    ev = (pr.select(pl.col("address").alias("destination"), "event_time")
          .with_columns(pl.lit("ethereum").alias("chain"),
                        pl.lit("__probe__").alias("sender"),
                        pl.lit(1000.0).alias("amount_usd")).with_row_index("event_id"))

    # Dust is a real confound, not a tidiness issue. 14.9% of inbound ether to
    # frozen addresses is worth under a dollar, and dusting campaigns target
    # addresses that are already known-bad - so if ether lifted the model purely
    # by inflating `dest_senders_all` with spam, that lift would evaporate the
    # moment a wallet asked us about a dusted ordinary address. The floor is
    # applied to BOTH assets in BOTH arms, because a filter that only touches
    # ether would be comparing two different definitions of a payment.
    MIN_USD = 1.0
    stable_f = stable.filter(pl.col("amount_usd") >= MIN_USD)
    both_f = both.filter(pl.col("amount_usd") >= MIN_USD)

    print("\n  warehouse                      ROC-AUC  PR-AUC  TPR@5%FPR")
    out = {}
    for name, wh in (("stablecoins only (ships)", stable),
                     ("stablecoins + native ETH", both),
                     (f"stablecoins only, >=${MIN_USD:.0f}", stable_f),
                     (f"stablecoins + ETH, >=${MIN_USD:.0f}", both_f)):
        engine = FeatureEngine(wh)
        feats = engine.compute(ev)
        engine.close()
        X = feats.select(BROWSER).to_numpy()
        ytr = y[~is_test]
        b = lgb.train({
            "objective": "binary", "learning_rate": 0.05, "num_leaves": 31,
            "min_data_in_leaf": 30, "feature_fraction": 0.8, "bagging_fraction": 0.8,
            "bagging_freq": 1, "lambda_l2": 1.0,
            "scale_pos_weight": (ytr == 0).sum() / max((ytr == 1).sum(), 1),
            "verbose": -1, "seed": SEED, "num_threads": 4,
        }, lgb.Dataset(X[~is_test], label=ytr, feature_name=BROWSER),
            num_boost_round=400)
        s, yt = b.predict(X[is_test]), y[is_test]
        thr = float(np.quantile(np.sort(s[yt == 0]), 0.95))
        r = {"roc_auc": float(roc_auc_score(yt, s)),
             "pr_auc": float(average_precision_score(yt, s)),
             "tpr_at_5pct_fpr": float((s[yt == 1] >= thr).mean())}
        out[name] = r
        print(f"  {name:<30} {r['roc_auc']:.4f}  {r['pr_auc']:.4f}   "
              f"{r['tpr_at_5pct_fpr']:.1%}")

    a, b_ = out["stablecoins only (ships)"], out["stablecoins + native ETH"]
    d_auc = b_["roc_auc"] - a["roc_auc"]
    d_tpr = b_["tpr_at_5pct_fpr"] - a["tpr_at_5pct_fpr"]
    af, bf = out[f"stablecoins only, >=${MIN_USD:.0f}"], out[f"stablecoins + ETH, >=${MIN_USD:.0f}"]
    d_auc_f = bf["roc_auc"] - af["roc_auc"]
    print(f"\n  reading native ETH moves ROC-AUC {d_auc:+.4f} and recall at a 5% "
          f"false-alarm budget {d_tpr:+.1%}")
    print(f"  with dust under ${MIN_USD:.0f} removed from both arms: ROC-AUC {d_auc_f:+.4f}")
    if d_auc > 0 and d_auc_f <= 0:
        print("  -> the gain was dust. Native ETH does not help once spam is excluded.")
    (PROCESSED / "eth_native_report.json").write_text(json.dumps(
        {"results": out, "delta_roc_auc": d_auc, "delta_tpr_at_5pct_fpr": d_tpr,
         "delta_roc_auc_dust_filtered": d_auc_f, "min_usd": MIN_USD,
         "n_frozen": len(pos_addrs), "n_ordinary": len(ctrl_addrs),
         "n_probes": int(pr.height), "native_rows": int(native.height),
         "probes_from": "stablecoin warehouse, identical for both arms"}, indent=2))
    print("wrote eth_native_report.json")


if __name__ == "__main__":
    main()

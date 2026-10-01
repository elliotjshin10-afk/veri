"""Train the two-sided Ethereum model - the one a connected wallet unlocks.

The shipped Ethereum model scores a destination from its own history, which is
all an address lookup can ask. With the sender known, fifteen relationship
features become available, and they are the strongest signal in the Tron model:
whether this is the first transfer to this address, how it compares to every
earlier one, whether the two addresses share any counterparty at all.

Feature set is EXACTLY the Tron pair model's 37 - destination, context and
relationship - for one reason that is not aesthetic: the browser already
computes those 37 and is parity-locked to them. A different list would mean
porting new JS feature code, and every hand-ported feature is a chance for the
serving path to diverge from the training path, which is the failure this
project guards against hardest. `dest_funder_fanout` stays dropped because it
needs a third address's history.

The 37 SENDER-only features are a separate, measured upgrade (+8.9pp recall on
Tron) and are deliberately not here: they would roughly triple the JS surface
to keep at exact parity. This script reports what they would buy on Ethereum,
so the decision stays a measurement rather than an assumption.

Honesty gates, the same ones the destination model earned the hard way:

  * both sides of every event must have a COMPLETE fetched history - every
    lifetime feature is wrong on a partial one
  * the negative arm's destinations are the independently-sampled ordinary
    wallets, never counterparties of frozen addresses
  * positives are strictly pre-freeze, and the split is forward in time, so a
    test event is never scored with knowledge from a later freeze
  * a sender on the positive arm is never reused as a control sender

Run: python scripts/m10_eth_pair_model.py
"""
from __future__ import annotations

import json, pathlib, sys
sys.path.insert(0, "src")

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from veridis.config import CONTROL_RATIO, INTERIM, PROCESSED, SITE
from veridis.dataset.events import (MAX_EVENTS_PER_PAIR, RETAIL_MAX_SENDERS,
                                    RETAIL_MAX_USD_PER_SENDER,
                                    RETAIL_MIN_INBOUND_USD, RETAIL_MIN_SENDERS)
from veridis.dataset.holdings import eth_complete, eth_transfers
from veridis.dataset.matching import match_controls_nn, standardised_mean_difference
from veridis.features.asof import FEATURE_COLUMNS, FEATURE_FAMILY, FeatureEngine

SEED = 17
TEST_FRAC = 0.30
NEEDS_THIRD_ADDRESS = {"dest_funder_fanout"}

DEST_ONLY = [c for c in FEATURE_COLUMNS
             if FEATURE_FAMILY[c] == "destination" and c not in NEEDS_THIRD_ADDRESS]
PAIR_FEATURES = [c for c in FEATURE_COLUMNS
                 if FEATURE_FAMILY[c] in ("destination", "context", "relationship")
                 and c not in NEEDS_THIRD_ADDRESS]
WITH_SENDER = [c for c in FEATURE_COLUMNS if c not in NEEDS_THIRD_ADDRESS]


def slim(t):
    """Full precision on purpose: rounding thresholds to six decimals moved
    scores by up to 0.19 against Python, because a value either side of a
    rounded threshold takes the other branch across hundreds of trees."""
    def node(n):
        if "leaf_value" in n:
            return {"v": n["leaf_value"]}
        return {"f": n["split_feature"], "t": n["threshold"],
                "d": 1 if n.get("default_left") else 0,
                "l": node(n["left_child"]), "r": node(n["right_child"])}
    return node(t["tree_structure"])


def retail_frozen(tf: pl.DataFrame) -> pl.DataFrame:
    sa = pl.read_parquet(INTERIM / "scam_addresses.parquet")
    frozen = (sa.filter((pl.col("chain") == "ethereum") & pl.col("accepted")
                        & pl.col("first_reported_at").is_not_null())
              .select("address", pl.col("first_reported_at").alias("frozen_at")))
    prof = (tf.join(frozen, left_on="to_address", right_on="address", how="inner")
            .group_by("to_address", "frozen_at")
            .agg(pl.col("from_address").n_unique().alias("senders"),
                 pl.col("amount_usd").sum().alias("usd")))
    return prof.filter(
        (pl.col("senders") >= RETAIL_MIN_SENDERS)
        & (pl.col("senders") <= RETAIL_MAX_SENDERS)
        & ((pl.col("usd") / pl.col("senders")) <= RETAIL_MAX_USD_PER_SENDER)
        & (pl.col("usd") >= RETAIL_MIN_INBOUND_USD)
    ).select("to_address", "frozen_at")


def main() -> None:
    tf = eth_transfers()
    complete = eth_complete()
    sa = pl.read_parquet(INTERIM / "scam_addresses.parquet")
    scam = set(sa.filter(pl.col("accepted"))["address"].to_list())
    print(f"warehouse {tf.height:,} transfers; {len(complete):,} complete histories")

    # ---- positives: victim -> retail collection point, strictly pre-freeze ----
    retail = retail_frozen(tf)
    pos = (tf.join(retail, on="to_address", how="inner")
           .filter((pl.col("block_time") < pl.col("frozen_at"))
                   & (pl.col("amount_usd") > 0)
                   & pl.col("from_address").is_in(list(complete))
                   & pl.col("to_address").is_in(list(complete))
                   & ~pl.col("from_address").is_in(list(scam)))
           .rename({"from_address": "sender", "to_address": "destination"})
           .with_columns(pl.col("block_time").alias("event_time")))
    # Cap events per relationship. Without it one chatty relationship with 40
    # small recurring transfers outweighs a textbook case that escalated in
    # four, and the escalation signal is averaged away.
    pos = (pos.sort(["sender", "destination", "event_time"])
           .with_columns((pl.int_range(pl.len())
                          .over(["sender", "destination"]) + 1).alias("transfer_k"))
           .filter(pl.col("transfer_k") <= MAX_EVENTS_PER_PAIR)
           .with_columns(pl.lit(1).cast(pl.Int8).alias("label")))

    # ---- negatives: sender -> independently sampled ordinary wallet ----
    c3 = pl.read_parquet(INTERIM / "eth_control3_truncated.parquet")
    indep = set(c3.filter(~pl.col("truncated"))["address"].to_list())
    victims = set(pos["sender"].unique().to_list())
    neg = (tf.filter(pl.col("to_address").is_in(list(indep))
                     & (pl.col("amount_usd") > 0)
                     & pl.col("from_address").is_in(list(complete))
                     & ~pl.col("from_address").is_in(list(scam))
                     & ~pl.col("from_address").is_in(list(victims)))
           .rename({"from_address": "sender", "to_address": "destination"})
           .with_columns(pl.col("block_time").alias("event_time")))
    neg = (neg.sort(["sender", "destination", "event_time"])
           .with_columns((pl.int_range(pl.len())
                          .over(["sender", "destination"]) + 1).alias("transfer_k"))
           .filter(pl.col("transfer_k") <= MAX_EVENTS_PER_PAIR)
           .with_columns(pl.lit(0).cast(pl.Int8).alias("label")))

    print(f"positive events {pos.height:,} over {pos['destination'].n_unique():,} "
          f"destinations and {pos['sender'].n_unique():,} senders")
    print(f"negative events {neg.height:,} over {neg['destination'].n_unique():,} "
          f"destinations and {neg['sender'].n_unique():,} senders")
    if pos.height < 300 or neg.height < 300:
        sys.exit("too few events on one arm - run scripts/m10_eth_senders.py first")

    cols = ["sender", "destination", "amount_usd", "event_time", "transfer_k", "label"]
    ev = (pl.concat([pos.select(cols), neg.select(cols)], how="vertical_relaxed")
          .with_columns(pl.lit("ethereum").alias("chain"))
          .sort("event_time").with_row_index("event_id"))

    engine = FeatureEngine(tf)
    feats = engine.compute(ev)
    engine.close()

    # Match the controls, exactly as the Tron side does, for two reasons.
    #
    # The first is that it makes the two chains' numbers mean the same thing;
    # an unmatched figure is not comparable to Tron's and reporting them side by
    # side would be a quiet apples-to-oranges claim.
    #
    # The second matters more. Matching is EXACT on is_first_send_to_dest, and
    # without that the classes are separable on a confound rather than on
    # behaviour: a victim paying a fresh collection point is usually sending to
    # it for the first time, while an ordinary payment is often the latest of
    # many to a known counterparty. A model allowed to learn "first transfer =
    # scam" would score well here and be useless in a send flow, where most
    # honest first payments are also first transfers.
    unmatched = feats
    feats = match_controls_nn(feats, ratio=CONTROL_RATIO)
    print(f"\nmatched {feats.height:,} of {unmatched.height:,} events "
          f"({int((feats['label'] == 1).sum()):,} scam / "
          f"{int((feats['label'] == 0).sum()):,} ordinary)")
    print("covariate balance after matching (|SMD| < 0.1 is balanced):")
    print(standardised_mean_difference(feats))

    # Forward in time: the test set is the LATEST events, so nothing is scored
    # using behaviour that had not happened yet.
    t = feats["event_time"].to_numpy()
    cut = float(np.quantile(t, 1 - TEST_FRAC))
    is_test = t >= cut
    y = feats["label"].to_numpy()
    print(f"\nsplit at {pl.from_epoch(pl.Series([int(cut)]), 'ms')[0]:%Y-%m-%d}: "
          f"train {int((~is_test).sum()):,} events, test {int(is_test.sum()):,} "
          f"({int((y[is_test] == 1).sum()):,} scam / "
          f"{int((y[is_test] == 0).sum()):,} ordinary)")

    results, shipped = {}, None
    print("\n  arm                       ROC-AUC  PR-AUC   @1%FPR  @10%FPR")
    for name, feature_set in (("destination only", DEST_ONLY),
                              ("+ context + relationship", PAIR_FEATURES),
                              ("+ sender as well", WITH_SENDER)):
        X = feats.select(feature_set).to_numpy()
        ytr = y[~is_test]
        if len(set(ytr)) < 2:
            sys.exit("one arm of the training split has a single class")
        b = lgb.train({
            "objective": "binary", "learning_rate": 0.05, "num_leaves": 31,
            "min_data_in_leaf": 30, "feature_fraction": 0.8, "bagging_fraction": 0.8,
            "bagging_freq": 1, "lambda_l2": 1.0,
            "scale_pos_weight": (ytr == 0).sum() / max((ytr == 1).sum(), 1),
            "verbose": -1, "seed": SEED, "num_threads": 4,
        }, lgb.Dataset(X[~is_test], label=ytr, feature_name=feature_set),
            num_boost_round=300)
        s, yt = b.predict(X[is_test]), y[is_test]
        negs = np.sort(s[yt == 0])
        thr = {"elevated": float(np.quantile(negs, 0.90)),
               "high": float(np.quantile(negs, 0.99))}
        r = {"n": int(len(yt)), "n_scam": int((yt == 1).sum()),
             "n_control": int((yt == 0).sum()),
             "roc_auc": float(roc_auc_score(yt, s)),
             "pr_auc": float(average_precision_score(yt, s)),
             "tpr_at_1pct_fpr": float((s[yt == 1] >= thr["high"]).mean()),
             "tpr_at_10pct_fpr": float((s[yt == 1] >= thr["elevated"]).mean())}
        results[name] = r
        print(f"  {name:<24}  {r['roc_auc']:.4f}  {r['pr_auc']:.4f}   "
              f"{r['tpr_at_1pct_fpr']:6.1%}  {r['tpr_at_10pct_fpr']:6.1%}"
              f"   ({len(feature_set)} features)")
        if name == "+ context + relationship":
            shipped = (b, thr, r, s, yt, X)

    b, thr, r, s, yt, X = shipped
    out = {"features": PAIR_FEATURES,
           "trees": [slim(tr) for tr in b.dump_model()["tree_info"]],
           "thresholds": thr,
           "evaluation": {**r, "chain": "ethereum", "controls": "independent"}}
    path = SITE / "model_pair_eth.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    b.save_model(str(PROCESSED / "model_pair_eth.txt"))
    print(f"\nwrote {path.name} ({path.stat().st_size/1024:.0f} KB, "
          f"{len(out['trees'])} trees)")

    # The rows nearest a band edge, with LightGBM's own score, so the browser's
    # tree-walk can be checked against the library that produced it.
    edge = np.minimum(np.abs(s - thr["elevated"]), np.abs(s - thr["high"]))
    take = np.argsort(edge)[:400]
    Xt = X[is_test]
    (PROCESSED / "eth_pair_parity_sample.json").write_text(json.dumps(
        {"features": PAIR_FEATURES,
         "rows": [{"x": [float(v) for v in Xt[i]], "p": float(s[i])}
                  for i in take]}, separators=(",", ":")))

    (PROCESSED / "eth_pair_report.json").write_text(json.dumps(
        {"results": results, "split_event_time_ms": int(cut),
         "n_events": int(feats.height), "n_events_unmatched": int(unmatched.height),
         "max_events_per_pair": MAX_EVENTS_PER_PAIR, "control_ratio": CONTROL_RATIO,
         "matched": "nearest-neighbour, exact on is_first_send_to_dest",
         "complete_only": True, "controls": "independent"}, indent=2))
    print("wrote eth_pair_report.json and eth_pair_parity_sample.json")

    gain = (results["+ sender as well"]["tpr_at_1pct_fpr"]
            - results["+ context + relationship"]["tpr_at_1pct_fpr"])
    print(f"\nthe 37 sender-only features would add {gain:+.1%} recall at 1% false "
          f"alarms, for {len(WITH_SENDER) - len(PAIR_FEATURES)} more features to "
          f"keep at exact parity in the browser")


if __name__ == "__main__":
    main()

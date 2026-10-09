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
from veridis.dataset.matching import (MATCH_COVARIATES, match_controls_nn,
                                      standardised_mean_difference)
from veridis.model.quantise import quantise_matrix
from veridis.features.asof import FEATURE_COLUMNS, FEATURE_FAMILY, FeatureEngine
from veridis.features import payout

SEED = 17
TEST_FRAC = 0.30
NEEDS_THIRD_ADDRESS = {"dest_funder_fanout"}

# The second hop is computed and deliberately NOT used here, which is the one
# place the two chains legitimately differ.
#
# Tron's two-sided model gained +3.1pp of recall at a 1% budget from it, on six
# of seven seeds, so the obvious move was to do the same on Ethereum. Measured
# the same way, on the matrix this script now dumps:
#
#     37 features, no hop    @1%FPR 24.5%   @10% 60.4%   ROC 0.8693
#     39 features, with hop  @1%FPR 22.8%   @10% 60.3%   ROC 0.8699
#     difference at 1%: -1.75pp, winning 1 of 7 seeds
#
# It hurts, consistently. Ethereum's two-sided training set is 4,714 events
# against Tron's 17,415, and its payout wallets are read four pages deep rather
# than fifteen, so there is less to learn from and more noise to learn it
# through. Shipping it for symmetry would have cost recall at the operating
# point to make a feature list look tidy.
#
# The columns are still attached, so the ablation above re-runs from
# eth_pair_matrix.parquet whenever the data grows. The day Ethereum has the
# events to support it, this becomes a two-line change.
DEST_ONLY = [c for c in FEATURE_COLUMNS
             if FEATURE_FAMILY[c] == "destination" and c not in NEEDS_THIRD_ADDRESS]
PAIR_FEATURES = [c for c in FEATURE_COLUMNS
                 if FEATURE_FAMILY[c] in ("destination", "context", "relationship")
                 and c not in NEEDS_THIRD_ADDRESS]
WITH_SENDER = [c for c in FEATURE_COLUMNS if c not in NEEDS_THIRD_ADDRESS]


def with_payout(tf, frame):
    """The second hop, from the same module Tron and the Ethereum address model
    use. A payout wallet whose own history is already in the warehouse is
    complete there, so it counts as fetched and not truncated."""
    tx_path = INTERIM / "eth_payout_transfers.parquet"
    tr_path = INTERIM / "eth_payout_truncated.parquet"
    if not tx_path.exists() or not tr_path.exists():
        sys.exit("no Ethereum payout history - run scripts/m14_eth_payout_fetch.py")
    ptx = pl.read_parquet(tx_path)
    tr = pl.read_parquet(tr_path)
    trunc = dict(zip(tr["address"].to_list(), tr["truncated"].to_list()))
    for a in tf["to_address"].unique().to_list():
        trunc.setdefault(a, False)
    hop = (pl.concat([ptx.select(tf.columns), tf], how="vertical_relaxed")
           .unique(subset=["tx_hash", "to_address"]))
    f = payout.batch(frame.select("event_id", "destination", "event_time"),
                     tf, hop, trunc)
    return frame.join(f, on="event_id", how="left").with_columns(
        pl.col("payout_fanin_pit").fill_null(0),
        pl.col("payout_fanin_exact").fill_null(0))


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
    feats = with_payout(tf, engine.compute(ev))
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

    # Pick the ratio that actually balances, rather than assuming Tron's.
    #
    # Matching without replacement at 10:1 against a pool holding under 2
    # controls per positive does almost nothing: the matcher exhausts the pool
    # and takes every control, poor matches included, so the "matched" set is
    # the raw set wearing a different name. Asking for fewer, closer controls
    # is what buys balance - at the cost of negatives, and therefore of
    # resolution at low false-alarm rates. So try the ratios in order and keep
    # the largest one that balances, instead of picking either end blind.
    TARGET_SMD = 0.1
    best, best_ratio, best_worst = None, None, None
    for r in (CONTROL_RATIO, 5, 3, 2, 1):
        if best is not None and r >= best_ratio:
            continue
        cand = match_controls_nn(unmatched, ratio=r)
        bal = standardised_mean_difference(cand)
        # Judge balance on the covariates matching actually controls. amount_usd
        # is reported alongside them but never matched on, here or on Tron: it is
        # a FEATURE the model is meant to use, and victims really do send larger,
        # rounder sums than ordinary payers. Counting it as a matching failure
        # would be holding Ethereum to a bar the shipped Tron model does not meet
        # either - its own matched set sits at |SMD| 0.54 on amount.
        worst = float(bal.filter(pl.col("covariate").is_in(MATCH_COVARIATES))
                      ["abs_smd"].max())
        n_neg = int((cand["label"] == 0).sum())
        print(f"  ratio {r:>2}: {cand.height:>6,} events, {n_neg:>6,} controls, "
              f"worst |SMD| {worst:.3f}" + ("  <- balanced" if worst < TARGET_SMD else ""))
        if best_worst is None or worst < best_worst:
            best, best_ratio, best_worst = cand, r, worst
        if worst < TARGET_SMD:
            best, best_ratio, best_worst = cand, r, worst
            break
    feats = best
    print(f"\nmatched at {best_ratio}:1 - {feats.height:,} of {unmatched.height:,} "
          f"events ({int((feats['label'] == 1).sum()):,} scam / "
          f"{int((feats['label'] == 0).sum()):,} ordinary)")
    print("covariate balance (|SMD| < 0.1 is balanced; amount_usd is reported "
          "but not matched on):")
    print(standardised_mean_difference(feats))
    if best_worst >= TARGET_SMD:
        print(f"\n  WARNING: worst matched |SMD| is {best_worst:.3f}, above "
              f"{TARGET_SMD}. The classes are still partly separable on sender\n"
              f"  profile rather than on behaviour alone, so the figures below are "
              f"optimistic. More control senders in the thin strata is the fix.")

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

    # The prepared matrix, so an ablation does not have to re-derive the
    # control matching to ask whether one feature earns its place. A single
    # run's @1%FPR on 821 controls puts the threshold on about eight scores,
    # which is far too noisy to retire a feature on.
    (feats.with_columns(pl.Series("label", y), pl.Series("is_test", is_test))
     .write_parquet(PROCESSED / "eth_pair_matrix.parquet"))

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
    # Same reason as the destination sample: reference scores must come from the
    # features the serving path actually sees, which are quantised.
    Xt = quantise_matrix(X[is_test])
    s = b.predict(Xt)
    edge = np.minimum(np.abs(s - thr["elevated"]), np.abs(s - thr["high"]))
    take = np.argsort(edge)[:400]
    # NaN is written as null, not as the bare NaN Python's json emits. Bare NaN
    # is valid to Python and rejected by JSON.parse, so the sample could not be
    # read by the very browser it exists to check. null is also the right value
    # rather than a convenient one: the browser's tree walk sends null, undefined
    # and NaN down the same default branch LightGBM uses for a missing value, so
    # a missing feature is scored identically on both sides.
    def jsonable(v):
        v = float(v)
        return v if v == v and v not in (float("inf"), float("-inf")) else None

    (PROCESSED / "eth_pair_parity_sample.json").write_text(json.dumps(
        {"features": PAIR_FEATURES,
         "rows": [{"x": [jsonable(v) for v in Xt[i]], "p": float(s[i])}
                  for i in take]}, separators=(",", ":")))

    (PROCESSED / "eth_pair_report.json").write_text(json.dumps(
        {"results": results, "split_event_time_ms": int(cut),
         "n_events": int(feats.height), "n_events_unmatched": int(unmatched.height),
         "max_events_per_pair": MAX_EVENTS_PER_PAIR, "control_ratio": best_ratio,
         "worst_abs_smd": best_worst, "balanced": bool(best_worst < TARGET_SMD),
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

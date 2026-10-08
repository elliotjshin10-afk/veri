"""Train the address model the way the site uses it, and measure it honestly.

Two problems with what shipped, both found by testing rather than reading:

1. WRONG TRAINING TASK. The model was trained on victim-send events, but the
   site applies it to an address as it stands. Trained instead on point-in-time
   address probes, out-of-sample address ROC-AUC goes 0.770 -> 0.907 and recall
   at a 1% false-alarm budget goes 5.9% -> 13.0%.

2. FLATTERED EVALUATION. The published address-level figure was computed over
   every labelled address, including the ones the model trained on. Removing
   them takes the shipped model from 0.9146 to 0.7703. The number on the site
   was measured partly on material the model had already seen.

So: positives are frozen addresses probed at several horizons before the freeze
that listed them; negatives are ordinary wallets probed before a pseudo-freeze
drawn from the positives' own date distribution, so both arms face the same
calendar. Split temporally on the mark date, which also gives group integrity
for free - an address has exactly one of those.

Evaluation scores each held-out address at its own last activity, because that
is what serving does, and reports only addresses no model here has seen.
"""
from __future__ import annotations

import json, pathlib, sys
sys.path.insert(0, "src")
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from veridis.config import INTERIM, PROCESSED, SITE
from veridis.features.asof import FeatureEngine
from veridis.features import payout
from veridis.model.address_risk import DEST_FEATURES
from veridis.dataset.holdings import all_transfers, all_indexed

DAY = 86_400_000
HORIZONS = [7, 30, 90, 180]
TEST_FRAC = 0.30
SEED = 17
# dest_funder_fanout needs a third address's history and the browser cannot get
# it, so a model trained with it would be fed a default at serving time. The
# payout features need a third address too, and the browser DOES fetch it, which
# is the whole difference: see veridis/features/payout.py.
NEEDS_THIRD_ADDRESS = {"dest_funder_fanout"}
TRAIN_FEATURES = DEST_FEATURES + payout.FEATURES
BROWSER_FEATURES = ([f for f in DEST_FEATURES if f not in NEEDS_THIRD_ADDRESS]
                    + payout.FEATURES)


def payout_data(tf: pl.DataFrame, addresses) -> tuple:
    """The second hop, or a clear instruction to go and fetch it.

    Falling back to the 17-feature model when the file is missing would be the
    worst option: the site would quietly ship a weaker model and the only sign
    would be a number moving in a report nobody reads that week."""
    tx_path, tr_path = INTERIM / "payout_transfers.parquet", INTERIM / "payout_truncated.parquet"
    if not tx_path.exists() or not tr_path.exists():
        sys.exit("no payout wallet history - run scripts/m13_payout_fetch.py first")
    ptx = pl.read_parquet(tx_path)
    tr = pl.read_parquet(tr_path)
    trunc = dict(zip(tr["address"].to_list(), tr["truncated"].to_list()))
    link = payout.links(tf, addresses)
    covered = link.filter(pl.col("payout").is_in(list(trunc)))
    print(f"payout wallets: {len(trunc):,} held, covering {covered.height:,} of "
          f"{link.height:,} addresses ({covered.height/max(link.height,1):.0%})")
    return link, ptx, trunc


def attach(frame: pl.DataFrame, link, ptx, trunc) -> pl.DataFrame:
    f = payout.batch(frame.select("event_id", "destination", "event_time"),
                     link, ptx, trunc)
    return frame.join(f, on="event_id", how="left").with_columns(
        pl.col("payout_fanin_pit").fill_null(0),
        pl.col("payout_fanin_exact").fill_null(0))


def slim(t):
    def node(n):
        if "leaf_value" in n:
            return {"v": n["leaf_value"]}
        return {"f": n["split_feature"], "t": n["threshold"],
                "d": 1 if n.get("default_left") else 0,
                "l": node(n["left_child"]), "r": node(n["right_child"])}
    return node(t["tree_structure"])


def fit(X, y, cols):
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    return lgb.train({
        "objective": "binary", "learning_rate": 0.05, "num_leaves": 31,
        "min_data_in_leaf": 40, "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "bagging_freq": 1, "lambda_l2": 1.0, "scale_pos_weight": neg / max(pos, 1),
        "verbose": -1, "seed": SEED, "num_threads": 4,
    }, lgb.Dataset(X, label=y, feature_name=cols), num_boost_round=400)


def main() -> None:
    rng = np.random.default_rng(SEED)
    tf = all_transfers()
    held = all_indexed()
    span = (pl.concat([
        tf.select(pl.col("to_address").alias("address"), "block_time"),
        tf.select(pl.col("from_address").alias("address"), "block_time")],
        how="vertical_relaxed").group_by("address")
        .agg(pl.col("block_time").min().alias("first_seen"),
             pl.col("block_time").max().alias("last_seen")))
    trunc = pl.concat([pl.read_parquet(INTERIM / f"{n}.parquet")
                       for n in ("scam_truncated", "victim_truncated", "control_truncated",
                                 "leadtime_truncated", "coverage_truncated", "control2_truncated")
                       if (INTERIM / f"{n}.parquet").exists()],
                      how="vertical_relaxed").unique("address")

    scam_all = pl.read_parquet(INTERIM / "scam_addresses.parquet")
    scam = set(scam_all.filter(pl.col("accepted"))["address"].to_list())
    vic = set(pl.read_parquet(INTERIM / "victims_all.parquet")["victim_address"].to_list())
    ctrl: set[str] = set()
    for n in ("control_seeds", "control_peers", "control2_truncated"):
        p = INTERIM / f"{n}.parquet"
        if p.exists():
            ctrl |= set(pl.read_parquet(p)["address"].to_list())
    ctrl -= scam | vic

    pos = (scam_all.filter((pl.col("chain") == "tron") & pl.col("accepted")
                           & pl.col("first_reported_at").is_not_null())
           .select("address", pl.col("first_reported_at").alias("mark"))
           .filter(pl.col("address").is_in(list(held))))
    marks = np.sort(pos["mark"].to_numpy())
    # Match each control's pseudo-freeze to the positives on AGE.
    #
    # Drawing the mark uniformly looks neutral and is not. A frozen address is
    # short-lived - median 239 days old when Tether freezes it - while a random
    # mark lands long after an ordinary address first appeared, median 833 days.
    # Under 30 days old: 13% of frozen probes against 1% of ordinary. The model
    # could then separate the arms on age alone, which holds in the evaluation
    # and fails in production, where a young address is usually a new wallet.
    # On the Ethereum side this showed up as 10% of live candidates flagged
    # against a 2% measured false-positive rate; age-matching took it to 0 of
    # 207.
    first_seen = dict(zip(span["address"].to_list(), span["first_seen"].to_list()))
    pos_age = np.sort(np.array([
        r["mark"] - first_seen[r["address"]] for r in pos.iter_rows(named=True)
        if first_seen.get(r["address"]) is not None]))
    pos_age = pos_age[pos_age > 0]
    neg_rows = []
    for a in sorted(ctrl & held):
        f = first_seen.get(a)
        if f is None:
            continue
        usable = marks[marks - min(HORIZONS) * DAY > f]
        if not len(usable):
            continue
        want = f + int(rng.choice(pos_age))
        neg_rows.append({"address": a, "mark": int(usable[np.abs(usable - want).argmin()])})
    neg = pl.DataFrame(neg_rows)

    rows = []
    for df, label in ((pos, 1), (neg, 0)):
        d = (df.join(trunc, on="address", how="left").join(span, on="address", how="left")
             .with_columns(pl.col("truncated").fill_null(False))
             .filter(pl.col("first_seen").is_not_null()))
        for r in d.iter_rows(named=True):
            for h in HORIZONS:
                t = r["mark"] - h * DAY
                if t <= r["first_seen"] or (r["truncated"] and t > r["last_seen"]):
                    continue
                rows.append({"address": r["address"], "label": label,
                             "event_time": t, "mark": r["mark"]})
    pr = pl.DataFrame(rows)
    cut = float(np.quantile(pr["mark"].to_numpy(), 1 - TEST_FRAC))
    is_test = pr["mark"].to_numpy() >= cut
    holdout = set(pr.filter(pl.Series(is_test))["address"].to_list())
    print(f"probes {pr.height:,} over {pr['address'].n_unique():,} addresses; "
          f"{len(holdout):,} addresses held out")

    ev = (pr.select(pl.col("address").alias("destination"), "event_time")
          .with_columns(pl.lit("tron").alias("chain"), pl.lit("__probe__").alias("sender"),
                        pl.lit(1000.0).alias("amount_usd")).with_row_index("event_id"))
    engine = FeatureEngine(tf)
    link, ptx, trunc = payout_data(tf, pr["address"].unique().to_list())
    feats = attach(engine.compute(ev), link, ptx, trunc)
    y = pr["label"].to_numpy()
    full = fit(feats.filter(pl.Series(~is_test)).select(TRAIN_FEATURES).to_numpy(),
               y[~is_test], TRAIN_FEATURES)
    lite = fit(feats.filter(pl.Series(~is_test)).select(BROWSER_FEATURES).to_numpy(),
               y[~is_test], BROWSER_FEATURES)

    # --- evaluate the way it is served: each address at its own last activity ---
    last = (pl.concat([tf.select(pl.col("to_address").alias("address"), "block_time"),
                       tf.select(pl.col("from_address").alias("address"), "block_time")],
                      how="vertical_relaxed")
            .filter(pl.col("address").is_in(list(held)))
            .group_by("address").agg(pl.col("block_time").max().alias("last_seen")))
    probe = (last.select(pl.col("address").alias("destination"),
                         (pl.col("last_seen") + 1000).alias("event_time"))
             .with_columns(pl.lit("tron").alias("chain"), pl.lit("__probe__").alias("sender"),
                           pl.lit(1000.0).alias("amount_usd")).with_row_index("event_id"))
    served = attach(engine.compute(probe), link, ptx, trunc) \
        .with_columns(pl.col("destination").alias("address"))
    engine.close()

    def tpr_ci(y, sc, budget=0.01, n=600):
        """A 1% budget over ~750 controls puts the threshold on the top ~8
        scores. The point estimate looks far firmer than it is, so resample
        both arms and report the interval beside it."""
        out, pi, ni = [], np.where(y == 1)[0], np.where(y == 0)[0]
        for _ in range(n):
            a = rng.choice(pi, len(pi), replace=True)
            b = rng.choice(ni, len(ni), replace=True)
            t = float(np.quantile(sc[b], 1 - budget))
            out.append(float((sc[a] >= t).mean()))
        return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))

    lab = served.filter(pl.col("address").is_in(list(holdout & (scam | ctrl))))
    yt = np.array([1 if a in scam else 0 for a in lab["address"].to_list()])
    out = {}
    print(f"\nheld out: {int(yt.sum()):,} listed, {int((yt==0).sum()):,} ordinary")
    print("\n  arm        n      ROC-AUC  PR-AUC   TPR@1%FPR")
    for name, booster, cols in (("full  ", full, TRAIN_FEATURES), ("browser", lite, BROWSER_FEATURES)):
        s = booster.predict(lab.select(cols).to_numpy())
        thr = float(np.quantile(np.sort(s[yt == 0]), 0.99))
        out[name.strip()] = {
            "n": int(len(yt)), "n_scam": int(yt.sum()), "n_control": int((yt == 0).sum()),
            "roc_auc": float(roc_auc_score(yt, s)), "pr_auc": float(average_precision_score(yt, s)),
            "tpr_at_1pct_fpr": float((s[yt == 1] >= thr).mean()),
            "tpr_at_1pct_fpr_ci": list(tpr_ci(yt, s)),
            "held_out": True}
        r = out[name.strip()]
        print(f"  {name}  {len(yt):,}   {r['roc_auc']:.4f}  {r['pr_auc']:.4f}   "
              f"{r['tpr_at_1pct_fpr']:.1%}  (95% CI {r['tpr_at_1pct_fpr_ci'][0]:.0%}"
              f"-{r['tpr_at_1pct_fpr_ci'][1]:.0%})")

    # Bands on the held-out CONTROL distribution: "high" means the top 1% of
    # ordinary wallets, not a number read off the positives.
    s_lite = lite.predict(lab.select(BROWSER_FEATURES).to_numpy())
    neg_s = np.sort(s_lite[yt == 0])
    thr = {"elevated": float(np.quantile(neg_s, 0.90)), "high": float(np.quantile(neg_s, 0.99))}

    # Score EVERY indexed address, not only the held-out ones: the index the
    # site ships is built from this file. build_address_model.py used to write
    # it; when this script replaced that one it did not take the job over, and
    # 3,000 newly fetched addresses landed in the index with a null band.
    all_scores = lite.predict(served.select(BROWSER_FEATURES).to_numpy())
    pl.DataFrame({"address": served["address"].to_list(),
                  "dest_score": all_scores}).write_parquet(PROCESSED / "address_scores.parquet")
    print(f"scored {served.height:,} indexed addresses -> address_scores.parquet")

    full.save_model(str(PROCESSED / "model_dest_latest.txt"))
    lite.save_model(str(PROCESSED / "model_browser.txt"))
    (SITE / "model_dest.json").write_text(json.dumps(
        {"features": BROWSER_FEATURES, "trees": [slim(t) for t in lite.dump_model()["tree_info"]],
         "thresholds": thr, "evaluation": out["browser"]}, separators=(",", ":")))
    # The backtests quote numbers for the site, so they must run on addresses
    # this model never saw. Persist the holdout rather than leaving each script
    # to re-derive it and risk deriving it differently.
    (PROCESSED / "pit_holdout.json").write_text(json.dumps(
        {"addresses": sorted(holdout), "cut_ms": cut, "horizons": HORIZONS,
         "test_frac": TEST_FRAC, "seed": SEED}))
    (PROCESSED / "address_model_report.json").write_text(json.dumps(
        {"evaluation": out["full"], "thresholds": thr, "features": TRAIN_FEATURES,
         "trained_on": "point-in-time address probes", "holdout_addresses": len(holdout)}, indent=2))
    print(f"\nthresholds (held-out controls): {thr}")
    print(f"wrote model_dest_latest.txt, model_browser.txt, site/model_dest.json, "
          f"pit_holdout.json ({len(holdout):,} addresses)")


if __name__ == "__main__":
    main()

"""M9g - train the destination model the way it is actually used.

Two things were wrong with the shipped model, and they compound:

1. It saw 117 distinct scam destinations. Coverage has since reached 8,532.
2. It was trained on events at victim-send moments, but it is *used* to judge
   an address at an arbitrary point in its life - and evaluated at T-90. Train
   and serve disagreed about what a row even is.

This trains on point-in-time probes instead: an address, as it looked N days
before the date it was frozen. Negatives are ordinary wallets probed N days
before a pseudo-freeze date drawn from the positives' own date distribution, so
both arms face the same calendar and the same market.

Honesty of the split:
  * Temporal. Train on earlier freezes, test on later ones. Because an address
    has exactly one freeze date, splitting on it also guarantees group
    integrity - every probe of an address lands on one side.
  * The old model's 1,409 training destinations are dropped from BOTH sides, so
    the comparison is between two models on addresses neither has seen.
"""
from __future__ import annotations
import json, os, sys
sys.path.insert(0, "src")
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score
from veridis.config import INTERIM, PROCESSED
from veridis.features.asof import FeatureEngine
from veridis.model.address_risk import DEST_FEATURES
from veridis.model.train import temporal_split
from veridis.dataset.holdings import all_transfers, all_indexed

DAY = 86_400_000
HORIZONS = [7, 30, 90, 180]
TEST_FRAC = 0.30
SEED = 17
# STRICT=1 keeps every freshly-sampled control out of training and tests on
# those alone. The default run still trains on some of them (different
# addresses, separated by the temporal split), which answers "can it score
# addresses it has not seen" but not "can it score a control population it has
# not seen". Only the strict run answers the second, and that is the one the
# headline should rest on.
STRICT = os.environ.get("STRICT") == "1"


def tpr_at_fpr(y, s, budget):
    neg = np.sort(s[y == 0])
    if not len(neg):
        return float("nan"), float("nan")
    thr = float(np.quantile(neg, 1 - budget))
    return float((s[y == 1] >= thr).mean()), thr


def tpr_ci(y, s, budget, rng, n=400):
    """A 1% budget puts the threshold among the top few negative scores, so with
    a few hundred negatives the point estimate is far less certain than it
    looks. Resample both arms to say how uncertain."""
    out = []
    pi, ni = np.where(y == 1)[0], np.where(y == 0)[0]
    if len(ni) < 20:
        return (float("nan"), float("nan"))
    for _ in range(n):
        p = rng.choice(pi, len(pi), replace=True)
        q = rng.choice(ni, len(ni), replace=True)
        thr = float(np.quantile(s[q], 1 - budget))
        out.append(float((s[p] >= thr).mean()))
    return (float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5)))


def main() -> None:
    rng = np.random.default_rng(SEED)
    tf = all_transfers()
    held = all_indexed()
    span = (pl.concat([
        tf.select(pl.col("to_address").alias("address"), "block_time"),
        tf.select(pl.col("from_address").alias("address"), "block_time")], how="vertical_relaxed")
        .group_by("address").agg(pl.col("block_time").min().alias("first_seen"),
                                 pl.col("block_time").max().alias("last_seen")))
    trunc = pl.concat([pl.read_parquet(INTERIM / f"{n}.parquet")
                       for n in ("scam_truncated", "victim_truncated", "control_truncated",
                                 "leadtime_truncated", "coverage_truncated")
                       if (INTERIM / f"{n}.parquet").exists()], how="vertical_relaxed").unique("address")

    old_train = set(temporal_split(
        pl.read_parquet(PROCESSED / "events_features.parquet")).train["destination"].unique())

    # ---- positives: every frozen Tron address whose history we hold ----
    pos = (pl.read_parquet(INTERIM / "scam_addresses.parquet")
           .filter((pl.col("chain") == "tron") & pl.col("accepted")
                   & pl.col("first_reported_at").is_not_null())
           .select("address", pl.col("first_reported_at").alias("mark_at"))
           .filter(pl.col("address").is_in(list(held)) & ~pl.col("address").is_in(list(old_train))))

    # ---- negatives: ordinary wallets, pseudo-dated from the positives ----
    scam = set(pl.read_parquet(INTERIM / "scam_addresses.parquet")["address"].to_list())
    vic = set(pl.read_parquet(INTERIM / "victims_all.parquet")["victim_address"].to_list())
    ctrl: set[str] = set()
    for n in ("control_seeds", "control_peers"):
        p = INTERIM / f"{n}.parquet"
        if p.exists():
            ctrl |= set(pl.read_parquet(p)["address"].to_list())
    # Sampled by a different procedure and a different seed. Held out entirely,
    # it answers the question a same-pool split cannot: does this generalise to
    # ordinary wallets we did not draw the training negatives from?
    fresh: set[str] = set()
    fp = INTERIM / "control2_truncated.parquet"
    if fp.exists():
        fresh = set(pl.read_parquet(fp)["address"].to_list()) - scam - vic - old_train - ctrl
    ctrl = (ctrl | fresh) - scam - vic - old_train
    print(f"controls: {len(ctrl - fresh):,} original + {len(fresh):,} freshly sampled"
          + ("   [STRICT: fresh ones are test-only]" if STRICT else ""))
    marks = pos["mark_at"].to_numpy()
    neg = pl.DataFrame({"address": sorted(ctrl)}).with_columns(
        pl.Series("mark_at", rng.choice(marks, size=len(ctrl), replace=True).astype("int64")))

    rows = []
    for df, label in ((pos, 1), (neg, 0)):
        d = (df.join(trunc, on="address", how="left").join(span, on="address", how="left")
             .with_columns(pl.col("truncated").fill_null(False))
             .filter(pl.col("first_seen").is_not_null()))
        for r in d.iter_rows(named=True):
            for h in HORIZONS:
                t = r["mark_at"] - h * DAY
                if t <= r["first_seen"]:
                    continue
                if r["truncated"] and t > r["last_seen"]:
                    continue
                rows.append({"address": r["address"], "label": label, "horizon": h,
                             "event_time": t, "mark_at": r["mark_at"]})
    pr = pl.DataFrame(rows)
    print(f"probes {pr.height:,} over {pr['address'].n_unique():,} addresses "
          f"({int((pr['label']==1).sum()):,} pos / {int((pr['label']==0).sum()):,} neg)")

    ev = (pr.select(pl.col("address").alias("destination"), "event_time")
          .with_columns(pl.lit("tron").alias("chain"), pl.lit("__probe__").alias("sender"),
                        pl.lit(1000.0).alias("amount_usd")).with_row_index("event_id"))
    engine = FeatureEngine(tf)
    feats = engine.compute(ev)
    engine.close()
    X = feats.select(DEST_FEATURES).to_numpy()

    # ---- temporal split on the mark date; one address sits on one side ----
    cut = float(np.quantile(pr["mark_at"].to_numpy(), 1 - TEST_FRAC))
    is_test = pr["mark_at"].to_numpy() >= cut
    is_fresh_pre = np.array([a in fresh for a in pr["address"].to_list()])
    if STRICT:
        # Every fresh control moves to the test side regardless of its date, so
        # none of that population reaches the model during training.
        is_test = is_test | is_fresh_pre
    y = pr["label"].to_numpy()
    print(f"split at {cut:.0f}: train {int((~is_test).sum()):,} / test {int(is_test.sum()):,}"
          f"  (test positives {int((y[is_test]==1).sum()):,}, negatives {int((y[is_test]==0).sum()):,})")

    ytr, _yte = y[~is_test], y[is_test]
    pos_n, neg_n = int((ytr == 1).sum()), int((ytr == 0).sum())
    booster = lgb.train({
        "objective": "binary", "learning_rate": 0.05, "num_leaves": 31,
        "min_data_in_leaf": 40, "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "bagging_freq": 1, "lambda_l2": 1.0, "scale_pos_weight": neg_n / max(pos_n, 1),
        "verbose": -1, "seed": SEED, "num_threads": 4,
    }, lgb.Dataset(X[~is_test], label=ytr, feature_name=DEST_FEATURES), num_boost_round=400)

    old = lgb.Booster(model_file=str(PROCESSED / "model_dest_latest.txt"))
    s_new, s_old = booster.predict(X[is_test]), old.predict(X[is_test])
    yt = y[is_test]
    hz = pr["horizon"].to_numpy()[is_test]

    is_fresh = np.array([a in fresh for a in pr["address"].to_list()])

    print("\n                       ROC-AUC            TPR @ 1% FPR")
    print("horizon    n_pos/n_neg   old     new      old      new")
    report = {}
    for h in HORIZONS + ["all"]:
        m = np.ones_like(yt, bool) if h == "all" else (hz == h)
        if len(set(yt[m])) < 2:
            continue
        ao, an = roc_auc_score(yt[m], s_old[m]), roc_auc_score(yt[m], s_new[m])
        to, _ = tpr_at_fpr(yt[m], s_old[m], 0.01)
        tn, _ = tpr_at_fpr(yt[m], s_new[m], 0.01)
        report[str(h)] = {"roc_auc_old": ao, "roc_auc_new": an, "tpr1_old": to, "tpr1_new": tn,
                          "n_pos": int((yt[m] == 1).sum()), "n_neg": int((yt[m] == 0).sum())}
        lab = "all" if h == "all" else f"{h}d"
        print(f"  {lab:>5}   {int((yt[m]==1).sum()):5,}/{int((yt[m]==0).sum()):<5,}  "
              f"{ao:.3f}  {an:.3f}   {to:6.1%}  {tn:6.1%}")

    # ---- the strict test: negatives the model has never seen the likes of ----
    if fresh:
        m = is_fresh[is_test] | (yt == 1)
        if len(set(yt[m])) == 2:
            a = roc_auc_score(yt[m], s_new[m])
            t, _ = tpr_at_fpr(yt[m], s_new[m], 0.01)
            lo, hi = tpr_ci(yt[m], s_new[m], 0.01, rng)
            ao = roc_auc_score(yt[m], s_old[m])
            to, _ = tpr_at_fpr(yt[m], s_old[m], 0.01)
            print(f"\nheld-out FRESH controls only ({int((yt[m]==0).sum()):,} negatives, "
                  f"sampled separately, never trained on):")
            print(f"   old model  ROC-AUC {ao:.3f}   TPR@1%FPR {to:6.1%}")
            print(f"   new model  ROC-AUC {a:.3f}   TPR@1%FPR {t:6.1%}  (95% CI {lo:.1%}-{hi:.1%})")
            report["fresh"] = {"roc_auc_old": ao, "roc_auc_new": a, "tpr1_old": to,
                               "tpr1_new": t, "tpr1_ci": [lo, hi],
                               "n_pos": int((yt[m] == 1).sum()), "n_neg": int((yt[m] == 0).sum())}

    booster.save_model(str(PROCESSED / "model_dest_pit.txt"))
    (PROCESSED / "retrain_report.json").write_text(json.dumps(
        {"horizons": HORIZONS, "test_frac": TEST_FRAC, "results": report,
         "train_positive_addresses": int(pos.height)}, indent=2))
    print(f"\nwrote model_dest_pit.txt and retrain_report.json")


if __name__ == "__main__":
    main()

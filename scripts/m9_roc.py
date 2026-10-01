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
from veridis.dataset.holdings import all_transfers

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
    # holdings.all_transfers() exists so no script keeps its own list. This one
    # did, and silently missed control2_transfers - 3,000 freshly fetched
    # controls had no history here, so they were dropped for having no first
    # transfer and the ordinary arm stayed at 195 instead of ~750.
    transfers = all_transfers()
    print(f"warehouse: {transfers.height:,} transfers")
    span = spans(transfers)
    print(f"warehouse: {transfers.height:,} transfers")

    train_dest = set(temporal_split(
        pl.read_parquet(PROCESSED / "events_features.parquet")).train["destination"].unique())
    # The point-in-time model trains on these very addresses, so a backtest
    # including them would report on its own training data - the exact flattery
    # this project has already had to correct twice.
    keep_only = set(json.loads((PROCESSED / "pit_holdout.json").read_text())["addresses"])

    # ---- positives ----
    pos = (pl.read_parquet(INTERIM / "leadtime_labels.parquet")
           .select("address", pl.col("first_reported_at").alias("mark_at"))
           .join(pl.read_parquet(INTERIM / "leadtime_truncated.parquet"), on="address", how="left")
           .join(span, on="address", how="left")
           .with_columns(pl.col("truncated").fill_null(False))
           .filter(pl.col("first_seen").is_not_null()
                   & ~pl.col("address").is_in(list(train_dest))
                   & pl.col("address").is_in(list(keep_only))))

    # ---- negatives, with pseudo-freeze dates drawn from the positives ----
    scam = set(pl.read_parquet(INTERIM / "scam_addresses.parquet")
               .filter(pl.col("accepted"))["address"].to_list())
    victims = set(pl.read_parquet(INTERIM / "victims.parquet")["victim_address"].to_list())
    # All three control sets, not just the first two. The 2,200 freshly sampled
    # wallets were being left out, which with the holdout applied left 180
    # negatives - a 1% threshold set by the top two scores.
    ctrl: set[str] = set()
    for n in ("control_seeds", "control_peers", "control2_truncated"):
        cp = INTERIM / f"{n}.parquet"
        if cp.exists():
            ctrl |= set(pl.read_parquet(cp)["address"].to_list())
    ctrl -= scam | victims | train_dest
    # The SAME restriction as the positives. Leaving it off the negatives was
    # measuring the false-alarm rate on wallets the model trained against, and
    # it showed: FPR came out at 0.7-1.1% where the honest figure is several
    # times that. A holdout applied to one arm only is not a holdout.
    ctrl &= keep_only

    marks = pos["mark_at"].to_numpy()
    neg = (pl.DataFrame({"address": sorted(ctrl)})
           .join(pl.concat([pl.read_parquet(INTERIM / f"{n}.parquet")
                            for n in ("control_truncated", "control2_truncated")
                            if (INTERIM / f"{n}.parquet").exists()],
                           how="vertical_relaxed").unique("address"),
                 on="address", how="left")
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
    def ci(budget, n=600):
        """Resample both arms. A threshold set far into the tail rests on a
        handful of control scores, so the point estimate is less certain than
        it reads - and quoting it bare is a mistake this project has already
        made twice."""
        out = []
        for _ in range(n):
            a = rng.choice(ps, len(ps), replace=True)
            b = rng.choice(ns, len(ns), replace=True)
            out.append(float((a >= float(np.quantile(b, 1 - budget))).mean()))
        return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]

    ops = {}
    print(f"\noperating points at {head} days, both arms point-in-time")
    print(f"  ({len(ps):,} listed vs {len(ns):,} ordinary - the ordinary arm is thin,")
    print("   so each figure carries its bootstrap interval)")
    print("  FPR budget   threshold      TPR      95% CI")
    for target in (0.01, 0.02, 0.05, 0.10):
        t = float(np.quantile(ns, 1 - target))
        lo, hi = ci(target)
        ops[f"{target:.2f}"] = {"threshold": t, "tpr": float((ps >= t).mean()),
                                "fpr": float((ns >= t).mean()), "tpr_ci": [lo, hi]}
        print(f"    {target:6.0%}     {t:9.4f}   {float((ps >= t).mean()):6.1%}"
              f"   {lo:5.0%}-{hi:5.0%}")

    (PROCESSED / "leadtime_roc.json").write_text(json.dumps(
        {"horizons": out, "operating_points": ops, "headline_horizon": head,
         "n_pos_headline": int(len(ps)), "n_neg_headline": int(len(ns)),
         "threshold_high": hi, "seed": SEED,
         "generated_at": dt.datetime.utcnow().isoformat() + "Z"}, indent=2))
    print(f"\nwrote {PROCESSED / 'leadtime_roc.json'}")


if __name__ == "__main__":
    main()

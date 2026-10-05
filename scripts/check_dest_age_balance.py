"""Is the pair model separating on behaviour, or on how old the destination is?

The destination model was, badly. Its controls drew a pseudo-freeze date at
random, so ordinary addresses came out a median 833 days old against 239 for
frozen ones, and "old means ordinary" did a quarter of the work - true in the
backtest, false in production, where a young address is usually a new wallet.

The pair model is built differently: its events are real transfers to real
destinations, so nothing forces the ages apart. This measures whether that holds
rather than assuming it, and reports what rebalancing would cost. It is a
measurement, not a gate - the number it prints belongs in the written claims.

Run: python scripts/check_dest_age_balance.py
"""
from __future__ import annotations

import sys
sys.path.insert(0, "src")

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score

from veridis.config import PROCESSED
from veridis.features.asof import FEATURE_COLUMNS, FEATURE_FAMILY
from veridis.model.train import temporal_split

PAIR = [c for c in FEATURE_COLUMNS
        if FEATURE_FAMILY[c] in ("destination", "context", "relationship")
        and c != "dest_funder_fanout"]
BINS = [7, 30, 90, 365, 1095]


def smd(a: np.ndarray, y: np.ndarray) -> float:
    sd = np.sqrt((a[y == 1].var() + a[y == 0].var()) / 2)
    return float(abs(a[y == 1].mean() - a[y == 0].mean()) / sd) if sd else 0.0


def fit(df: pl.DataFrame, label: str) -> None:
    sp = temporal_split(df)
    y = sp.train["label"].to_numpy()
    yt = sp.test["label"].to_numpy()
    if len(set(y)) < 2 or len(set(yt)) < 2:
        print(f"  {label}: one class only, skipped")
        return
    b = lgb.train({
        "objective": "binary", "learning_rate": 0.05, "num_leaves": 31,
        "min_data_in_leaf": 30, "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "bagging_freq": 1, "lambda_l2": 1.0,
        "scale_pos_weight": (y == 0).sum() / max((y == 1).sum(), 1),
        "verbose": -1, "seed": 17, "num_threads": 4,
    }, lgb.Dataset(sp.train.select(PAIR).to_numpy(), label=y, feature_name=PAIR),
        num_boost_round=300)
    s = b.predict(sp.test.select(PAIR).to_numpy())
    hi = float(np.quantile(np.sort(s[yt == 0]), 0.99))
    age = sp.test["dest_age_days"].fill_null(0).to_numpy()
    print(f"  {label:<26} ROC-AUC {roc_auc_score(yt, s):.4f}   "
          f"TPR@1%FPR {(s[yt == 1] >= hi).mean():>6.1%}   "
          f"|SMD| age {smd(age, yt):.3f}   n={len(yt):,}")


def main() -> None:
    ev = pl.read_parquet(PROCESSED / "events_features.parquet")
    a = ev["dest_age_days"].fill_null(0).to_numpy()
    y = ev["label"].to_numpy()
    print(f"{ev.height:,} events  ({int((y == 1).sum()):,} scam / "
          f"{int((y == 0).sum()):,} control)")
    print(f"destination age: median {np.median(a[y == 1]):.0f}d scam vs "
          f"{np.median(a[y == 0]):.0f}d control, |SMD| {smd(a, y):.3f}\n")

    fit(ev, "as shipped")

    ev = ev.with_columns(
        pl.col("dest_age_days").fill_null(0)
        .cut(BINS, labels=[f"b{i}" for i in range(len(BINS) + 1)])
        .cast(pl.Utf8).alias("agebin"))
    pos, neg = ev.filter(pl.col("label") == 1), ev.filter(pl.col("label") == 0)
    parts = [pos]
    for r in pos.group_by("agebin").len().to_dicts():
        pool = neg.filter(pl.col("agebin") == r["agebin"])
        if pool.height:
            parts.append(pool.sample(n=min(pool.height, r["len"] * 3),
                                     seed=17, shuffle=True))
    fit(pl.concat(parts, how="vertical_relaxed"), "destination-age balanced")
    print("\nThe gap between those two lines is what the figure owes to age "
          "rather than\nto behaviour. Put it in the written claims; do not "
          "retrain on the balanced\nsubset, which is less than half the size - "
          "that buys a noisier estimate, not\na truer one.")


if __name__ == "__main__":
    main()

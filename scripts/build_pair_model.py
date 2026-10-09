"""Train and export the two-sided model the browser can run on a real send.

The page ships a destination-only model because an address lookup has no
sender. But the product is a check at the moment of sending, where both sides
and the amount are visible - and that is a different, much better model.

Measured on the same event-level test set, at a 1% false-alarm budget:

    destination only (shipped)      18 features   ROC-AUC 0.936   TPR 24.2%
    + context + relationship        38 features   ROC-AUC 0.971   TPR 35.5%
    + sender as well (everything)   75 features   ROC-AUC 0.970   TPR 44.4%

The 37 sender features buy another 8.9pp of recall and nothing in ROC-AUC, at
roughly triple the porting surface - 57 hand-written JS features to keep at
exact parity instead of 20. So this exports the 38-feature arm: it matches the
full model's ranking quality for a third of the drift risk, and the sender
features remain a measured, separable upgrade rather than an assumed one.

`dest_funder_fanout` is dropped as it always has been - it needs a third
address's history, and computing it in Python while defaulting it in the
browser would be exactly the train/serve skew this project keeps guarding
against. Everything else here comes from the two addresses' own histories:
`shared_counterparties` is the intersection of two counterparty sets we already
hold, not another fetch.
"""
import json, pathlib, sys
sys.path.insert(0, "src")
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from veridis.features.asof import FEATURE_COLUMNS, FEATURE_FAMILY
from veridis.features import payout
from veridis.model.train import temporal_split
from veridis.config import INTERIM, PROCESSED
from veridis.dataset.holdings import all_transfers

NEEDS_THIRD_ADDRESS = {"dest_funder_fanout"}
PAIR_FEATURES = ([c for c in FEATURE_COLUMNS
                  if FEATURE_FAMILY[c] in ("destination", "context", "relationship")
                  and c not in NEEDS_THIRD_ADDRESS]
                 + payout.FEATURES)


def with_payout(events: pl.DataFrame) -> pl.DataFrame:
    """The second hop, on the two-sided path.

    The destination model gained it first and the two-sided model is the one
    that actually runs when a wallet is connected, so leaving it off here would
    have meant the better model was the one fewer people see. Measured the same
    way, seven seeds: +3.1pp of recall at a 1% false-alarm budget, 36.7% to
    39.8%, winning on six of seven.
    """
    tx_path = INTERIM / "payout_transfers.parquet"
    tr_path = INTERIM / "payout_truncated.parquet"
    if not tx_path.exists() or not tr_path.exists():
        sys.exit("no payout wallet history - run scripts/m13_payout_fetch.py first")
    ptx = pl.read_parquet(tx_path)
    tr = pl.read_parquet(tr_path)
    trunc = dict(zip(tr["address"].to_list(), tr["truncated"].to_list()))
    f = payout.batch(events.select("event_id", "destination", "event_time"),
                     all_transfers(), ptx, trunc)
    return events.join(f, on="event_id", how="left").with_columns(
        pl.col("payout_fanin_pit").fill_null(0),
        pl.col("payout_fanin_exact").fill_null(0))


def slim(t):
    """Full precision on purpose: rounding thresholds to six decimals moved
    scores by up to 0.19 against Python, because a value either side of a
    rounded threshold takes the other branch across 300 trees."""
    def node(n):
        if "leaf_value" in n:
            return {"v": n["leaf_value"]}
        return {"f": n["split_feature"], "t": n["threshold"],
                "d": 1 if n.get("default_left") else 0,
                "l": node(n["left_child"]), "r": node(n["right_child"])}
    return node(t["tree_structure"])


def main() -> None:
    events = with_payout(pl.read_parquet(PROCESSED / "events_features.parquet"))
    sp = temporal_split(events)
    y = sp.train["label"].to_numpy()
    print(f"{len(PAIR_FEATURES)} features; train {sp.train.height:,} events, "
          f"test {sp.test.height:,}")

    booster = lgb.train({
        "objective": "binary", "learning_rate": 0.05, "num_leaves": 31,
        "min_data_in_leaf": 30, "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "bagging_freq": 1, "lambda_l2": 1.0,
        "scale_pos_weight": (y == 0).sum() / max((y == 1).sum(), 1),
        "verbose": -1, "seed": 17, "num_threads": 4,
    }, lgb.Dataset(sp.train.select(PAIR_FEATURES).to_numpy(), label=y,
                   feature_name=PAIR_FEATURES), num_boost_round=300)

    s = booster.predict(sp.test.select(PAIR_FEATURES).to_numpy())
    yt = sp.test["label"].to_numpy()
    neg = np.sort(s[yt == 0])
    # Bands are set on the CONTROL distribution, so "high" means "in the top
    # 1% of ordinary transfers" rather than a number read off the positives.
    thr = {"elevated": float(np.quantile(neg, 0.90)), "high": float(np.quantile(neg, 0.99))}
    ev = {"n": int(len(yt)), "n_scam": int((yt == 1).sum()), "n_control": int((yt == 0).sum()),
          "roc_auc": float(roc_auc_score(yt, s)), "pr_auc": float(average_precision_score(yt, s)),
          "tpr_at_1pct_fpr": float((s[yt == 1] >= thr["high"]).mean()),
          "tpr_at_10pct_fpr": float((s[yt == 1] >= thr["elevated"]).mean())}
    print(f"  ROC-AUC {ev['roc_auc']:.4f}   PR-AUC {ev['pr_auc']:.4f}")
    print(f"  recall at 1% false alarms: {ev['tpr_at_1pct_fpr']:.1%}")
    print(f"  recall at 10% false alarms: {ev['tpr_at_10pct_fpr']:.1%}")

    out = {"features": PAIR_FEATURES, "trees": [slim(t) for t in booster.dump_model()["tree_info"]],
           "thresholds": thr, "evaluation": ev}
    path = pathlib.Path("site/model_pair.json")
    path.write_text(json.dumps(out, separators=(",", ":")))
    booster.save_model(str(PROCESSED / "model_pair.txt"))
    print(f"\nwrote {path} ({path.stat().st_size/1024:.0f} KB, {len(out['trees'])} trees)")
    print(f"thresholds: {thr}")


if __name__ == "__main__":
    main()

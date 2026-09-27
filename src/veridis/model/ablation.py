"""Feature-family ablation: what can be detected without the destination?

This is the differentiation experiment, not a diagnostic.

Recipient-side fraud prevention - blocklists, sanctions screening, and
off-chain scam intelligence that identifies scammer infrastructure - answers
"is this destination known-bad?". Everything it can possibly know about a
transfer lives in the destination-only feature family.

The sender-side question is different: "does this look like someone being
talked into a transfer?" It is answerable with *zero* information about the
destination. If a sender-only model detects victims, the signal is
structurally unavailable to recipient-side systems, which is the entire basis
for this being a separate product rather than a worse blocklist.
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

from veridis.config import TARGET_FPR
from veridis.features.asof import FEATURE_COLUMNS, FEATURE_FAMILY
from veridis.model.evaluate import core_metrics, early_detection, threshold_at_fpr

log = logging.getLogger(__name__)

# What each arm is allowed to see.
ARMS: dict[str, tuple[str, ...]] = {
    "full": ("destination", "sender", "relationship", "context"),
    # What a recipient-side system can see: properties of the destination.
    "destination_only": ("destination", "context"),
    # Our contribution: the sender's own behaviour and their relationship to
    # this counterparty, with nothing about the destination itself.
    "sender_side_only": ("sender", "relationship", "context"),
    "sender_behaviour_only": ("sender", "context"),
}


def _cols(families: tuple[str, ...]) -> list[str]:
    return [c for c in FEATURE_COLUMNS if FEATURE_FAMILY[c] in families]


def run_ablation(
    train: pl.DataFrame,
    test: pl.DataFrame,
    target_fpr: float = TARGET_FPR,
    seed: int = 17,
) -> list[dict]:
    import lightgbm as lgb

    out = []
    for arm, families in ARMS.items():
        cols = _cols(families)
        y_tr = train["label"].to_numpy()
        pos, neg = int((y_tr == 1).sum()), int((y_tr == 0).sum())
        params = {
            "objective": "binary", "learning_rate": 0.05, "num_leaves": 31,
            "min_data_in_leaf": 30, "feature_fraction": 0.8,
            "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
            "scale_pos_weight": neg / max(pos, 1), "verbose": -1,
            "seed": seed, "num_threads": 4,
        }
        ds = lgb.Dataset(train.select(cols).to_numpy(), label=y_tr, feature_name=cols)
        booster = lgb.train(params, ds, num_boost_round=300)

        s = booster.predict(test.select(cols).to_numpy())
        y = test["label"].to_numpy()
        thr = threshold_at_fpr(y, s, target_fpr)
        m = core_metrics(y, s, thr)
        ed = early_detection(
            test.with_columns(pl.Series("raw_score", s)), thr, score_col="raw_score"
        ).to_dicts()
        out.append({
            "arm": arm,
            "n_features": len(cols),
            "pr_auc": m["pr_auc"],
            "roc_auc": m["roc_auc"],
            "base_rate": m["base_rate"],
            "lift": m["pr_auc"] / max(m["base_rate"], 1e-9),
            "fpr": m["fpr"],
            "precision": m["precision_event"],
            "recall": m["recall_event"],
            "early_k1": next((r["rate"] for r in ed if r["k"] == 1), 0.0),
            "early_k3": next((r["rate"] for r in ed if r["k"] == 3), 0.0),
            "early_k5": next((r["rate"] for r in ed if r["k"] == 5), 0.0),
        })
        log.info(
            "ablation %-22s features=%-3d PR-AUC=%.3f lift=%.1fx early_k1=%.1f%%",
            arm, len(cols), m["pr_auc"], out[-1]["lift"], 100 * out[-1]["early_k1"],
        )
    return out


def unseen_destination_test(
    train: pl.DataFrame, test: pl.DataFrame, target_fpr: float = TARGET_FPR
) -> dict:
    """The sharpest version of the claim.

    Restrict scoring to events whose destination had *no* prior inbound
    activity at all at scoring time - an address that is effectively invisible
    to any system that works by knowing something about the recipient. If the
    sender-side model still separates here, it is detecting the victim, not
    the destination.
    """
    import lightgbm as lgb

    cols = _cols(("sender", "relationship", "context"))
    y_tr = train["label"].to_numpy()
    pos, neg = int((y_tr == 1).sum()), int((y_tr == 0).sum())
    ds = lgb.Dataset(train.select(cols).to_numpy(), label=y_tr, feature_name=cols)
    booster = lgb.train({
        "objective": "binary", "learning_rate": 0.05, "num_leaves": 31,
        "min_data_in_leaf": 30, "scale_pos_weight": neg / max(pos, 1),
        "verbose": -1, "seed": 17, "num_threads": 4,
    }, ds, num_boost_round=300)

    fresh = test.filter(pl.col("dest_senders_all") <= 1)
    if fresh.height < 30 or int(fresh["label"].sum()) < 5:
        return {"n": int(fresh.height), "note": "too few fresh-destination events"}
    s = booster.predict(fresh.select(cols).to_numpy())
    y = fresh["label"].to_numpy()
    thr = threshold_at_fpr(y, s, target_fpr)
    m = core_metrics(y, s, thr)
    return {
        "n": int(fresh.height),
        "n_positive": int(y.sum()),
        "pr_auc": m["pr_auc"],
        "base_rate": m["base_rate"],
        "lift": m["pr_auc"] / max(m["base_rate"], 1e-9),
        "roc_auc": m["roc_auc"],
    }

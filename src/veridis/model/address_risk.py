"""Address-level risk: a measured model, not hand-written rules.

The first version of `/research` scored addresses with a rule set - many
payers, few payees, funds forwarded fast, high consolidation. Measured against
our own labels it was *anti-correlated*: it called 32.2% of ordinary control
wallets "behaves like a collection address" against 23.5% of addresses actually
on a public scam list.

The reason is structural, not a bug in the thresholds. An exchange deposit
wallet, a consolidation address and a scam collector have the same shape - many
unrelated payers in, one or two payees out, nothing held. Shape alone cannot
separate them, so a shape rule set fires on the whole population.

This replaces it with the destination-only arm of the model, trained on the
same labelled events and evaluated at the address level, so the number quoted
has a measured discrimination behind it and the page can state what it is
worth. The checkable facts stay - they are what a user verifies - but they are
presented as observations, not as a verdict.
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

from veridis.features.asof import FEATURE_COLUMNS, FEATURE_FAMILY

log = logging.getLogger(__name__)

DEST_FEATURES = [c for c in FEATURE_COLUMNS if FEATURE_FAMILY[c] == "destination"]


def train_destination_model(train: pl.DataFrame, seed: int = 17):
    """Destination-only model: everything knowable without a sender."""
    import lightgbm as lgb

    y = train["label"].to_numpy()
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    ds = lgb.Dataset(
        train.select(DEST_FEATURES).to_numpy(), label=y, feature_name=DEST_FEATURES)
    return lgb.train({
        "objective": "binary", "learning_rate": 0.05, "num_leaves": 31,
        "min_data_in_leaf": 30, "feature_fraction": 0.8, "bagging_fraction": 0.8,
        "bagging_freq": 1, "lambda_l2": 1.0,
        "scale_pos_weight": neg / max(pos, 1), "verbose": -1,
        "seed": seed, "num_threads": 4,
    }, ds, num_boost_round=300)


def evaluate_at_address_level(
    booster, features: pl.DataFrame, scam: set[str], control: set[str]
) -> dict:
    """How well does this separate *addresses*, not transfers?

    The model is trained on events; `/research` applies it to an address. Those
    are different tasks, so the address-level number is measured directly
    rather than inherited from the event-level metrics.
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    keep = features.filter(
        pl.col("address").is_in(list(scam | control))
    )
    if keep.height == 0:
        return {"n": 0, "note": "no labelled addresses to evaluate"}
    y = np.array([1 if a in scam else 0 for a in keep["address"].to_list()])
    if len(np.unique(y)) < 2:
        return {"n": int(keep.height), "note": "only one class present"}
    s = booster.predict(keep.select(DEST_FEATURES).to_numpy())
    return {
        "n": int(keep.height),
        "n_scam": int(y.sum()),
        "n_control": int((y == 0).sum()),
        "roc_auc": float(roc_auc_score(y, s)),
        "pr_auc": float(average_precision_score(y, s)),
        "base_rate": float(y.mean()),
        "score_median_scam": float(np.median(s[y == 1])),
        "score_median_control": float(np.median(s[y == 0])),
    }


def risk_band(score: float, thresholds: dict) -> str:
    if score >= thresholds["high"]:
        return "high"
    if score >= thresholds["elevated"]:
        return "elevated"
    return "ordinary"


def address_thresholds(scores: np.ndarray, labels: np.ndarray,
                       target_fpr: float = 0.05) -> dict:
    """Thresholds set on the control distribution, not on round numbers."""
    ctrl = scores[labels == 0]
    if ctrl.size == 0:
        return {"elevated": 0.5, "high": 0.8}
    return {
        "elevated": float(np.quantile(ctrl, 1 - 0.20)),
        "high": float(np.quantile(ctrl, 1 - target_fpr)),
    }

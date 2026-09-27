"""M4 metrics.

The headline number is not AUC. It is: at a false-positive rate a wallet will
actually tolerate, what fraction of victims do we flag by their k-th transfer,
and what share of their losses happened after that first flag.
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

log = logging.getLogger(__name__)


def threshold_at_fpr(y_true: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    """Lowest threshold whose achieved FPR on the negatives is <= target.

    A plain quantile is wrong here. Isotonic calibration maps many raw scores
    onto the same calibrated value, so the score distribution has large ties:
    the 99th percentile can sit *inside* a tied block and admit several percent
    of negatives, silently blowing the FPR budget. Scan the achievable
    thresholds instead and take the least conservative one that actually holds
    the budget, so the reported operating point is one that exists.
    """
    neg = scores[y_true == 0]
    if neg.size == 0:
        return 0.5
    candidates = np.unique(neg)
    # FPR is a step function of the threshold; walk from high to low and stop
    # before the budget is exceeded.
    best = None
    for t in candidates[::-1]:
        if (neg >= t).mean() <= target_fpr:
            best = float(t)
        else:
            break
    if best is None:
        # No threshold satisfies the budget - the score distribution is too
        # coarse (ties) for this FPR at this sample size. Fall back to the
        # least-bad achievable point and let the caller report the real FPR
        # rather than silently flagging nothing.
        log.warning(
            "no threshold achieves FPR <= %.3f; falling back to the top score",
            target_fpr,
        )
        best = float(candidates[-1])
    return best


def core_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict:
    pred = scores >= threshold
    tp = int(((y_true == 1) & pred).sum())
    fp = int(((y_true == 0) & pred).sum())
    fn = int(((y_true == 1) & ~pred).sum())
    tn = int(((y_true == 0) & ~pred).sum())
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "base_rate": float(y_true.mean()),
        "threshold": float(threshold),
        "fpr": fp / max(fp + tn, 1),
        "recall_event": tp / max(tp + fn, 1),
        "precision_event": tp / max(tp + fp, 1),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def early_detection(
    df: pl.DataFrame, threshold: float, ks: tuple[int, ...] = (1, 2, 3, 5),
    score_col: str = "score",
) -> pl.DataFrame:
    """Share of victim relationships flagged at or before transfer k.

    A victim relationship is one (sender, scam destination) pair; transfer_k is
    the position of each transfer within it.
    """
    pos = df.filter(pl.col("label") == 1)
    if pos.height == 0:
        return pl.DataFrame({"k": list(ks), "victims": 0, "flagged": 0, "rate": 0.0})
    flagged = pos.with_columns((pl.col(score_col) >= threshold).alias("hit"))
    rows = []
    total = flagged.select(pl.struct("sender", "destination").n_unique()).item()
    for k in ks:
        sub = flagged.filter(pl.col("transfer_k") <= k)
        hit = (
            sub.filter(pl.col("hit"))
            .select(pl.struct("sender", "destination").n_unique())
            .item()
        )
        rows.append({"k": k, "victims": total, "flagged": hit,
                     "rate": hit / max(total, 1)})
    return pl.DataFrame(rows)


def dollars_protected(df: pl.DataFrame, threshold: float,
                      score_col: str = "score") -> dict:
    """Share of victim losses that occurred at or after our first flag.

    Money moved in transfers before the first flag is money we did not save;
    everything from the first flag onward is what a blocking interstitial had
    the chance to stop.
    """
    pos = (
        df.filter(pl.col("label") == 1)
        .with_columns((pl.col(score_col) >= threshold).alias("hit"))
        .sort(["sender", "destination", "transfer_k"])
    )
    if pos.height == 0:
        return {"total_usd": 0.0, "protected_usd": 0.0, "share": 0.0}
    # First flagged position within each victim relationship.
    first_hit = (
        pos.filter(pl.col("hit"))
        .group_by(["sender", "destination"])
        .agg(pl.col("transfer_k").min().alias("first_flag_k"))
    )
    joined = pos.join(first_hit, on=["sender", "destination"], how="left")
    protected = joined.filter(
        pl.col("first_flag_k").is_not_null()
        & (pl.col("transfer_k") >= pl.col("first_flag_k"))
    )["amount_usd"].sum()
    total = pos["amount_usd"].sum()
    return {
        "total_usd": float(total),
        "protected_usd": float(protected or 0.0),
        "share": float((protected or 0.0) / max(total, 1e-9)),
        "victims_ever_flagged": int(first_hit.height),
        "victims_total": int(
            pos.select(pl.struct("sender", "destination").n_unique()).item()
        ),
    }


def family_importance(booster, feature_names: list[str], families: dict) -> pl.DataFrame:
    """Importance aggregated by feature family.

    The brief's failure check: if sender-behaviour features contribute nothing,
    we have rebuilt a blocklist with extra steps.
    """
    gain = booster.feature_importance(importance_type="gain")
    df = pl.DataFrame({
        "feature": feature_names,
        "gain": gain.astype(float),
        "family": [families.get(f, "other") for f in feature_names],
    })
    total = max(float(df["gain"].sum()), 1e-9)
    return (
        df.group_by("family")
        .agg(pl.col("gain").sum())
        .with_columns((pl.col("gain") / total).alias("share"))
        .sort("gain", descending=True)
    )

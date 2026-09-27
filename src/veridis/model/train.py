"""M4 - training with temporal + grouped splitting and probability calibration."""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression

from veridis.config import PROCESSED, SEED, SPLIT_QUANTILE, TARGET_FPR
from veridis.features.asof import FEATURE_COLUMNS, FEATURE_FAMILY
from veridis.model.evaluate import (
    core_metrics, dollars_protected, early_detection, family_importance,
    threshold_at_fpr,
)

log = logging.getLogger(__name__)


@dataclass
class SplitResult:
    train: pl.DataFrame
    test: pl.DataFrame
    cutoff: int          # label-time boundary (scam reported before/after)
    event_cutoff: int    # transfer-time boundary, same quantile


def temporal_split(events: pl.DataFrame, quantile: float = SPLIT_QUANTILE) -> SplitResult:
    """Split on label time, not randomly.

    Positives are split by `frozen_at` (when the scam was reported), which also
    enforces group integrity: every event for one scam address shares a
    `frozen_at`, so a scam operation lands entirely on one side. Negatives are
    split on their own event time so the two sides cover the same period.
    """
    pos = events.filter(pl.col("label") == 1)
    # Two aligned cutoffs at the same quantile. Positives split on when the
    # scam was *reported*, which is what keeps a scam operation on one side.
    # Negatives split on transfer time, because a report date is systematically
    # later than the transfers it describes - using one cutoff for both would
    # push nearly every control into train and starve the test set of
    # negatives, making test-set FPR meaningless.
    cutoff = int(pos["frozen_at"].quantile(quantile))
    event_cutoff = int(pos["event_time"].quantile(quantile))
    train = events.filter(
        ((pl.col("label") == 1) & (pl.col("frozen_at") < cutoff))
        | ((pl.col("label") == 0) & (pl.col("event_time") < event_cutoff))
    )
    test = events.filter(
        ((pl.col("label") == 1) & (pl.col("frozen_at") >= cutoff))
        | ((pl.col("label") == 0) & (pl.col("event_time") >= event_cutoff))
    )
    return SplitResult(train, test, cutoff, event_cutoff)


def assert_group_integrity(split: SplitResult) -> dict:
    """No scam address may appear on both sides of the split."""
    tr = set(split.train.filter(pl.col("label") == 1)["destination"].unique())
    te = set(split.test.filter(pl.col("label") == 1)["destination"].unique())
    overlap = tr & te
    if overlap:
        raise AssertionError(
            f"{len(overlap)} scam addresses appear in both train and test - "
            "grouped splitting is broken"
        )
    v_tr = set(split.train.filter(pl.col("label") == 1)["sender"].unique())
    v_te = set(split.test.filter(pl.col("label") == 1)["sender"].unique())
    return {
        "train_scam_addresses": len(tr),
        "test_scam_addresses": len(te),
        "scam_address_overlap": 0,
        # Reported, not fatal: a person can be victimised by two operations.
        "victim_overlap": len(v_tr & v_te),
    }


def train_model(train: pl.DataFrame, seed: int = SEED) -> lgb.Booster:
    X = train.select(FEATURE_COLUMNS).to_numpy()
    y = train["label"].to_numpy()
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    params = {
        "objective": "binary",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        # Keep the real base rate; do not resample it away.
        "scale_pos_weight": neg / max(pos, 1),
        "verbose": -1,
        "seed": seed,
        "num_threads": 4,
    }
    ds = lgb.Dataset(X, label=y, feature_name=FEATURE_COLUMNS)
    return lgb.train(params, ds, num_boost_round=400)


def calibrate(booster: lgb.Booster, valid: pl.DataFrame):
    """Platt (logistic) calibration, fitted on held-out data.

    Two deliberate choices:

    * **Platt, not isotonic.** Isotonic is only weakly monotone: at this sample
      size it collapses scores into a handful of steps, and the top step alone
      held more than 1% of negatives - so *no* threshold could satisfy a 1% FPR
      budget and the model was forced to flag nothing. Platt is strictly
      monotone, so it preserves the ranking that the operating point is chosen
      on.
    * **Held out.** Calibrating on the rows the trees were fit to produces
      overconfident probabilities, because the model has already memorised
      those labels.
    """
    from sklearn.linear_model import LogisticRegression

    raw = booster.predict(valid.select(FEATURE_COLUMNS).to_numpy())
    y = valid["label"].to_numpy()
    if len(np.unique(y)) < 2:
        return None
    # Fit on the LOGIT of the model's output, not the output itself.
    #
    # LightGBM already returns a probability, so regressing a logistic on it
    # asks a sigmoid to correct a sigmoid and it cannot: measured calibration
    # error was 0.098 with a worst-bin gap of 0.285. Platt scaling is defined
    # on the score's logit, and fitting it there drops ECE to 0.015.
    # Regularised so the fit cannot saturate and report 1.000 for everything.
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(_as_logit(raw), y)
    return _LogitPlatt(lr)


def _as_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p)).reshape(-1, 1)


class _LogitPlatt:
    """Platt scaling on the score's logit, with the usual predict_proba shape."""

    def __init__(self, lr) -> None:
        self.lr = lr

    def predict_proba(self, raw: np.ndarray) -> np.ndarray:
        return self.lr.predict_proba(_as_logit(np.asarray(raw).ravel()))


def calibration_split(train: pl.DataFrame, frac: float = 0.25) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Hold out the most recent slice of train for calibration."""
    cutoff = int(train["event_time"].quantile(1 - frac))
    fit = train.filter(pl.col("event_time") < cutoff)
    cal = train.filter(pl.col("event_time") >= cutoff)
    if cal.height == 0 or int(cal["label"].sum()) == 0 or fit.height == 0:
        return train, train
    return fit, cal


# A base rate this high means the control pool is too thin for the metrics to
# mean what they normally mean: PR-AUC approaches 1 trivially, and an FPR
# budget computed on a handful of negatives is not a budget. Seen in practice
# when a larger victim fetch landed before the matching control fetch.
UNHEALTHY_BASE_RATE = 0.25


def evaluate_all(
    booster: lgb.Booster,
    iso: IsotonicRegression | None,
    split: SplitResult,
    target_fpr: float = TARGET_FPR,
) -> dict:
    def scored(df: pl.DataFrame) -> pl.DataFrame:
        raw = booster.predict(df.select(FEATURE_COLUMNS).to_numpy())
        if iso is None:
            prob = raw
        elif hasattr(iso, "predict_proba"):
            prob = iso.predict_proba(raw.reshape(-1, 1))[:, 1]
        else:
            prob = iso.predict(raw)
        return df.with_columns(
            pl.Series("raw_score", raw), pl.Series("score", prob)
        )

    te = scored(split.test)
    y = te["label"].to_numpy()
    # The operating point is chosen on the raw model ranking, not on the
    # calibrated probability: calibration is for telling a customer how likely
    # this is, thresholding is for holding an FPR budget, and tying the two
    # together lets calibration granularity dictate the achievable FPR.
    thr = threshold_at_fpr(y, te["raw_score"].to_numpy(), target_fpr)
    s = te["raw_score"].to_numpy()

    # Robustness check against label-selection bias.
    #
    # Scam addresses were *selected* for looking like retail collection points
    # (>= 5 distinct inbound senders), so positives have that property by
    # construction while ordinary controls need not. A model could ride that
    # artefact instead of behaviour. Re-scoring against only those negatives
    # whose destination already had >= 5 distinct senders at scoring time puts
    # both classes behind the same structural bar.
    hard = te.filter((pl.col("label") == 1) | (pl.col("dest_senders_all") >= 5))
    hard_block = None
    if int((hard["label"] == 0).sum()) >= 50:
        hy = hard["label"].to_numpy()
        hs = hard["raw_score"].to_numpy()
        hthr = threshold_at_fpr(hy, hs, target_fpr)
        hard_block = {
            "n_negatives": int((hy == 0).sum()),
            "metrics": core_metrics(hy, hs, hthr),
            "early_detection": early_detection(hard, hthr).to_dicts(),
        }

    base_rate = float(y.mean())
    warnings: list[str] = []
    if base_rate > UNHEALTHY_BASE_RATE:
        warnings.append(
            f"base rate {base_rate:.2%} is far above a realistic prevalence: the "
            f"control pool is too thin, so PR-AUC is inflated and the FPR budget "
            f"is computed on too few negatives. Fetch more controls before "
            f"reporting these numbers."
        )
        log.warning("UNRELIABLE METRICS - %s", warnings[-1])

    # The operationally critical slice, and the one free of the control-
    # construction artefact.
    #
    # A blocking interstitial fires on a *first* transfer to an address the
    # user has never paid before. Restricting both classes to those events
    # makes `is_first_send_to_dest` constant, so it carries no information and
    # the model has to separate on behaviour alone - which also neutralises the
    # fact that our control pool over-represents repeat payment pairs.
    first_send = te.filter(pl.col("is_first_send_to_dest") == 1)
    first_block = None
    if int((first_send["label"] == 0).sum()) >= 50 and int(first_send["label"].sum()) >= 20:
        fy = first_send["label"].to_numpy()
        fs = first_send["raw_score"].to_numpy()
        fthr = threshold_at_fpr(fy, fs, target_fpr)
        first_block = {
            "n": int(first_send.height),
            "n_positive": int(fy.sum()),
            "n_negative": int((fy == 0).sum()),
            "metrics": core_metrics(fy, fs, fthr),
            "early_detection": early_detection(
                first_send, fthr, score_col="raw_score").to_dicts(),
        }

    return {
        "cutoff_ms": split.cutoff,
        "warnings": warnings,
        "first_send_only": first_block,
        "event_cutoff_ms": split.event_cutoff,
        "n_train": split.train.height,
        "n_test": split.test.height,
        "groups": assert_group_integrity(split),
        "metrics": core_metrics(y, s, thr),
        "calibration": _calibration_quality(y, te["score"].to_numpy()),
        "early_detection": early_detection(te, thr, score_col="raw_score").to_dicts(),
        "dollars": dollars_protected(te, thr, score_col="raw_score"),
        "hard_negatives": hard_block,
        "family_importance": family_importance(
            booster, FEATURE_COLUMNS, FEATURE_FAMILY
        ).to_dicts(),
        "top_features": pl.DataFrame({
            "feature": FEATURE_COLUMNS,
            "gain": booster.feature_importance("gain").astype(float),
        }).sort("gain", descending=True).head(15).to_dicts(),
    }, te


def _calibration_quality(y: np.ndarray, prob: np.ndarray, bins: int = 5) -> dict:
    """Brier score plus a reliability table: predicted vs observed frequency."""
    brier = float(np.mean((prob - y) ** 2))
    edges = np.quantile(prob, np.linspace(0, 1, bins + 1))
    edges = np.unique(edges)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob >= lo) & (prob <= hi)
        if m.sum() == 0:
            continue
        rows.append({
            "bin": f"{lo:.2f}-{hi:.2f}",
            "n": int(m.sum()),
            "predicted": float(prob[m].mean()),
            "observed": float(y[m].mean()),
        })
    return {"brier": brier, "reliability": rows,
            "max_prob": float(prob.max()), "mean_prob": float(prob.mean())}


def save(booster: lgb.Booster, iso, report: dict, version: str) -> None:
    booster.save_model(str(PROCESSED / f"model_{version}.txt"))
    if iso is not None:
        import pickle
        (PROCESSED / f"calibrator_{version}.pkl").write_bytes(pickle.dumps(iso))
    (PROCESSED / f"report_{version}.json").write_text(json.dumps(report, indent=2, default=str))

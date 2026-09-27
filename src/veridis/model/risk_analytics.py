"""Risk analytics beyond a single score.

Four things a risk function expects that a raw classifier does not give you:

1. **Sequential evidence.** A victim relationship is a sequence, not one event.
   Scoring each transfer independently throws away the fact that the second
   borderline transfer to the same new counterparty is more alarming than the
   first. Accumulating log-likelihood ratios across the relationship is the
   standard sequential test (Wald's SPRT), and it is the right shape for a
   product whose whole claim is early detection.
2. **Uncertainty.** A detection rate computed on a few hundred relationships
   needs an interval, and the bootstrap gives one without distributional
   assumptions.
3. **Cost, not accuracy.** Nobody buys recall. A wallet trades dollars kept
   against customers interrupted, so the curve that matters is dollars
   protected per interruption.
4. **Stability.** Feature drift between the train and test periods is the
   usual reason a model that validated well degrades in production; PSI is the
   standard detector, and calibration error the standard summary of whether a
   probability means what it says.
"""
from __future__ import annotations

import logging

import numpy as np
import polars as pl

log = logging.getLogger(__name__)


# ------------------------------------------------------------ 1. sequential
def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def fit_llr(train_scores: np.ndarray, train_labels: np.ndarray, bins: int = 20):
    """Continuous log-likelihood ratio, fitted on train.

    Two earlier versions were wrong in instructive ways.

    `logit(score) - logit(prior)` is not a likelihood ratio: with a
    miscalibrated score an ordinary transfer contributes slightly *positive*
    evidence, so a long legitimate relationship drifts upward and drags the
    threshold past anything a single victim transfer can reach. Detection at
    k=1 collapsed to zero.

    Binning the ratio empirically fixed the sign but introduced a second
    failure: 20 bins is far too coarse for a 1% threshold, so whole bins sit
    inside or outside it and k=1 still read zero - the same discretisation trap
    that made isotonic calibration unusable for threshold-setting.

    So: refit calibration on train (logistic on the model's logit), and take
    the LLR as `logit(posterior) - logit(prior)`. That is a genuine likelihood
    ratio, strictly monotone in the model score, which means the sequential
    statistic at k=1 ranks identically to the point score - as it must, since
    at the first transfer the two arms hold exactly the same information.
    """
    from sklearn.linear_model import LogisticRegression

    z = _logit(train_scores).reshape(-1, 1)
    lr = LogisticRegression(C=1.0, max_iter=1000).fit(z, train_labels)
    prior = float(train_labels.mean())
    # Take the coefficients directly rather than going through predict_proba.
    # Platt's posterior logit is exactly `a*z + b`, so the LLR is linear in the
    # score's logit; round-tripping through a probability and back re-clips at
    # the tails and flattens the statistic exactly where it should be most
    # decisive.
    return {
        "a": float(lr.coef_[0][0]),
        "b": float(lr.intercept_[0]),
        "prior_logit": float(_logit(np.array([prior]))[0]),
    }


def apply_llr(scores: np.ndarray, model: dict) -> np.ndarray:
    z = _logit(np.asarray(scores)).ravel()
    return model["a"] * z + model["b"] - model["prior_logit"]


def sequential_scores(
    df: pl.DataFrame, llr_model: dict,
    score_col: str = "score", decay: float = 1.0,
) -> pl.DataFrame:
    """Accumulate evidence along each (sender, destination) relationship.

    Causal by construction: the statistic at transfer k uses transfers 1..k
    only, which is the information the live product would hold. `decay` < 1
    discounts older transfers so a relationship is not condemned forever.
    """
    ordered = df.sort(["sender", "destination", "event_time", "transfer_k"])
    llr = apply_llr(ordered[score_col].to_numpy(), llr_model)
    out = ordered.with_columns(pl.Series("_llr", llr))
    if decay >= 1.0:
        out = out.with_columns(
            pl.col("_llr").cum_sum().over(["sender", "destination"]).alias("cum_llr"))
    else:
        vals: list[float] = []
        for (_s, _d), grp in out.group_by(["sender", "destination"], maintain_order=True):
            acc = 0.0
            for v in grp["_llr"].to_list():
                acc = acc * decay + v
                vals.append(acc)
        out = out.with_columns(pl.Series("cum_llr", vals))
    return out.drop("_llr")


def compare_sequential(
    scored: pl.DataFrame, train_scored: pl.DataFrame, target_fpr: float,
    ks: tuple[int, ...] = (1, 2, 3, 5), decay: float = 1.0,
) -> dict:
    """Does accumulating evidence beat scoring each transfer alone?

    The likelihood ratio is fitted on train and applied to test, so the
    sequential arm gets no information the point arm lacks. Both arms are held
    to the same false-positive budget on the same control relationships.
    """
    from veridis.model.evaluate import early_detection, threshold_at_fpr

    llr_model = fit_llr(train_scored["score"].to_numpy(),
                        train_scored["label"].to_numpy())
    seq = sequential_scores(scored, llr_model, score_col="score", decay=decay)
    y = seq["label"].to_numpy()

    point_thr = threshold_at_fpr(y, seq["raw_score"].to_numpy(), target_fpr)
    seq_thr = threshold_at_fpr(y, seq["cum_llr"].to_numpy(), target_fpr)

    point = {r["k"]: r["rate"] for r in early_detection(
        seq, point_thr, ks, score_col="raw_score").to_dicts()}
    cum = {r["k"]: r["rate"] for r in early_detection(
        seq, seq_thr, ks, score_col="cum_llr").to_dicts()}

    neg = y == 0
    return {
        "decay": decay,
        "point_threshold": float(point_thr),
        "sequential_threshold": float(seq_thr),
        "point_fpr": float((seq["raw_score"].to_numpy()[neg] >= point_thr).mean()),
        "sequential_fpr": float((seq["cum_llr"].to_numpy()[neg] >= seq_thr).mean()),
        "early_point": {str(k): point.get(k, 0.0) for k in ks},
        "early_sequential": {str(k): cum.get(k, 0.0) for k in ks},
        "delta": {str(k): cum.get(k, 0.0) - point.get(k, 0.0) for k in ks},
        "median_llr_victim": float(np.median(apply_llr(
            scored.filter(pl.col("label") == 1)["score"].to_numpy(), llr_model))),
        "median_llr_control": float(np.median(apply_llr(
            scored.filter(pl.col("label") == 0)["score"].to_numpy(), llr_model))),
    }


# ------------------------------------------------------------- 2. bootstrap
def bootstrap_early_detection(
    scored: pl.DataFrame, threshold: float, ks: tuple[int, ...] = (1, 2, 3, 5),
    n_boot: int = 400, seed: int = 17, score_col: str = "raw_score",
) -> dict:
    """Resample victim *relationships*, not events.

    Events inside one relationship are not independent, so resampling events
    would give intervals that are far too tight.
    """
    pos = scored.filter(pl.col("label") == 1).with_columns(
        (pl.col(score_col) >= threshold).alias("hit"))
    if pos.height == 0:
        return {}
    pairs = (pos.group_by(["sender", "destination"])
             .agg([(pl.col("hit") & (pl.col("transfer_k") <= k)).any().alias(f"k{k}")
                   for k in ks]))
    n = pairs.height
    cols = {k: pairs[f"k{k}"].to_numpy().astype(float) for k in ks}
    rng = np.random.default_rng(seed)
    out: dict[str, dict] = {}
    for k in ks:
        stats = np.array([cols[k][rng.integers(0, n, n)].mean() for _ in range(n_boot)])
        out[str(k)] = {
            "rate": float(cols[k].mean()),
            "lo": float(np.percentile(stats, 2.5)),
            "hi": float(np.percentile(stats, 97.5)),
            "n_relationships": int(n),
        }
    return out


def bootstrap_metric(y: np.ndarray, s: np.ndarray, fn, n_boot: int = 400,
                     seed: int = 17) -> dict:
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(fn(y[idx], s[idx]))
    if not vals:
        return {}
    return {"value": float(fn(y, s)), "lo": float(np.percentile(vals, 2.5)),
            "hi": float(np.percentile(vals, 97.5))}


# ------------------------------------------------------------------ 3. cost
def cost_curve(scored: pl.DataFrame, budgets: list[float],
               score_col: str = "raw_score") -> list[dict]:
    """Dollars protected against customers interrupted, at each threshold.

    Deliberately free of an assumed cost per interruption - that number is the
    customer's, not ours. Reporting dollars protected *per interruption* lets a
    wallet put its own price on friction and read the trade directly.
    """
    from veridis.model.evaluate import threshold_at_fpr

    y = scored["label"].to_numpy()
    s = scored[score_col].to_numpy()
    pos = scored.filter(pl.col("label") == 1)
    total_usd = float(pos["amount_usd"].sum())
    rows = []
    for b in budgets:
        thr = threshold_at_fpr(y, s, b)
        flagged = pos.with_columns((pl.col(score_col) >= thr).alias("hit"))
        first = (flagged.filter(pl.col("hit"))
                 .group_by(["sender", "destination"])
                 .agg(pl.col("transfer_k").min().alias("fk")))
        joined = flagged.join(first, on=["sender", "destination"], how="left")
        protected = float(joined.filter(
            pl.col("fk").is_not_null() & (pl.col("transfer_k") >= pl.col("fk"))
        )["amount_usd"].sum() or 0.0)
        fp = int(((y == 0) & (s >= thr)).sum())
        n_neg = int((y == 0).sum())
        rows.append({
            "budget": b,
            "fpr": fp / max(n_neg, 1),
            "interruptions": fp,
            "protected_usd": protected,
            "protected_share": protected / max(total_usd, 1e-9),
            "usd_per_interruption": protected / fp if fp else None,
        })
    return rows


# ------------------------------------------------------- 4. drift + calibration
def population_stability_index(
    train: pl.DataFrame, test: pl.DataFrame, features: list[str], bins: int = 10
) -> list[dict]:
    """PSI per feature between the train and test periods.

    Convention: < 0.1 stable, 0.1-0.25 moderate shift, > 0.25 material shift.
    A material shift is not automatically a fault - the test period is later by
    construction - but it says which features the model should be trusted least
    on out of period.
    """
    out = []
    for f in features:
        a = train[f].fill_null(0).to_numpy().astype(float)
        b = test[f].fill_null(0).to_numpy().astype(float)
        if np.allclose(a, a[0] if len(a) else 0):
            continue
        edges = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))
        if len(edges) < 3:
            continue
        pa = np.histogram(a, bins=edges)[0].astype(float)
        pb = np.histogram(b, bins=edges)[0].astype(float)
        pa = np.clip(pa / max(pa.sum(), 1), 1e-6, None)
        pb = np.clip(pb / max(pb.sum(), 1), 1e-6, None)
        psi = float(np.sum((pb - pa) * np.log(pb / pa)))
        out.append({"feature": f, "psi": psi,
                    "verdict": ("stable" if psi < 0.1
                                else "moderate shift" if psi < 0.25
                                else "material shift")})
    return sorted(out, key=lambda r: -r["psi"])


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> dict:
    """ECE plus the worst single bin, which is what actually bites."""
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return {}
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, len(edges) - 2)
    ece, worst, rows = 0.0, 0.0, []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        conf, acc = float(p[m].mean()), float(y[m].mean())
        w = m.sum() / len(y)
        ece += w * abs(conf - acc)
        worst = max(worst, abs(conf - acc))
        rows.append({"bin": b, "n": int(m.sum()), "predicted": conf, "observed": acc})
    return {"ece": float(ece), "max_gap": float(worst), "bins": rows}

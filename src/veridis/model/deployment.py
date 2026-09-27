"""What the model does at deployment prevalence, not evaluation prevalence.

Every metric elsewhere in this repo is measured on a matched population where
roughly one event in seven is a victim transfer. No wallet looks like that. On
real traffic the prevalence of scam sends is somewhere around 1 in 1,000 to
1 in 10,000, and precision is brutally sensitive to that number while recall
and false-positive rate are not.

Concretely: a model with 44% recall at a 1% false-positive rate looks excellent
at a 15% base rate (precision 0.89) and very different at 0.1% (precision
0.04 - roughly 23 false alerts for every real catch). Both describe the same
model. Only the second describes what a customer experiences, so it is the one
that has to be on the table before anybody signs anything.

TPR and FPR are conditional on the true class and therefore invariant to
prevalence; precision is not. That is the whole calculation.
"""
from __future__ import annotations

import logging

import numpy as np

log = logging.getLogger(__name__)

# Plausible deployment prevalences for scam sends in consumer stablecoin flow.
DEPLOY_PREVALENCE = (0.01, 0.003, 0.001, 0.0003, 0.0001)


def precision_at_prevalence(tpr: float, fpr: float, prevalence: float) -> float:
    """PPV under a different class balance, holding TPR and FPR fixed."""
    num = tpr * prevalence
    den = num + fpr * (1.0 - prevalence)
    return float(num / den) if den > 0 else 0.0


def deployment_table(
    tpr: float, fpr: float, prevalences=DEPLOY_PREVALENCE, per: int = 100_000
) -> list[dict]:
    """What an operating point costs and catches per `per` real sends."""
    rows = []
    for pi in prevalences:
        ppv = precision_at_prevalence(tpr, fpr, pi)
        true_pos = tpr * pi * per
        false_pos = fpr * (1 - pi) * per
        rows.append({
            "prevalence": pi,
            "precision": ppv,
            "alerts_per_n": true_pos + false_pos,
            "true_alerts_per_n": true_pos,
            "false_alerts_per_n": false_pos,
            "false_per_true": (false_pos / true_pos) if true_pos > 0 else None,
            "per": per,
        })
    return rows


def shift_prior(scores: np.ndarray, eval_prior: float, deploy_prior: float) -> np.ndarray:
    """Re-express calibrated probabilities under a different prior.

    A probability calibrated at a 15% base rate does not mean the same thing on
    traffic where the base rate is 0.1%. Shifting the odds by the ratio of the
    priors keeps the ranking identical and makes the number mean what it says
    where it is actually used.
    """
    p = np.clip(np.asarray(scores, dtype=float), 1e-9, 1 - 1e-9)
    odds = p / (1 - p)
    ratio = (deploy_prior / (1 - deploy_prior)) / (eval_prior / (1 - eval_prior))
    shifted = odds * ratio
    return shifted / (1 + shifted)


def operating_points_at_prevalence(
    y: np.ndarray, scores: np.ndarray, budgets, prevalence: float,
    per: int = 100_000,
) -> list[dict]:
    """Sweep the FPR budget and report what each costs at a real prevalence."""
    from veridis.model.evaluate import threshold_at_fpr

    out = []
    for b in budgets:
        thr = threshold_at_fpr(y, scores, b)
        pred = scores >= thr
        tp = int(((y == 1) & pred).sum()); fn = int(((y == 1) & ~pred).sum())
        fp = int(((y == 0) & pred).sum()); tn = int(((y == 0) & ~pred).sum())
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        ppv = precision_at_prevalence(tpr, fpr, prevalence)
        true_alerts = tpr * prevalence * per
        false_alerts = fpr * (1 - prevalence) * per
        out.append({
            "budget": b, "threshold": float(thr), "tpr": tpr, "fpr": fpr,
            "precision_at_prevalence": ppv,
            "alerts_per_n": true_alerts + false_alerts,
            "false_per_true": (false_alerts / true_alerts) if true_alerts else None,
        })
    return out


def deployment_economics(
    scored, budgets, prevalence: float, per: int = 100_000,
    score_col: str = "raw_score",
) -> list[dict]:
    """What an operating point protects and costs on real traffic.

    Precision is the wrong lens for this product. At a realistic prevalence
    most alerts are false, and precision therefore looks terrible - but a false
    alert costs a few seconds of friction while a true one stops a five-figure
    irreversible loss. The decision number is dollars protected per false
    alert, and the customer supplies the price of friction.

    Victim *relationships* are the unit, not events: catching a relationship
    once is enough, so the alert rate is scaled by the average number of
    transfers per relationship.
    """
    import polars as pl

    from veridis.model.evaluate import threshold_at_fpr

    y = scored["label"].to_numpy()
    s = scored[score_col].to_numpy()
    pos = scored.filter(pl.col("label") == 1)
    n_rel = pos.select(pl.struct("sender", "destination").n_unique()).item()
    transfers_per_rel = pos.height / max(n_rel, 1)

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
        caught = first.height
        per_rel = protected / max(caught, 1)

        tpr = float(((y == 1) & (s >= thr)).sum() / max((y == 1).sum(), 1))
        fpr = float(((y == 0) & (s >= thr)).sum() / max((y == 0).sum(), 1))
        true_alerts = tpr * prevalence * per / max(transfers_per_rel, 1e-9)
        false_alerts = fpr * (1 - prevalence) * per
        dollars = true_alerts * per_rel
        rows.append({
            "budget": b, "recall": tpr, "fpr": fpr,
            "relationships_caught": caught,
            "usd_per_relationship": per_rel,
            "precision_at_prevalence": precision_at_prevalence(tpr, fpr, prevalence),
            "true_alerts_per_n": true_alerts,
            "false_alerts_per_n": false_alerts,
            "usd_protected_per_n": dollars,
            "usd_per_false_alert": dollars / max(false_alerts, 1e-9),
            "prevalence": prevalence, "per": per,
        })
    return rows

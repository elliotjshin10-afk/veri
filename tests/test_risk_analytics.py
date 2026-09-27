"""Risk analytics: sequential evidence, uncertainty, cost and stability."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from veridis.model.risk_analytics import (  # noqa: E402
    apply_llr, bootstrap_early_detection, bootstrap_metric, cost_curve,
    expected_calibration_error, fit_llr, population_stability_index,
    sequential_scores,
)


def _fitted(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    s = np.clip(np.where(y == 1, rng.normal(0.75, 0.15, n),
                         rng.normal(0.12, 0.10, n)), 1e-4, 1 - 1e-4)
    return fit_llr(s, y), s, y


def test_llr_is_positive_for_victims_and_negative_for_ordinary():
    """The sign is the whole point: an ordinary transfer must argue *against*.

    The first implementation used logit(score) - logit(prior), which on a
    miscalibrated score made ordinary transfers argue weakly *for* fraud, so a
    long legitimate relationship drifted toward the alarm threshold.
    """
    model, s, y = _fitted()
    llr = apply_llr(s, model)
    assert np.median(llr[y == 1]) > 0
    assert np.median(llr[y == 0]) < 0


def test_llr_is_strictly_monotone_in_the_score():
    """At the first transfer the sequential arm holds exactly the same
    information as the point score, so it must rank identically."""
    model, _, _ = _fitted()
    probe = np.linspace(0.01, 0.99, 50)
    llr = apply_llr(probe, model)
    assert np.all(np.diff(llr) > 0)


def _seq_frame():
    rows = []
    for k, sc in enumerate([0.2, 0.6, 0.9, 0.95], start=1):
        rows.append({"sender": "v", "destination": "d", "transfer_k": k,
                     "event_time": 1000 * k, "score": sc, "label": 1,
                     "amount_usd": 100.0 * k})
    return pl.DataFrame(rows)


def test_sequential_statistic_only_uses_transfers_up_to_k():
    """Causality: the value at k must not move when later transfers change."""
    model, _, _ = _fitted()
    a = sequential_scores(_seq_frame(), model).sort("transfer_k")
    changed = _seq_frame().with_columns(
        pl.when(pl.col("transfer_k") == 4).then(0.001)
        .otherwise(pl.col("score")).alias("score"))
    b = sequential_scores(changed, model).sort("transfer_k")
    assert a["cum_llr"].to_list()[:3] == pytest.approx(b["cum_llr"].to_list()[:3])


def test_sequential_statistic_accumulates():
    model, _, _ = _fitted()
    out = sequential_scores(_seq_frame(), model).sort("transfer_k")
    cum = out["cum_llr"].to_list()
    assert cum[-1] > cum[0]


def test_decay_bounds_accumulation():
    """Decay must damp a long run, or one long relationship dominates."""
    model, _, _ = _fitted()
    full = sequential_scores(_seq_frame(), model, decay=1.0)["cum_llr"].to_list()[-1]
    damped = sequential_scores(_seq_frame(), model, decay=0.3)["cum_llr"].to_list()[-1]
    assert damped < full


def test_bootstrap_interval_contains_the_point_estimate():
    rows = []
    for i in range(60):
        for k in (1, 2, 3):
            rows.append({"label": 1, "sender": f"v{i}", "destination": "d",
                         "transfer_k": k, "raw_score": 0.9 if i % 2 else 0.1,
                         "amount_usd": 100.0})
    df = pl.DataFrame(rows)
    out = bootstrap_early_detection(df, threshold=0.5, n_boot=120)
    for k, v in out.items():
        assert v["lo"] <= v["rate"] <= v["hi"]
        assert v["n_relationships"] == 60


def test_bootstrap_metric_brackets_the_value():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    y = (rng.random(600) < 0.3).astype(int)
    s = np.where(y == 1, rng.normal(0.7, 0.2, 600), rng.normal(0.3, 0.2, 600))
    out = bootstrap_metric(y, s, roc_auc_score, n_boot=120)
    assert out["lo"] <= out["value"] <= out["hi"]


def test_cost_curve_trades_dollars_against_interruptions():
    """A looser budget must protect more and interrupt more. Both, always."""
    rows = []
    for i in range(40):
        for k in (1, 2):
            rows.append({"label": 1, "sender": f"v{i}", "destination": "d",
                         "transfer_k": k, "raw_score": 0.4 + 0.01 * i,
                         "amount_usd": 500.0})
    for i in range(400):
        rows.append({"label": 0, "sender": f"c{i}", "destination": "e",
                     "transfer_k": 0, "raw_score": 0.001 * i, "amount_usd": 100.0})
    df = pl.DataFrame(rows)
    curve = cost_curve(df, [0.01, 0.05, 0.20])
    assert curve[0]["interruptions"] <= curve[-1]["interruptions"]
    assert curve[0]["protected_usd"] <= curve[-1]["protected_usd"]


def test_psi_is_zero_for_an_unchanged_distribution():
    rng = np.random.default_rng(3)
    a = pl.DataFrame({"f": rng.normal(0, 1, 3000)})
    out = population_stability_index(a, a, ["f"])
    assert out and out[0]["psi"] < 1e-6
    assert out[0]["verdict"] == "stable"


def test_psi_detects_a_shifted_distribution():
    rng = np.random.default_rng(4)
    a = pl.DataFrame({"f": rng.normal(0, 1, 3000)})
    b = pl.DataFrame({"f": rng.normal(3, 1, 3000)})
    out = population_stability_index(a, b, ["f"])
    assert out[0]["psi"] > 0.25
    assert out[0]["verdict"] == "material shift"


def test_calibration_error_is_small_when_probabilities_are_honest():
    rng = np.random.default_rng(5)
    p = rng.random(6000)
    y = (rng.random(6000) < p).astype(int)
    assert expected_calibration_error(y, p)["ece"] < 0.05


def test_calibration_error_catches_overconfidence():
    rng = np.random.default_rng(6)
    p = np.clip(rng.random(4000) * 0.5 + 0.5, 0, 1)
    y = np.zeros(4000, dtype=int)          # confident and wrong
    assert expected_calibration_error(y, p)["ece"] > 0.4


# ---------------------------------------------------- deployment prevalence
def test_precision_collapses_as_prevalence_falls():
    """The number a customer experiences, not the one we evaluate at.

    TPR and FPR are conditional on the true class and so are invariant to
    prevalence; precision is not. Evaluating at a 15% base rate and quoting
    that precision to a wallet running at 0.1% overstates it by an order of
    magnitude.
    """
    from veridis.model.deployment import precision_at_prevalence

    tpr, fpr = 0.437, 0.0098
    at_eval = precision_at_prevalence(tpr, fpr, 0.149)
    at_real = precision_at_prevalence(tpr, fpr, 0.001)
    assert at_eval > 0.85
    assert at_real < 0.06
    # strictly decreasing as prevalence falls
    seq = [precision_at_prevalence(tpr, fpr, p)
           for p in (0.1, 0.01, 0.001, 0.0001)]
    assert all(a > b for a, b in zip(seq, seq[1:]))


def test_prior_shift_preserves_ranking_and_moves_probabilities():
    import numpy as np
    from veridis.model.deployment import shift_prior

    p = np.array([0.01, 0.2, 0.5, 0.8, 0.99])
    out = shift_prior(p, eval_prior=0.15, deploy_prior=0.001)
    assert np.all(np.diff(out) > 0), "ranking must be unchanged"
    assert np.all(out < p), "a rarer prior must lower every probability"


def test_prior_shift_is_identity_when_priors_match():
    import numpy as np
    from veridis.model.deployment import shift_prior

    p = np.array([0.05, 0.4, 0.9])
    assert np.allclose(shift_prior(p, 0.2, 0.2), p)


def test_deployment_table_alert_counts_are_consistent():
    from veridis.model.deployment import deployment_table

    rows = deployment_table(tpr=0.5, fpr=0.01, prevalences=(0.001,), per=100_000)
    r = rows[0]
    assert r["true_alerts_per_n"] == pytest.approx(0.5 * 0.001 * 100_000)
    assert r["false_alerts_per_n"] == pytest.approx(0.01 * 0.999 * 100_000)
    assert r["alerts_per_n"] == pytest.approx(
        r["true_alerts_per_n"] + r["false_alerts_per_n"])

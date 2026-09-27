"""M4 - train, evaluate with temporal+grouped splitting, report early detection."""
import json, logging, sys
sys.path.insert(0, "src")
import polars as pl
from veridis.model.train import (
    temporal_split, train_model, calibrate, calibration_split, evaluate_all, save,
)
from veridis.config import PROCESSED, TARGET_FPR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")

events = pl.read_parquet(PROCESSED / "events_features.parquet")
split = temporal_split(events)
import datetime as dt
print(f"temporal cutoff: {dt.datetime.utcfromtimestamp(split.cutoff/1000).date()}")
print(f"train={split.train.height:,} (pos={split.train['label'].sum():,})  "
      f"test={split.test.height:,} (pos={split.test['label'].sum():,})")

fit, cal = calibration_split(split.train)
print(f"calibration holdout: fit={fit.height:,} cal={cal.height:,}")
booster = train_model(fit)
iso = calibrate(booster, cal)
report, scored = evaluate_all(booster, iso, split, TARGET_FPR)

# Bands from the FPR budget, not from round score values.
import numpy as np
neg = scored.filter(pl.col("label") == 0)["raw_score"].to_numpy()
thresholds = {
    "caution": float(np.quantile(neg, 1 - 0.05)),
    "high_risk": float(np.quantile(neg, 1 - TARGET_FPR)),
}
(PROCESSED / "thresholds_latest.json").write_text(json.dumps(thresholds, indent=2))
report["thresholds"] = thresholds
save(booster, iso, report, "latest")
scored.write_parquet(PROCESSED / "scored_test.parquet")

for w in report.get("warnings", []):
    print(f"\n!!  UNRELIABLE METRICS: {w}\n")

m = report["metrics"]
print(f"\n{'='*64}\nPR-AUC {m['pr_auc']:.4f}   ROC-AUC {m['roc_auc']:.4f}   "
      f"base rate {m['base_rate']:.4f}")
print(f"at FPR {m['fpr']:.4%} (target {TARGET_FPR:.1%}): "
      f"event recall {m['recall_event']:.3f}, precision {m['precision_event']:.3f}")
print(f"\nEARLY DETECTION (flagged by transfer k, FPR<={TARGET_FPR:.0%}):")
for r in report["early_detection"]:
    print(f"  k={r['k']}: {r['rate']:.1%}  ({r['flagged']}/{r['victims']} victims)")
d = report["dollars"]
print(f"\nDOLLARS: {d['share']:.1%} of ${d['total_usd']:,.0f} victim losses "
      f"occurred at/after our first flag")
print(f"victims ever flagged: {d['victims_ever_flagged']}/{d['victims_total']}")
# Differentiation experiment: what survives without the destination?
from veridis.model.ablation import run_ablation, unseen_destination_test
abl = run_ablation(split.train, scored)
report["ablation"] = abl
report["unseen_destination"] = unseen_destination_test(split.train, scored)
print("\nABLATION - what each view of the transfer can detect:")
print(f"  {'arm':<24}{'feats':>6}{'PR-AUC':>9}{'lift':>7}{'k=1':>8}{'k=3':>8}")
for a in abl:
    print(f"  {a['arm']:<24}{a['n_features']:>6}{a['pr_auc']:>9.3f}"
          f"{a['lift']:>6.1f}x{a['early_k1']:>7.1%}{a['early_k3']:>8.1%}")
u = report["unseen_destination"]
if "pr_auc" in u:
    print(f"\nUNSEEN DESTINATIONS (dest had <=1 prior sender; n={u['n']}, pos={u['n_positive']}):")
    print(f"  sender-side-only PR-AUC {u['pr_auc']:.3f} vs base {u['base_rate']:.3f} "
          f"({u['lift']:.1f}x)")
else:
    print(f"\nUNSEEN DESTINATIONS: {u.get('note')}")

f = report.get("first_send_only")
if f:
    fm = f["metrics"]
    print(f"\nFIRST-SEND ONLY (the case an interstitial actually fires on; "
          f"{f['n_positive']} victims vs {f['n_negative']} controls):")
    print(f"  PR-AUC {fm['pr_auc']:.4f} (base {fm['base_rate']:.4f}, "
          f"{fm['pr_auc']/max(fm['base_rate'],1e-9):.1f}x)  "
          f"ROC-AUC {fm['roc_auc']:.4f}  FPR {fm['fpr']:.2%}")
    print(f"  recall {fm['recall_event']:.3f}  precision {fm['precision_event']:.3f}")

h = report.get("hard_negatives")
if h:
    hm = h["metrics"]
    print(f"\nHARD NEGATIVES (controls whose destination already had >=5 senders, "
          f"n={h['n_negatives']:,}):")
    print(f"  PR-AUC {hm['pr_auc']:.4f} (base {hm['base_rate']:.4f}), "
          f"FPR {hm['fpr']:.2%}")
    for r in h["early_detection"]:
        print(f"  k={r['k']}: {r['rate']:.1%}")
else:
    print("\nHARD NEGATIVES: too few qualifying controls to evaluate")

print("\nFEATURE IMPORTANCE BY FAMILY:")
for r in report["family_importance"]:
    print(f"  {r['family']:<14} {r['share']:.1%}")
print("\nTOP FEATURES:")
for r in report["top_features"][:10]:
    print(f"  {r['feature']:<32} {r['gain']:,.0f}")
print(f"{'='*64}")

# ---- risk analytics: sequential evidence, uncertainty, cost, stability ----
from veridis.model.risk_analytics import (
    bootstrap_early_detection, bootstrap_metric, compare_sequential, cost_curve,
    expected_calibration_error, population_stability_index,
)
from veridis.features.asof import FEATURE_COLUMNS
from sklearn.metrics import roc_auc_score
import numpy as np

# Score the train split too, so the likelihood ratio is fitted out of sample.
_raw_tr = booster.predict(split.train.select(FEATURE_COLUMNS).to_numpy())
_prob_tr = (iso.predict_proba(_raw_tr.reshape(-1, 1))[:, 1]
            if hasattr(iso, "predict_proba") else _raw_tr)
train_scored = split.train.with_columns(
    pl.Series("raw_score", _raw_tr), pl.Series("score", _prob_tr))
# Sequential test across a decay sweep. Reported whichever way it lands.
sweep = []
for _d in (1.0, 0.9, 0.7, 0.5, 0.3, 0.15):
    sweep.append(compare_sequential(scored, train_scored,
                                    target_fpr=TARGET_FPR, decay=_d))
seq = sweep[0]
report["sequential"] = {"sweep": sweep, "point": seq["early_point"]}
print("\nSEQUENTIAL EVIDENCE (accumulate log-likelihood along each relationship)")
print(f"  both arms held to the same budget: point FPR {seq['point_fpr']:.2%}")
print(f"  median LLR per transfer: victim {seq['median_llr_victim']:+.2f}, "
      f"control {seq['median_llr_control']:+.2f}")
print(f"  {'decay':>7}{'k=1':>9}{'k=2':>9}{'k=3':>9}{'k=5':>9}")
for r in sweep:
    print(f"  {r['decay']:>7}{r['early_sequential']['1']:>9.1%}"
          f"{r['early_sequential']['2']:>9.1%}{r['early_sequential']['3']:>9.1%}"
          f"{r['early_sequential']['5']:>9.1%}")
print(f"  {'point':>7}{seq['early_point']['1']:>9.1%}{seq['early_point']['2']:>9.1%}"
      f"{seq['early_point']['3']:>9.1%}{seq['early_point']['5']:>9.1%}")
_best = max(r["early_sequential"]["1"] for r in sweep)
report["sequential"]["verdict"] = (
    "no improvement: the statistic converges to the point score as decay -> 0 "
    "and never exceeds it. The as-of-time features already encode the "
    "relationship history, so accumulating the output double-counts it and "
    "adds noise from long control relationships."
    if _best <= seq["early_point"]["1"] else "sequential improves early detection")
print(f"  -> {report['sequential']['verdict']}")

boot = bootstrap_early_detection(scored, report["metrics"]["threshold"])
report["early_detection_ci"] = boot
print("\nEARLY DETECTION, bootstrapped over relationships (95% CI):")
for k, v in boot.items():
    print(f"  k={k}: {v['rate']:.1%}  [{v['lo']:.1%}, {v['hi']:.1%}]  n={v['n_relationships']}")

y_t = scored["label"].to_numpy(); s_t = scored["raw_score"].to_numpy()
report["roc_auc_ci"] = bootstrap_metric(y_t, s_t, roc_auc_score)
ci = report["roc_auc_ci"]
if ci:
    print(f"  ROC-AUC {ci['value']:.3f}  [{ci['lo']:.3f}, {ci['hi']:.3f}]")

curve = cost_curve(scored, [0.001, 0.005, 0.01, 0.02, 0.05, 0.10])
report["cost_curve"] = curve
print("\nCOST CURVE (dollars protected vs customers interrupted):")
print(f"  {'budget':>8}{'FPR':>8}{'interrupts':>12}{'protected':>14}{'$/interrupt':>14}")
for c in curve:
    upi = f"${c['usd_per_interruption']:,.0f}" if c["usd_per_interruption"] else "-"
    print(f"  {c['budget']:>8.2%}{c['fpr']:>8.2%}{c['interruptions']:>12,}"
          f"{c['protected_usd']:>14,.0f}{upi:>14}")

# ---- what this costs and protects at a realistic prevalence ----
from veridis.model.deployment import DEPLOY_PREVALENCE, deployment_economics, deployment_table
_budgets = [0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01]
_m = report["metrics"]
report["prevalence_sensitivity"] = deployment_table(_m["recall_event"], _m["fpr"])
print("\nPRECISION AT DEPLOYMENT PREVALENCE (same model, same TPR/FPR):")
print(f"  {'prevalence':>12}{'precision':>11}{'false per true':>16}")
for r in report["prevalence_sensitivity"]:
    fpt = f"{r['false_per_true']:.1f}" if r["false_per_true"] else "-"
    print(f"  {r['prevalence']:>12.4%}{r['precision']:>11.3f}{fpt:>16}")
print(f"  (evaluation prevalence here is {_m['base_rate']:.1%}, "
      f"precision {_m['precision_event']:.3f})")

econ = deployment_economics(scored, _budgets, prevalence=0.001)
report["deployment_economics"] = econ
print("\nDEPLOYMENT ECONOMICS at 0.1% prevalence, per 100,000 real sends:")
print(f"  {'budget':>8}{'recall':>8}{'false alerts':>14}{'$ protected':>14}{'$/false alert':>15}")
for r in econ:
    print(f"  {r['budget']:>8.2%}{r['recall']:>8.3f}{r['false_alerts_per_n']:>14.0f}"
          f"{r['usd_protected_per_n']:>14,.0f}{r['usd_per_false_alert']:>15,.0f}")
print(f"  average value of one caught victim relationship: "
      f"${econ[0]['usd_per_relationship']:,.0f}")

psi = population_stability_index(split.train, split.test, FEATURE_COLUMNS)
report["psi"] = psi[:12]
from collections import defaultdict as _dd
from veridis.features.asof import FEATURE_FAMILY as _FF
_byfam = _dd(list)
for _r in psi:
    _byfam[_FF[_r["feature"]]].append(_r["psi"])
report["psi_by_family"] = {
    f: {"median": float(np.median(v)), "n": len(v),
        "material": int(sum(1 for x in v if x >= 0.25))}
    for f, v in _byfam.items()}
print("\nDRIFT BY FEATURE FAMILY (median PSI; higher = less stable out of period):")
for f, v in sorted(report["psi_by_family"].items(), key=lambda x: -x[1]["median"]):
    print(f"  {f:<14} {v['median']:.3f}  ({v['material']}/{v['n']} material shifts)")
shifted = [r for r in psi if r["psi"] >= 0.25]
print(f"\nFEATURE DRIFT train->test: {len(shifted)}/{len(psi)} features with material shift")
for r in psi[:5]:
    print(f"  {r['feature']:<34} PSI {r['psi']:.3f}  {r['verdict']}")

ece = expected_calibration_error(y_t, scored["score"].to_numpy())
report["ece"] = ece
if ece:
    print(f"\nCALIBRATION: ECE {ece['ece']:.4f}, worst bin gap {ece['max_gap']:.3f}")

# Re-save: the ablation and slice analyses are computed after the first save,
# and the UI reads the report from disk.
save(booster, iso, report, "latest")
print(f"\nreport written with: {sorted(k for k in report if report[k] is not None)}")

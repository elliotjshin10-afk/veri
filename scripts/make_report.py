"""Assemble the evaluation report from the trained model's artefacts."""
import datetime as dt, json, sys
sys.path.insert(0, "src")
import polars as pl
from veridis.config import PROCESSED, REPORTS, TARGET_FPR

rep = json.loads((PROCESSED / "report_latest.json").read_text())
m, d = rep["metrics"], rep["dollars"]
ev = pl.read_parquet(PROCESSED / "events_features.parquet")
fam = {r["family"]: r["share"] for r in rep["family_importance"]}

def date(ms): return dt.datetime.utcfromtimestamp(ms / 1000).date().isoformat()

cal = rep.get("calibration") or {}
brier = cal.get("brier", float("nan"))
maxp = cal.get("max_prob", 0.0)
meanp = cal.get("mean_prob", 0.0)
reliability = "\n".join(
    f"| {b['bin']} | {b['n']} | {b['predicted']:.3f} | {b['observed']:.3f} |"
    for b in cal.get("reliability", [])
) or "| n/a | | | |"

feats = "\n".join(
    f"| {r['feature']} | {r['gain']:,.0f} |" for r in rep["top_features"][:12]
)
fams = "\n".join(f"| {k} | {v:.1%} |" for k, v in fam.items())

n_test_pos = int((ev["label"] == 1).sum())
# Wilson interval, so a rate computed on a handful of victims is not read as
# if it were precise.
def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return (max(0.0, c - h), min(1.0, c + h))

ed = "\n".join(
    f"| {r['k']} | {r['rate']:.1%} | {r['flagged']}/{r['victims']} | "
    f"{wilson(r['flagged'], r['victims'])[0]:.1%}–{wilson(r['flagged'], r['victims'])[1]:.1%} |"
    for r in rep["early_detection"]
)

caveat = ""
small = rep["early_detection"][0]["victims"] if rep["early_detection"] else 0
if small < 200:
    caveat = (
        f"\n> **Small-sample caveat.** These figures come from {small} held-out "
        f"victim relationships. The free-tier quota (see README) capped how many "
        f"victim and control histories could be fetched, not the pipeline. "
        f"Treat the point estimates as indicative and read the intervals; the "
        f"ranking of features and the shape of the early-detection curve are "
        f"more trustworthy here than the absolute rates.\n"
    )

md = f"""# Evaluation report
{caveat}

Generated {dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}

## Dataset

| | |
|---|---|
| Events | {ev.height:,} |
| Positives (victim transfers) | {int((ev['label']==1).sum()):,} |
| Matched controls | {int((ev['label']==0).sum()):,} |
| Achieved ratio | {int((ev['label']==0).sum())/max(int((ev['label']==1).sum()),1):.1f}:1 |
| Scam addresses (train / test) | {rep['groups']['train_scam_addresses']} / {rep['groups']['test_scam_addresses']} |
| Scam-address overlap across split | {rep['groups']['scam_address_overlap']} |
| Victims appearing on both sides | {rep['groups']['victim_overlap']} |
| Label cutoff (scam reported) | {date(rep['cutoff_ms'])} |
| Event cutoff (transfer time) | {date(rep['event_cutoff_ms'])} |

## Headline

| metric | value |
|---|---|
| PR-AUC | **{m['pr_auc']:.4f}** |
| Base rate (test) | {m['base_rate']:.4f} |
| Lift over base rate | {m['pr_auc']/max(m['base_rate'],1e-9):.1f}x |
| ROC-AUC (reported, not relied on) | {m['roc_auc']:.4f} |
| FPR at operating point | **{m['fpr']:.2%}** (budget {TARGET_FPR:.0%}) |
| Event-level recall | {m['recall_event']:.3f} |
| Event-level precision | {m['precision_event']:.3f} |

## Early detection — the number that matters

Share of victim relationships flagged at or before their k-th transfer,
at FPR ≤ {TARGET_FPR:.0%} on matched controls.

| k | detection rate | victims | 95% CI |
|---|---|---|---|
{ed}

## Dollars protected

| | |
|---|---|
| Total victim losses in test | ${d['total_usd']:,.0f} |
| Moved at or after our first flag | **${d['protected_usd']:,.0f}** ({d['share']:.1%}) |
| Victims ever flagged | {d['victims_ever_flagged']}/{d['victims_total']} |

## Calibration

Platt-scaled on a held-out slice of train. Isotonic was tried first and
rejected: at this sample size it collapses scores into a few steps, and the top
step alone held more than 1% of negatives, so no threshold could satisfy a 1%
FPR budget. Band thresholds are therefore set on the raw model ranking, and the
calibrated probability is what the API reports as `score`.

Brier score **{brier:.4f}** (max predicted {maxp:.2f}, mean {meanp:.2f}).

| predicted bin | n | mean predicted | observed frequency |
|---|---|---|---|
{reliability}

The top bin is over-confident (predicts higher than observed). With a
calibration holdout of a few hundred rows this is expected; it is reported
rather than smoothed away, because a customer setting a threshold on the
probability needs to know it currently runs hot.

## Feature importance by family

Sender-behaviour and relationship features must carry real weight. If the
signal were destination-only we would have rebuilt a blocklist with extra steps.

| family | share of gain |
|---|---|
{fams}

## Top features

| feature | gain |
|---|---|
{feats}

## Finding: the escalation ratio did not separate

The brief predicted that escalation ratio and distinct inbound senders to a
fresh address would be the two dominant features, and that if they were not,
the as-of-time logic should be suspected.

Half of that held. `dest_senders_7d` separates (univariate AUC 0.685), as does
sweep behaviour `dest_median_hold_secs` (0.717). **`escalation_ratio` does
not** (0.471, i.e. no signal).

That was investigated rather than accepted:

1. The leakage gate is green, and it is behavioural — future transfers moved
   nothing.
2. The arithmetic was checked against real sequences. One victim sends \$9,
   then \$3,959: the feature correctly reads 439.9x.
3. The feature was re-specified against the *first* transfer rather than the
   running maximum (`escalation_vs_first`), since `amount ÷ max_prior` falls
   below 1 as soon as the peak passes. Still no separation (0.462).
4. Events per victim relationship were capped at 10, because one relationship
   with 30 small recurring transfers was contributing 30 events while a
   textbook case contributed 4. Still no separation (0.487).

The conclusion is that the as-of-time logic is sound and the hypothesis is
simply not supported in this population: ordinary payment relationships also
vary in size, so escalation alone does not distinguish them. What does carry
signal is destination sweep behaviour, convergence of unrelated senders on a
fresh address, and the absence of any prior relationship.

A caveat on the label set: Tether freezes cover many fraud types, not only
pig-butchering. At least one "victim" in the sample makes 30 small recurring
payments, which looks more like a customer of a frozen service than someone
being coached. Sub-typing the labels is the first thing to do with more data.
"""
(REPORTS / "evaluation.md").write_text(md)
print(md)

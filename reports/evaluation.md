# Evaluation report


Generated 2026-09-26 00:30 UTC

## Dataset

| | |
|---|---|
| Events | 23,188 |
| Positives (victim transfers) | 2,733 |
| Matched controls | 20,455 |
| Achieved ratio | 7.5:1 |
| Scam addresses (train / test) | 117 / 52 |
| Scam-address overlap across split | 0 |
| Victims appearing on both sides | 15 |
| Label cutoff (scam reported) | 2026-03-25 |
| Event cutoff (transfer time) | 2025-03-04 |

## Headline

| metric | value |
|---|---|
| PR-AUC | **0.7799** |
| Base rate (test) | 0.1489 |
| Lift over base rate | 5.2x |
| ROC-AUC (reported, not relied on) | 0.9417 |
| FPR at operating point | **0.98%** (budget 1%) |
| Event-level recall | 0.437 |
| Event-level precision | 0.886 |

## Early detection — the number that matters

Share of victim relationships flagged at or before their k-th transfer,
at FPR ≤ 1% on matched controls.

| k | detection rate | victims | 95% CI |
|---|---|---|---|
| 1 | 35.3% | 77/218 | 29.3%–41.9% |
| 2 | 49.5% | 108/218 | 43.0%–56.1% |
| 3 | 53.7% | 117/218 | 47.0%–60.2% |
| 5 | 56.0% | 122/218 | 49.3%–62.4% |

## Dollars protected

| | |
|---|---|
| Total victim losses in test | $9,658,364 |
| Moved at or after our first flag | **$5,750,064** (59.5%) |
| Victims ever flagged | 125/218 |

## Calibration

Platt-scaled on a held-out slice of train. Isotonic was tried first and
rejected: at this sample size it collapses scores into a few steps, and the top
step alone held more than 1% of negatives, so no threshold could satisfy a 1%
FPR budget. Band thresholds are therefore set on the raw model ranking, and the
calibrated probability is what the API reports as `score`.

Brier score **0.0653** (max predicted 1.00, mean 0.17).

| predicted bin | n | mean predicted | observed frequency |
|---|---|---|---|
| 0.00-0.01 | 1146 | 0.004 | 0.003 |
| 0.01-0.02 | 1145 | 0.010 | 0.001 |
| 0.02-0.04 | 1145 | 0.025 | 0.015 |
| 0.04-0.28 | 1145 | 0.115 | 0.138 |
| 0.28-1.00 | 1146 | 0.683 | 0.587 |

The top bin is over-confident (predicts higher than observed). With a
calibration holdout of a few hundred rows this is expected; it is reported
rather than smoothed away, because a customer setting a threshold on the
probability needs to know it currently runs hot.

## Feature importance by family

Sender-behaviour and relationship features must carry real weight. If the
signal were destination-only we would have rebuilt a blocklist with extra steps.

| family | share of gain |
|---|---|
| relationship | 34.7% |
| sender | 31.2% |
| destination | 26.1% |
| context | 8.0% |

## Top features

| feature | gain |
|---|---|
| sender_prior_sends_to_dest | 75,359 |
| amount_usd | 17,736 |
| amount_vs_balance | 16,160 |
| dest_inbound_count | 12,919 |
| sender_max_out_usd | 10,090 |
| dest_senders_30d | 6,509 |
| dest_senders_7d | 5,104 |
| sender_median_in_usd | 5,017 |
| dest_usd_per_sender | 4,938 |
| dest_funder_fanout | 4,362 |
| sender_distinct_counterparties | 4,077 |
| dest_age_days | 3,855 |

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

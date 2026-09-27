"""Assemble the demo's data: real API responses and a real operating curve.

The demo shows the product doing its job - a wallet calling /score before a
transfer settles - so every scenario on the page is a real held-out event put
through the real model and the real reason engine, and the request/response
shown is the one the API actually returns. Nothing is illustrative.
"""
from __future__ import annotations

import datetime as dt
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
import lightgbm as lgb
import numpy as np
import polars as pl

from veridis.config import DEMO, PROCESSED, TARGET_FPR
from veridis.features.asof import FEATURE_COLUMNS
from veridis.model.evaluate import early_detection, threshold_at_fpr
from veridis.model.reasons import ReasonEngine
from veridis.model.scoring import band as band_of, predict as predict_scores

booster = lgb.Booster(model_file=str(PROCESSED / "model_latest.txt"))
calib = pickle.loads((PROCESSED / "calibrator_latest.pkl").read_bytes())
thresholds = json.loads((PROCESSED / "thresholds_latest.json").read_text())
scored = pl.read_parquet(PROCESSED / "scored_test.parquet")
reasons = ReasonEngine(booster, FEATURE_COLUMNS)

hi, mid = thresholds["high_risk"], thresholds["caution"]


def api_response(row: dict) -> dict:
    """Exactly what POST /score returns for this event."""
    X = np.array([[row[c] for c in FEATURE_COLUMNS]], dtype=float)
    t0 = time.perf_counter()
    raw, prob = predict_scores(booster, calib, X)
    texts = reasons.explain(X, [row], top_n=5)[0]
    latency = (time.perf_counter() - t0) * 1000
    return {
        "score": round(float(prob[0]), 4),
        "band": band_of(float(raw[0]), thresholds),
        "reasons": texts,
        "model_version": "latest",
        "computed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "latency_ms": round(latency, 1),
    }


def scenario(row: dict, title: str, blurb: str, truth: str) -> dict:
    return {
        "title": title,
        "blurb": blurb,
        "ground_truth": truth,
        "request": {
            "chain": row["chain"],
            "sender": row["sender"],
            "destination": row["destination"],
            "amount_usd": round(float(row["amount_usd"]), 2),
            "asset": row.get("asset") or "USDT",
        },
        "response": api_response(row),
        "context": {
            "dest_age_days": round(float(row["dest_age_days"]), 1),
            "dest_senders_7d": int(row["dest_senders_7d"]),
            "prior_sends": int(row["sender_prior_sends_to_dest"]),
            "transfer_k": int(row["transfer_k"]),
            "sender_median_out_usd": round(float(row["sender_median_out_usd"]), 2),
        },
    }


pos = scored.filter(pl.col("label") == 1)
neg = scored.filter(pl.col("label") == 0)

scenarios = []

# 1. A victim mid-sequence, comfortably flagged.
cand = pos.filter((pl.col("raw_score") >= hi) & (pl.col("amount_usd") >= 2000)).sort(
    "amount_usd", descending=True)
if cand.height:
    scenarios.append(scenario(
        cand.row(0, named=True),
        "A victim being talked into a transfer",
        "A real wallet, mid-way through a real pig-butchering sequence. "
        "The destination was on no public list on this date.",
        "scam"))

# 2. A victim's FIRST transfer to the scam address - the hard case.
first = pos.filter((pl.col("transfer_k") == 1) & (pl.col("raw_score") >= mid)).sort(
    "raw_score", descending=True)
if first.height:
    scenarios.append(scenario(
        first.row(0, named=True),
        "The same pattern, on the very first transfer",
        "No prior relationship to read, and the scam has not started escalating. "
        "This is the hardest and most valuable case to catch.",
        "scam"))

# 3. A large legitimate transfer that must NOT be blocked.
big = neg.filter((pl.col("raw_score") < mid) & (pl.col("amount_usd") >= 50_000)).sort(
    "amount_usd", descending=True)
if big.height:
    scenarios.append(scenario(
        big.row(0, named=True),
        "A large legitimate payment",
        "Six figures moving between established counterparties. A model that "
        "learned 'big transfer = fraud' would stop this; a wallet would drop us.",
        "legitimate"))

# 4. A legitimate FIRST-time send - the classic false positive.
newpay = neg.filter(
    (pl.col("is_first_send_to_dest") == 1) & (pl.col("raw_score") < mid)
    & (pl.col("amount_usd") >= 500)).sort("amount_usd", descending=True)
if newpay.height:
    scenarios.append(scenario(
        newpay.row(0, named=True),
        "Paying a brand-new address, legitimately",
        "A first-ever transfer to an address this wallet has never used. "
        "Novelty alone must not be enough to interrupt someone.",
        "legitimate"))

# ---- operating curve: what each FPR budget buys, on held-out data ----
y = scored["label"].to_numpy()
s = scored["raw_score"].to_numpy()
curve = []
for fpr_budget in [0.001, 0.0025, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]:
    thr = threshold_at_fpr(y, s, fpr_budget)
    pred = s >= thr
    tp = int(((y == 1) & pred).sum()); fp = int(((y == 0) & pred).sum())
    fn = int(((y == 1) & ~pred).sum()); tn = int(((y == 0) & ~pred).sum())
    ed = {r["k"]: r["rate"] for r in early_detection(
        scored, thr, score_col="raw_score").to_dicts()}
    curve.append({
        "budget": fpr_budget,
        "threshold": float(thr),
        "fpr": fp / max(fp + tn, 1),
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "early_k1": ed.get(1, 0.0),
        "early_k3": ed.get(3, 0.0),
        "interrupted_per_1000": 1000 * fp / max(fp + tn, 1),
    })

out = {"scenarios": scenarios, "operating_curve": curve,
       "thresholds": thresholds, "target_fpr": TARGET_FPR}
(DEMO / "demo_data.json").write_text(json.dumps(out, indent=2))
print(f"wrote {DEMO/'demo_data.json'}")
for sc in scenarios:
    r = sc["response"]
    print(f"  {sc['ground_truth']:<11} {r['band']:<10} score={r['score']:.3f} "
          f"${sc['request']['amount_usd']:>10,.0f}  {sc['title']}")
print(f"  operating curve: {len(curve)} points")

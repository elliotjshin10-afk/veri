"""M7 - score Ethereum with the Tron-trained model. Nothing is refitted.

A note on truncation, because it runs the opposite way here. The Tron client
pages oldest-first, so a capped history is missing an address's *recent*
activity and the coverage guard drops events after the last transfer we hold.
Blockscout pages newest-first, so a capped history is missing an address's
*earliest* activity - which corrupts age and lifetime aggregates at every point
in time, not just late ones. There is no as-of cutoff that repairs that, so the
Ethereum test keeps only addresses whose entire history fits inside the page
budget. That restricts it to smaller wallets, which is the retail population the
product is for anyway, and it is stated rather than papered over.
"""
import json, pickle, sys
sys.path.insert(0, "src")
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from veridis.dataset.ingest import dedupe_transfers
from veridis.features.asof import FEATURE_COLUMNS, FeatureEngine
from veridis.model.evaluate import core_metrics, early_detection, threshold_at_fpr
from veridis.model.scoring import predict as predict_scores
from veridis.config import INTERIM, PROCESSED, TARGET_FPR


def load(name):
    p = INTERIM / name
    return pl.read_parquet(p) if p.exists() else None


scam_tx, vic_tx, ctl_tx = (load(f"eth_{k}_transfers.parquet")
                           for k in ("scam", "victim", "control"))
if scam_tx is None or vic_tx is None:
    raise SystemExit("run scripts/m7_crosschain.py first")
warehouse = dedupe_transfers([x for x in (scam_tx, vic_tx, ctl_tx) if x is not None])
print(f"ethereum warehouse: {warehouse.height:,} transfers")

trunc = load("eth_truncated.parquet")
complete = set(trunc.filter(~pl.col("truncated"))["address"].to_list()) if trunc is not None else set()
print(f"addresses with complete history: {len(complete):,} of "
      f"{trunc.height if trunc is not None else 0:,}")

labels = pl.read_parquet(INTERIM / "scam_addresses.parquet").filter(
    (pl.col("chain") == "ethereum") & pl.col("accepted"))
scam_addrs = set(a.lower() for a in labels["address"].to_list())
victims = pl.read_parquet(INTERIM / "eth_victims.parquet")

# --- positives: each victim transfer into a labelled address ---
pos = (warehouse.join(
        victims.select("from_address", "to_address"),
        on=["from_address", "to_address"], how="inner")
    .rename({"from_address": "sender", "to_address": "destination"})
    .with_columns(pl.col("block_time").alias("event_time"))
    .sort(["sender", "destination", "event_time"])
    .with_columns(
        (pl.int_range(pl.len()).over(["sender", "destination"]) + 1).alias("transfer_k"),
        pl.lit(1).cast(pl.Int8).alias("label")))

# --- negatives: control transfers where neither end is scam- or victim-linked ---
vaddrs = set(victims["from_address"].to_list())
if ctl_tx is not None:
    seeds = set(pl.read_parquet(INTERIM / "eth_control_seeds.parquet")["address"].to_list())
    peers = set(pl.read_parquet(INTERIM / "eth_control_peers.parquet")["address"].to_list())
    fetched = seeds | peers
    neg = (ctl_tx.filter(
            pl.col("from_address").is_in(list(fetched))
            & pl.col("to_address").is_in(list(fetched))
            & ~pl.col("from_address").is_in(list(scam_addrs | vaddrs))
            & ~pl.col("to_address").is_in(list(scam_addrs | vaddrs)))
        .rename({"from_address": "sender", "to_address": "destination"})
        .with_columns(pl.col("block_time").alias("event_time"),
                      pl.lit(0).cast(pl.Int64).alias("transfer_k"),
                      pl.lit(0).cast(pl.Int8).alias("label")))
else:
    neg = pos.head(0)

cols = ["chain", "sender", "destination", "amount_usd", "asset", "event_time",
        "tx_hash", "transfer_k", "label"]
events = pl.concat([pos.select(cols), neg.select(cols)], how="vertical_relaxed")

# Keep only events whose sender and destination we hold in full.
events = events.filter(
    pl.col("sender").is_in(list(complete)) & pl.col("destination").is_in(list(complete)))
print(f"events with complete history both ends: {events.height:,} "
      f"(pos={int((events['label']==1).sum()):,})")
if int((events["label"] == 1).sum()) < 20 or int((events["label"] == 0).sum()) < 50:
    raise SystemExit("too few complete-history events on Ethereum to evaluate")

events = events.with_row_index("event_id")
engine = FeatureEngine(warehouse)
feats = engine.compute(events)
engine.close()

# --- score with the Tron model, unchanged ---
booster = lgb.Booster(model_file=str(PROCESSED / "model_latest.txt"))
calib = pickle.loads((PROCESSED / "calibrator_latest.pkl").read_bytes())
X = feats.select(FEATURE_COLUMNS).to_numpy()
raw, prob = predict_scores(booster, calib, X)
feats = feats.with_columns(pl.Series("raw_score", raw), pl.Series("score", prob))

y = feats["label"].to_numpy()
thr_eth = threshold_at_fpr(y, raw, TARGET_FPR)
tron_thr = json.loads((PROCESSED / "thresholds_latest.json").read_text())["high_risk"]

m = core_metrics(y, raw, thr_eth)
ed = early_detection(feats, thr_eth, score_col="raw_score").to_dicts()
tron_report = json.loads((PROCESSED / "report_latest.json").read_text())
tm = tron_report["metrics"]

print(f"\n{'='*66}")
print("CROSS-CHAIN: Tron-trained model scoring Ethereum. No refitting.")
print(f"{'='*66}")
print(f"  events {feats.height:,}  positives {int(y.sum()):,}  "
      f"base rate {y.mean():.4f}")
print(f"  ROC-AUC  {m['roc_auc']:.4f}   (Tron test: {tm['roc_auc']:.4f})")
print(f"  PR-AUC   {m['pr_auc']:.4f}   lift {m['pr_auc']/max(y.mean(),1e-9):.2f}x"
      f"   (Tron: {tm['pr_auc']:.4f}, {tm['pr_auc']/tm['base_rate']:.2f}x)")
print(f"  at FPR {m['fpr']:.2%}: recall {m['recall_event']:.3f}, "
      f"precision {m['precision_event']:.3f}")
print("\n  early detection on Ethereum:")
for r in ed:
    print(f"    k={r['k']}: {r['rate']:.1%}  ({r['flagged']}/{r['victims']})")

# Transfer the Tron operating point unchanged - the harder test.
fpr_at_tron_thr = float((raw[y == 0] >= tron_thr).mean())
rec_at_tron_thr = float((raw[y == 1] >= tron_thr).mean())
print(f"\n  applying Tron's own threshold ({tron_thr:.3f}) to Ethereum:")
print(f"    FPR {fpr_at_tron_thr:.2%}, recall {rec_at_tron_thr:.3f}")

out = {
    "n_events": int(feats.height), "n_positive": int(y.sum()),
    "base_rate": float(y.mean()),
    "roc_auc": m["roc_auc"], "pr_auc": m["pr_auc"],
    "lift": m["pr_auc"] / max(float(y.mean()), 1e-9),
    "fpr": m["fpr"], "recall": m["recall_event"], "precision": m["precision_event"],
    "early_detection": ed,
    "tron_threshold_transferred": {
        "threshold": tron_thr, "fpr": fpr_at_tron_thr, "recall": rec_at_tron_thr},
    "tron_reference": {"roc_auc": tm["roc_auc"], "pr_auc": tm["pr_auc"],
                       "base_rate": tm["base_rate"]},
}
(PROCESSED / "crosschain_report.json").write_text(json.dumps(out, indent=2))
feats.write_parquet(PROCESSED / "eth_scored.parquet")
print(f"\nwrote crosschain_report.json")

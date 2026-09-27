"""M6 - replay harness.

Steps one real victim's real on-chain history transfer by transfer and shows,
side by side, what a public blocklist knew at each moment versus what the model
scored. The blocklist column is not a strawman: it is driven by the actual date
the destination first appeared on any public list we collected, so "clean" means
genuinely clean on that date.
"""
import datetime as dt, json, logging, pickle, sys
sys.path.insert(0, "src")
import lightgbm as lgb
import numpy as np
import polars as pl
from veridis.features.asof import FEATURE_COLUMNS, FeatureEngine
from veridis.model.reasons import ReasonEngine
from veridis.model.scoring import band as band_of, predict as predict_scores
from veridis.config import DEMO, INTERIM, PROCESSED

logging.basicConfig(level=logging.INFO, format="%(message)s")

warehouse = pl.read_parquet(PROCESSED / "warehouse_transfers.parquet")
events = pl.read_parquet(PROCESSED / "events_features.parquet")
labels = pl.read_parquet(INTERIM / "scam_addresses.parquet")
thresholds = json.loads((PROCESSED / "thresholds_latest.json").read_text())
booster = lgb.Booster(model_file=str(PROCESSED / "model_latest.txt"))
iso = pickle.loads((PROCESSED / "calibrator_latest.pkl").read_bytes())
reasons_engine = ReasonEngine(booster, FEATURE_COLUMNS)

def score_of(df):
    raw, prob = predict_scores(booster, iso, df.select(FEATURE_COLUMNS).to_numpy())
    return raw, prob

# Demo only from held-out data. Replaying a victim the model trained on would
# show the model recognising something it had already been shown, which is not
# what a partner is being asked to believe.
scored_path = PROCESSED / "scored_test.parquet"
if scored_path.exists():
    pos = pl.read_parquet(scored_path).filter(pl.col("label") == 1)
    print(f"replaying from held-out test set ({pos.height} positive events)")
else:
    raise SystemExit("run `make train` first: the demo replays test-set victims only")

# Pick a victim relationship that tells the story honestly: a real sequence of
# several transfers where the model crosses the threshold early and material
# money moved afterwards.
hi = thresholds["high_risk"]
caution = thresholds["caution"]

# Prefer a relationship where the score *rises across* the threshold rather
# than one already flagged at transfer 1.
#
# Two reasons. It is the honest picture - a model that fires on the opening
# transfer of every victim would also fire on a great many legitimate first
# sends, and showing a case that starts below the line makes that visible. And
# it is the more useful picture: the claim is early detection, which only means
# something if there is a "before" to see.
by_pair = (
    pos.sort(["sender", "destination", "transfer_k"])
    .group_by(["sender", "destination"], maintain_order=True)
    .agg(
        pl.len().alias("n"),
        pl.col("amount_usd").sum().alias("total"),
        pl.col("raw_score").first().alias("score_k1"),
        pl.col("raw_score").max().alias("best"),
        pl.col("amount_usd").max().alias("max_amt"),
        (pl.col("raw_score") >= hi).arg_max().alias("first_flag_idx"),
        (pl.col("raw_score") >= hi).any().alias("ever"),
    )
    .filter(pl.col("ever"))
)

rising = by_pair.filter(
    (pl.col("n") >= 4)
    & (pl.col("score_k1") < hi)          # not already flagged on transfer 1
    & (pl.col("first_flag_idx") <= 2)    # crosses by transfer 3 (0-indexed)
    & (pl.col("total") >= 2000)
).sort("total", descending=True)

if rising.height:
    cand = rising
    print(f"selected from {rising.height} relationships that cross the threshold "
          f"after transfer 1")
else:
    cand = by_pair.filter((pl.col("n") >= 4) & (pl.col("total") >= 2000)).sort(
        ["first_flag_idx", "total"], descending=[False, True])
    print("no rising case available; falling back to earliest flag")

if cand.height == 0:
    cand = by_pair.filter(pl.col("n") >= 3).sort("total", descending=True)
    if cand.height == 0:
        raise SystemExit("no victim relationship with enough transfers for a replay")

pick = cand.row(0, named=True)
sender, dest = pick["sender"], pick["destination"]
seq = pos.filter((pl.col("sender") == sender) & (pl.col("destination") == dest)).sort("transfer_k")
X = seq.select(FEATURE_COLUMNS).to_numpy()
texts = reasons_engine.explain(X, seq.to_dicts(), top_n=4)

lab = labels.filter(pl.col("address") == dest).row(0, named=True)
listed_at = lab["first_reported_at"]
fmt = lambda ms: dt.datetime.utcfromtimestamp(ms / 1000).strftime("%Y-%m-%d %H:%M")

steps, first_flag_k, protected = [], None, 0.0
for i, r in enumerate(seq.to_dicts()):
    flagged = r["raw_score"] >= hi
    if flagged and first_flag_k is None:
        first_flag_k = r["transfer_k"]
    if first_flag_k is not None and r["transfer_k"] >= first_flag_k:
        protected += r["amount_usd"]
    band = band_of(r["raw_score"], thresholds)
    steps.append({
        "k": r["transfer_k"], "when": fmt(r["event_time"]),
        "amount": r["amount_usd"], "score": round(float(r["score"]), 3),
        "band": band, "reasons": texts[i],
        "tx": r["tx_hash"],
        # The blocklist only knows once the address is actually listed.
        "blocklist": "listed" if (listed_at and r["event_time"] >= listed_at) else "clean",
        "dest_age_days": round(r["dest_age_days"], 1),
        "dest_senders_7d": int(r["dest_senders_7d"]),
    })

payload = {
    "sender": sender, "destination": dest,
    "listed_at": fmt(listed_at) if listed_at else None,
    "listed_sources": lab["sources"],
    "total_usd": float(seq["amount_usd"].sum()),
    "protected_usd": float(protected),
    "first_flag_k": first_flag_k,
    "thresholds": thresholds,
    "steps": steps,
    "generated_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
}
(DEMO / "replay.json").write_text(json.dumps(payload, indent=2))
print(json.dumps({k: v for k, v in payload.items() if k != "steps"}, indent=2))
print(f"\n{len(steps)} transfers; first flag at k={first_flag_k}; "
      f"${protected:,.0f} of ${payload['total_usd']:,.0f} after first flag")

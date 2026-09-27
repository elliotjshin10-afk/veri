"""Latency benchmark for /score.

A wallet will not block a send on a slow call; the brief's target is p95 under
300ms. This measures the in-process scoring path (feature computation + model +
SHAP reasons) so the number is not confounded by loopback HTTP overhead.
"""
import statistics, sys, time
sys.path.insert(0, "src")
import polars as pl
from veridis.api.main import load_state
from veridis.features.asof import FEATURE_COLUMNS
from veridis.model.scoring import band, predict as predict_scores

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
state = load_state()
events = pl.read_parquet("data/processed/events_features.parquet")
sample = events.sample(n=min(N, events.height), seed=1).to_dicts()

lat = []
for r in sample:
    t0 = time.perf_counter()
    ev = pl.DataFrame([{
        "event_id": 0, "chain": r["chain"], "sender": r["sender"],
        "destination": r["destination"], "amount_usd": float(r["amount_usd"]),
        "event_time": int(r["event_time"]),
    }], schema={"event_id": pl.Int64, "chain": pl.Utf8, "sender": pl.Utf8,
                "destination": pl.Utf8, "amount_usd": pl.Float64,
                "event_time": pl.Int64})
    feats = state["warehouse"].engine.compute(ev)
    X = feats.select(FEATURE_COLUMNS).to_numpy()
    raw, _prob = predict_scores(state["booster"], state["iso"], X)
    band(float(raw[0]), state["thresholds"])
    state["reasons"].explain(X, [feats.row(0, named=True)], top_n=5)
    lat.append((time.perf_counter() - t0) * 1000)

lat.sort()
n_tx = state["warehouse"].n_transfers
print(f"n={len(lat)} over {n_tx:,} warehouse transfers")
print(f"  p50 {statistics.median(lat):7.1f} ms")
print(f"  p95 {lat[int(len(lat)*0.95)]:7.1f} ms   (target < 300 ms)")
print(f"  p99 {lat[int(len(lat)*0.99)]:7.1f} ms")
print(f"  max {lat[-1]:7.1f} ms")

# Persist so the demo page reports the measured number instead of a literal
# that silently goes stale as the warehouse grows.
import json
from veridis.config import REPORTS
(REPORTS / "latency.json").write_text(json.dumps({
    "n": len(lat), "warehouse_transfers": n_tx,
    "p50_ms": statistics.median(lat), "p95_ms": lat[int(len(lat)*0.95)],
    "p99_ms": lat[int(len(lat)*0.99)], "max_ms": lat[-1],
}, indent=2))

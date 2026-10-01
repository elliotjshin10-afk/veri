"""Prove the browser scorer matches the Python feature layer.

The site computes features in JavaScript and scores with an exported copy of
the model. If that JS drifts from `features/asof.py` by even a little, the
model is being fed something it was never trained on and every score the public
sees is quietly wrong - train/serve skew, the failure this project has guarded
against everywhere else.

So: take real addresses, compute features both ways from identical transfer
data, and compare. Any disagreement fails.
"""
import json, subprocess, sys, tempfile
sys.path.insert(0, "src")
import numpy as np
import polars as pl

from veridis.features.asof import FeatureEngine
from veridis.dataset.holdings import all_indexed, all_transfers
from veridis.config import INTERIM, PROCESSED

N = int(sys.argv[1]) if len(sys.argv) > 1 else 60
model = json.loads((__import__("pathlib").Path("site/model_dest.json")).read_text())
FEATS = model["features"]

# holdings is the one list of what we hold. This script kept its own - the M3
# warehouse plus three truncated files - so it compared the browser against a
# feature layer fed different data from the one the index and the model were
# built on, and failed on whichever addresses happened to differ.
warehouse = all_transfers()
indexed = all_indexed()

last = (pl.concat([
    warehouse.select(pl.col("to_address").alias("address"), "block_time"),
    warehouse.select(pl.col("from_address").alias("address"), "block_time"),
], how="vertical_relaxed")
    .filter(pl.col("address").is_in(list(indexed)))
    .group_by("address").agg(pl.col("block_time").max().alias("last_seen")))
sample = last.sort("address").sample(n=min(N, last.height), seed=5)

# --- Python side: the real feature layer ---
probe = sample.select(
    pl.col("address").alias("destination"),
    (pl.col("last_seen") + 1000).alias("event_time"),
).with_columns(pl.lit("tron").alias("chain"), pl.lit("__probe__").alias("sender"),
               pl.lit(1000.0).alias("amount_usd")).with_row_index("event_id")
engine = FeatureEngine(warehouse)
pyf = engine.compute(probe)
engine.close()

import lightgbm as lgb
booster = lgb.Booster(model_file=str(PROCESSED / "model_browser.txt"))
py_scores = booster.predict(pyf.select(FEATS).to_numpy())
pyf = pyf.with_columns(pl.Series("py_score", py_scores))

# --- JS side: the exact transfers the browser would have fetched ---
cases = []
for row in pyf.iter_rows(named=True):
    a = row["destination"]
    tx = warehouse.filter(
        (pl.col("from_address") == a) | (pl.col("to_address") == a))
    cases.append({
        "address": a, "asOf": int(row["event_time"]),
        "transfers": [{"from": r["from_address"], "to": r["to_address"],
                       "usd": float(r["amount_usd"]), "t": int(r["block_time"])}
                      for r in tx.iter_rows(named=True)],
        "py": {f: (None if row[f] is None else float(row[f])) for f in FEATS},
        "py_score": float(row["py_score"]),
    })

with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
    json.dump({"cases": cases, "model": model}, fh)
    payload = fh.name

scorer_url = (__import__("pathlib").Path("site/scorer.js").resolve().as_uri())
runner = f"""
import {{computeFeatures, score}} from '{scorer_url}';
import {{readFileSync}} from 'fs';
const {{cases, model}} = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const out = cases.map(c => {{
  const f = computeFeatures(c.transfers, c.address, c.asOf);
  return {{address: c.address, js: f, js_score: f ? score(model, f) : null}};
}});
process.stdout.write(JSON.stringify(out));
"""
with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as fh:
    fh.write(runner)
    runner_path = fh.name

res = subprocess.run(["node", runner_path, payload], capture_output=True, text=True)
if res.returncode != 0:
    print(res.stderr[-2000:])
    raise SystemExit("node runner failed")
js = {r["address"]: r for r in json.loads(res.stdout)}

worst_feat, worst_score, checked, mismatches = 0.0, 0.0, 0, []
for c in cases:
    j = js.get(c["address"])
    if not j or not j["js"]:
        mismatches.append((c["address"], "no js features"))
        continue
    checked += 1
    for f in FEATS:
        a, b = c["py"][f], j["js"][f]
        if a is None or b is None:
            continue
        denom = max(abs(a), abs(b), 1.0)
        rel = abs(a - b) / denom
        worst_feat = max(worst_feat, rel)
        if rel > 1e-6:
            mismatches.append((c["address"], f"{f}: py={a!r} js={b!r}"))
    ds = abs(c["py_score"] - j["js_score"])
    worst_score = max(worst_score, ds)
    if ds > 1e-6:
        mismatches.append((c["address"], f"score py={c['py_score']:.8f} js={j['js_score']:.8f}"))

print(f"compared {checked} addresses across {len(FEATS)} features")
print(f"  worst relative feature difference: {worst_feat:.2e}")
print(f"  worst absolute score difference:   {worst_score:.2e}")
if mismatches:
    print(f"\nPARITY FAILED - {len(mismatches)} mismatches:")
    for a, m in mismatches[:12]:
        print(f"  {a[:14]}  {m}")
    raise SystemExit(1)
print("\nPARITY OK - the browser scores identically to the Python feature layer.")

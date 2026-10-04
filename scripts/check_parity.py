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
from veridis.model.quantise import quantise_matrix
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
# Quantised exactly as the browser does, or the two disagree on a band
# whenever a feature lands on a tree split. See veridis.model.quantise.
py_scores = booster.predict(quantise_matrix(pyf.select(FEATS).to_numpy()))
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

# Both sides must agree on the band exactly, and on the score to machine
# precision.
#
# This gate was loosened once, to 0.05, on the view that bit-identity was not
# available: a feature like dest_forward_ratio is a ratio of two sums over
# thousands of transfers, DuckDB and JavaScript accumulate them in different
# orders, and the results differ by about 7e-14 relative. Tree splits are exact
# comparisons, so a value landing on one took a different branch in each
# language and the score moved by up to 3e-02.
#
# That was the wrong conclusion. The features cannot be made bit-identical, but
# the DECISION can: quantising to twelve significant figures before scoring - far
# more precision than these features carry - makes both sides see the same double
# and branch the same way. See veridis.model.quantise.
#
# The loose gate was not harmless while it stood. It passed an address the ledger
# scored 0.649936 and the browser 0.644155, straddling the 0.644543 elevated
# threshold: the same model giving a public prediction and the live page two
# different verdicts. Quantised, the worst score difference across 400 addresses
# is 2.2e-16.
SCORE_TOL = 1e-9
worst_feat, worst_score, checked, mismatches = 0.0, 0.0, 0, []
# Which feature, not just how much. A score that diverges while every feature
# agrees to 1e-14 means a value landed exactly on a tree split and took the
# other branch - worth naming the feature rather than reporting a number nobody
# can act on.
worst_feat_name, worst_feat_pair = None, None
score_detail = []
bands_differ = []
THR = model["thresholds"]


def band(p):
    return ("high" if p >= THR["high"]
            else "elevated" if p >= THR["elevated"] else "ordinary")
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
        if rel > worst_feat:
            worst_feat, worst_feat_name, worst_feat_pair = rel, f, (a, b)
        if rel > 1e-6:
            mismatches.append((c["address"], f"{f}: py={a!r} js={b!r}"))
    ds = abs(c["py_score"] - j["js_score"])
    worst_score = max(worst_score, ds)
    # The band is what a person is told. Scores are sums of many floats, and two
    # languages will not agree on the last bit of a large sum - when such a value
    # sits exactly on a tree split the branch flips and the score moves. That is
    # tolerable; a different VERDICT is not. So the band is checked with no
    # tolerance at all, and the score divergence is reported as a diagnostic.
    if band(c["py_score"]) != band(j["js_score"]):
        bands_differ.append((c["address"], c["py_score"], j["js_score"],
                             band(c["py_score"]), band(j["js_score"])))
    if ds > SCORE_TOL:
        worst_in_row = max(
            ((abs(c["py"][f] - j["js"][f]) / max(abs(c["py"][f]), abs(j["js"][f]), 1.0), f)
             for f in FEATS if c["py"][f] is not None and j["js"][f] is not None),
            default=(0.0, "-"))
        score_detail.append((c["address"], c["py_score"], j["js_score"], worst_in_row))
        mismatches.append((c["address"], f"score py={c['py_score']:.8f} js={j['js_score']:.8f}"))

print(f"compared {checked} addresses across {len(FEATS)} features")
print(f"  band disagreements (zero tolerated): {len(bands_differ)}")
print(f"  worst relative feature difference: {worst_feat:.2e}")
print(f"  worst absolute score difference:   {worst_score:.2e}")
if worst_feat_name:
    print(f"  worst feature: {worst_feat_name} "
          f"(py={worst_feat_pair[0]!r} js={worst_feat_pair[1]!r})")
for addr, pys, jss, (rel, f) in score_detail:
    print(f"  {addr}: py={pys:.9f} js={jss:.9f}; "
          f"largest feature gap in that row {rel:.2e} on {f}")
for a, pys, jss, pb, jb in bands_differ:
    mismatches.append((a, f"BAND {pb} vs {jb} (py={pys:.9f} js={jss:.9f})"))

if mismatches:
    print(f"\nPARITY FAILED - {len(mismatches)} mismatches:")
    for a, m in mismatches[:12]:
        print(f"  {a[:14]}  {m}")
    raise SystemExit(1)
print("\nPARITY OK - the browser and the Python feature layer agree on every band, "
      "and on every feature to within float reproducibility.")

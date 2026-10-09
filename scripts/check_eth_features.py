"""Prove the browser computes the ETHEREUM features, not just scores them.

check_eth_parity.py takes a stored feature vector and checks the browser walks
the trees the same way LightGBM does. That is half the question. It says
nothing about whether the browser ARRIVES at that vector from a list of
transfers, which is where the harder bugs live: an off-by-one on a cutoff, a
zero-value row kept on one side, a payout wallet fetched to a different depth.
Tron has had that check since the start. Ethereum did not, and two bugs found
on 2026-10-08 were both of exactly this kind.

So: take real Ethereum addresses, hand the same transfers to both
implementations, and compare all nineteen features and the band.

The payout wallet's rows are supplied the same way, because the harness is
comparing two implementations and not two fetches. The browser still picks
WHICH wallet by itself: if the two ever disagreed about that, the fan-in would
differ and this is where it would show.

Run: python scripts/check_eth_features.py [n]
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys
import tempfile

sys.path.insert(0, "src")

import numpy as np
import polars as pl

from veridis.config import INTERIM, PROCESSED, ROOT, SITE
from veridis.dataset.holdings import eth_complete, eth_transfers
from veridis.features import payout
from veridis.features.asof import FeatureEngine
from veridis.model import browser_model

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
SCORE_TOL = 1e-9

model = json.loads((SITE / "model_eth.json").read_text())
FEATS = model["features"]
THR = model["thresholds"]


def band(p: float) -> str:
    return ("high" if p >= THR["high"]
            else "elevated" if p >= THR["elevated"] else "ordinary")


def main() -> None:
    tf = eth_transfers()
    complete = eth_complete()
    # Complete histories only. A truncated one is missing an address's earliest
    # activity on this chain, so the two sides would be compared on a feature
    # that is wrong in both.
    last = (pl.concat([tf.select(pl.col("to_address").alias("address"), "block_time"),
                       tf.select(pl.col("from_address").alias("address"), "block_time")],
                      how="vertical_relaxed")
            .filter(pl.col("address").is_in(list(complete)))
            .group_by("address").agg(pl.col("block_time").max().alias("last_seen")))
    sample = last.sort("address").sample(n=min(N, last.height), seed=5)

    probe = (sample.select(pl.col("address").alias("destination"),
                           (pl.col("last_seen") + 1000).alias("event_time"))
             .with_columns(pl.lit("ethereum").alias("chain"),
                           pl.lit("__probe__").alias("sender"),
                           pl.lit(1000.0).alias("amount_usd"))
             .with_row_index("event_id"))

    engine = FeatureEngine(tf)
    pyf = engine.compute(probe)
    engine.close()

    # The second hop, assembled exactly as m7_pit_model.py assembles it.
    ptx = pl.read_parquet(INTERIM / "eth_payout_transfers.parquet")
    tr = pl.read_parquet(INTERIM / "eth_payout_truncated.parquet")
    trunc = dict(zip(tr["address"].to_list(), tr["truncated"].to_list()))
    for a in tf["to_address"].unique().to_list():
        trunc.setdefault(a, False)
    hop = (pl.concat([ptx.select(tf.columns), tf], how="vertical_relaxed")
           .unique(subset=["tx_hash", "to_address"]))
    pyf = pyf.join(payout.batch(probe, tf, hop, trunc), on="event_id", how="left") \
        .with_columns(pl.col("payout_fanin_pit").fill_null(0),
                      pl.col("payout_fanin_exact").fill_null(0))

    # Scored by walking the shipped JSON, not the LightGBM booster beside it.
    # model_eth_pit.txt is the full arm and carries a different feature set, so
    # it would answer a different question; and this harness is about whether
    # the two languages reach the same FEATURES, which check_eth_parity.py
    # already complements by holding the browser's walk to LightGBM's.
    py_scores = browser_model.score(model, pyf.select(FEATS).to_numpy().tolist())
    pyf = pyf.with_columns(pl.Series("py_score", py_scores))

    def rows_for(df):
        return [{"from": r["from_address"], "to": r["to_address"],
                 "usd": float(r["amount_usd"]), "t": int(r["block_time"])}
                for r in df.iter_rows(named=True)]

    cases = []
    for row in pyf.iter_rows(named=True):
        a = row["destination"]
        own = tf.filter((pl.col("from_address") == a) | (pl.col("to_address") == a))
        pay = payout.top_payout(tf, a, int(row["event_time"]))
        ptx_rows = rows_for(hop.filter(pl.col("to_address") == pay)) if pay else []
        cases.append({
            "address": a, "asOf": int(row["event_time"]),
            "transfers": rows_for(own),
            "payoutTransfers": ptx_rows,
            "payoutTruncated": bool(trunc.get(pay, False)) if pay else False,
            "py": {f: (None if row[f] is None else float(row[f])) for f in FEATS},
            "py_score": float(row["py_score"]),
        })

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump({"cases": cases, "model": model}, fh)
        payload = fh.name

    scorer = (ROOT / "site" / "scorer.js").resolve().as_uri()
    runner = f"""
import {{computeFeatures, score, topPayout, payoutFanin}} from '{scorer}';
import {{readFileSync}} from 'fs';
const {{cases, model}} = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const out = cases.map(c => {{
  const f = computeFeatures(c.transfers, c.address, c.asOf);
  if (f) {{
    const p = topPayout(c.transfers, c.address, c.asOf);
    Object.assign(f, payoutFanin(c.payoutTransfers, p, c.asOf, c.address,
                                 c.payoutTruncated));
  }}
  return {{address: c.address, js: f, js_score: f ? score(model, f) : null}};
}});
process.stdout.write(JSON.stringify(out));
"""
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as fh:
        fh.write(runner)
        runner_path = fh.name

    res = subprocess.run(["node", runner_path, payload], capture_output=True,
                         text=True)
    if res.returncode != 0:
        print(res.stderr[-2000:])
        raise SystemExit("node runner failed")
    js = {r["address"]: r for r in json.loads(res.stdout)}

    worst_feat, worst_score, bad, checked = 0.0, 0.0, [], 0
    worst_name = None
    for c in cases:
        got = js.get(c["address"])
        if got is None or got["js"] is None:
            bad.append((c["address"], "no JS features"))
            continue
        checked += 1
        for f in FEATS:
            p, j = c["py"][f], got["js"].get(f)
            if p is None and j is None:
                continue
            if p is None or j is None:
                bad.append((c["address"], f"{f}: py={p} js={j}"))
                continue
            d = abs(p - j) / max(abs(p), abs(j), 1e-9)
            if d > worst_feat:
                worst_feat, worst_name = d, f"{f} (py={p} js={j})"
        ds = abs(c["py_score"] - got["js_score"])
        worst_score = max(worst_score, ds)
        if band(c["py_score"]) != band(got["js_score"]):
            bad.append((c["address"], f"band {band(c['py_score'])} vs "
                                      f"{band(got['js_score'])}"))

    print(f"checked {checked} Ethereum addresses on {len(FEATS)} features")
    print(f"  band disagreements (zero tolerated): "
          f"{sum(1 for _, m in bad if m.startswith('band'))}")
    print(f"  worst relative feature difference: {worst_feat:.2e}")
    print(f"  worst absolute score difference:   {worst_score:.2e}")
    if worst_name:
        print(f"  worst feature: {worst_name}")
    for addr, msg in bad[:6]:
        print(f"    {addr}  {msg}")
    if bad or worst_score > SCORE_TOL:
        raise SystemExit("ETH FEATURE PARITY FAILED")
    print("\nETH FEATURE PARITY OK - the browser and the Python feature layer "
          "agree on every band, and on every feature to within float "
          "reproducibility.")


if __name__ == "__main__":
    main()

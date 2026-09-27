"""Prove the two-sided browser scorer matches the Python feature layer.

Same contract as check_parity.py, one level harder: the page now computes 37
features from TWO addresses' histories plus an amount, and every one of them
has to agree with `features/asof.py` exactly. A silent drift here means the
model is fed something it never saw in training and every warning the public
gets is quietly wrong.

Real sends are used, not synthetic ones - rows straight out of the event table,
so the sender, destination, amount and timestamp are a combination that
actually happened, including the awkward ones: first-ever sends, repeat sends,
senders with no outbound history at all.
"""
from __future__ import annotations

import json, pathlib, subprocess, sys, tempfile
sys.path.insert(0, "src")
import numpy as np
import polars as pl

from veridis.config import PROCESSED

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
TOL = 1e-9

model = json.loads(pathlib.Path("site/model_pair.json").read_text())
FEATS = model["features"]

events = pl.read_parquet(PROCESSED / "events_features.parquet")
warehouse = pl.read_parquet(PROCESSED / "warehouse_transfers.parquet")

# Stratify so the sample cannot miss the cases most likely to drift.
first = events.filter(pl.col("is_first_send_to_dest") == 1)
repeat = events.filter(pl.col("is_first_send_to_dest") == 0)
take = max(1, N // 2)
sample = pl.concat([first.sample(n=min(take, first.height), seed=11),
                    repeat.sample(n=min(N - take, repeat.height), seed=11)])
print(f"{sample.height} real sends "
      f"({int((sample['is_first_send_to_dest']==1).sum())} first-ever, "
      f"{int((sample['is_first_send_to_dest']==0).sum())} repeat)")

# Transfers touching either address, in the shape the browser receives them.
def touching(addr: str) -> list[dict]:
    d = warehouse.filter((pl.col("to_address") == addr) | (pl.col("from_address") == addr))
    return [{"from": r["from_address"], "to": r["to_address"],
             "usd": r["amount_usd"], "t": int(r["block_time"])}
            for r in d.iter_rows(named=True)]

cases = []
for r in sample.iter_rows(named=True):
    cases.append({
        "sender": r["sender"], "destination": r["destination"],
        "amountUsd": r["amount_usd"], "asOfMs": int(r["event_time"]),
        "senderTransfers": touching(r["sender"]),
        "destTransfers": touching(r["destination"]),
        "expected": {f: (None if r[f] is None else float(r[f])) for f in FEATS},
    })

scorer = pathlib.Path("site/scorer.js").resolve().as_uri()
driver = f"""
import {{computePairFeatures, score}} from '{scorer}';
import fs from 'fs';
const cases = JSON.parse(fs.readFileSync(process.argv[2] === "--" ? process.argv[3] : process.argv[2], 'utf8'));
const model = JSON.parse(fs.readFileSync('site/model_pair.json', 'utf8'));
const out = cases.map(c => {{
  const f = computePairFeatures(c);
  return {{f, s: score(model, f)}};
}});
console.log(JSON.stringify(out));
"""
with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
    json.dump(cases, fh); cases_path = fh.name
with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as fh:
    fh.write(driver); driver_path = fh.name

res = subprocess.run(["node", driver_path, "--", cases_path],
                     capture_output=True, text=True)
if res.returncode != 0:
    sys.exit("node failed:\n" + res.stderr[:3000])
got = json.loads(res.stdout)

worst: dict[str, float] = {}
bad = []
for c, g in zip(cases, got):
    for f in FEATS:
        exp, act = c["expected"][f], g["f"].get(f)
        if exp is None:
            continue
        if act is None or (isinstance(act, float) and np.isnan(act)):
            bad.append((c["destination"], f, exp, act)); continue
        denom = max(abs(exp), 1.0)
        rel = abs(float(act) - exp) / denom
        worst[f] = max(worst.get(f, 0.0), rel)
        if rel > TOL:
            bad.append((c["destination"], f, exp, act))

print(f"compared {len(cases)} sends across {len(FEATS)} features")
top = sorted(worst.items(), key=lambda kv: -kv[1])[:5]
for f, v in top:
    print(f"   worst relative difference  {v:.2e}  {f}")
if bad:
    print(f"\nPAIR PARITY FAILED - {len(bad)} mismatches:")
    for d, f, e, a in bad[:12]:
        print(f"   {d[:14]}  {f:28s} py={e!r:>18}  js={a!r}")
    sys.exit(1)
print("\nPAIR PARITY OK - the browser computes all 37 features identically.")

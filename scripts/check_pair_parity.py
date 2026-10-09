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

# The second hop. The link is derived from the SAME frame the browser is handed,
# not from the full warehouse: if Python picked the payout wallet from a wider
# view than the browser can see, the two would disagree about which wallet and
# the harness would be comparing counts of different things.
from veridis.features import payout  # noqa: E402
from veridis.config import INTERIM   # noqa: E402

PAYOUT_ON = all(f in FEATS for f in payout.FEATURES)
payout_tx, payout_trunc = None, {}
if PAYOUT_ON:
    payout_tx = pl.read_parquet(INTERIM / "payout_transfers.parquet")
    _tr = pl.read_parquet(INTERIM / "payout_truncated.parquet")
    payout_trunc = dict(zip(_tr["address"].to_list(), _tr["truncated"].to_list()))
    events = events.join(
        payout.batch(events.select("event_id", "destination", "event_time"),
                     warehouse, payout_tx, payout_trunc),
        on="event_id", how="left").with_columns(
        pl.col("payout_fanin_pit").fill_null(0),
        pl.col("payout_fanin_exact").fill_null(0))

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

def _pay_for(addr: str, when: int):
    return payout.top_payout(warehouse, addr, when) if PAYOUT_ON else None


def payout_rows(addr: str, when: int):
    """None when we do not hold the wallet, which is not the same as holding it
    and finding nothing. The browser distinguishes the two by whether its fetch
    succeeded; here the distinction is whether the wallet is in the ingest."""
    pay = _pay_for(addr, when)
    if pay is None or pay not in payout_trunc:
        return None
    d = payout_tx.filter(pl.col("to_address") == pay)
    return [{"from": r["from_address"], "to": r["to_address"],
             "usd": r["amount_usd"], "t": int(r["block_time"])}
            for r in d.iter_rows(named=True)]


def payout_cut(addr: str, when: int) -> bool:
    pay = _pay_for(addr, when)
    return bool(payout_trunc.get(pay, False)) if pay else False


cases = []
for r in sample.iter_rows(named=True):
    cases.append({
        "sender": r["sender"], "destination": r["destination"],
        "amountUsd": r["amount_usd"], "asOfMs": int(r["event_time"]),
        "senderTransfers": touching(r["sender"]),
        "destTransfers": touching(r["destination"]),
        "expected": {f: (None if r[f] is None else float(r[f])) for f in FEATS},
        **({"payoutTransfers": payout_rows(r["destination"], int(r["event_time"])),
            "payoutTruncated": payout_cut(r["destination"], int(r["event_time"]))}
           if PAYOUT_ON else {}),
    })

scorer = pathlib.Path("site/scorer.js").resolve().as_uri()
driver = f"""
import {{computePairFeatures, score, topPayout, payoutFanin}} from '{scorer}';
import fs from 'fs';
const cases = JSON.parse(fs.readFileSync(process.argv[2] === "--" ? process.argv[3] : process.argv[2], 'utf8'));
const model = JSON.parse(fs.readFileSync('site/model_pair.json', 'utf8'));
const out = cases.map(c => {{
  const f = computePairFeatures(c);
  if (f && c.payoutTransfers !== undefined) {{
    // The browser picks WHICH wallet itself, from the destination's own rows.
    const p = topPayout(c.destTransfers, c.destination, c.asOfMs);
    Object.assign(f, payoutFanin(c.payoutTransfers, p, c.asOfMs, c.destination,
                                 c.payoutTruncated));
  }}
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
print(f"\nPAIR PARITY OK - the browser computes all {len(FEATS)} "
      f"features identically.")

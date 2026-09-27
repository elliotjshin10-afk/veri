"""Precompute /research output for every wallet whose history we actually hold.

The published demo is sandboxed and cannot reach TronGrid, so it cannot look an
address up live. What it can do honestly is ship the answers we already have:
one entry per *indexed* address - a wallet whose own transfer history was
fetched, not one glimpsed as somebody else's counterparty - carrying the same
facts and the same rule-based verdict the API would return.

Windowed counts are measured against each address's own last activity rather
than today, because most of this data is historical: "47 wallets paid it in its
final week" says something, "0 wallets paid it in the last 7 days" says only
that the data is old.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "src")
import duckdb
import polars as pl

from veridis.api.live import collection_verdict
from veridis.dataset.holdings import all_indexed, all_transfers
from veridis.model.address_risk import risk_band
from veridis.config import DEMO, INTERIM, PROCESSED, SITE

DAY_MS = 86_400_000

transfers = all_transfers()
indexed = all_indexed()
labels = pl.read_parquet(INTERIM / "scam_addresses.parquet").filter(pl.col("accepted"))
# Model score per address, with its measured address-level discrimination.
scores_path = PROCESSED / "address_scores.parquet"
report_path = PROCESSED / "address_model_report.json"
if not scores_path.exists():
    raise SystemExit("run scripts/build_address_model.py first")
score_map = dict(zip(*pl.read_parquet(scores_path).to_dict(as_series=False).values()))
model_report = json.loads(report_path.read_text())
THRESHOLDS = model_report["thresholds"]
label_map = dict(zip(labels["address"].to_list(), labels["sources"].to_list()))
label_when = dict(zip(labels["address"].to_list(), labels["first_reported_at"].to_list()))
print(f"indexed addresses: {len(indexed):,}  warehouse: {transfers.height:,} transfers")

con = duckdb.connect()
con.execute("PRAGMA threads=4")
con.register("tx", transfers.to_arrow())
con.register("idx", pl.DataFrame({"address": sorted(indexed)}).to_arrow())
con.execute("""
CREATE TEMP TABLE legs AS
SELECT from_address AS address,'out' AS side,to_address AS peer,amount_usd,block_time FROM tx
UNION ALL
SELECT to_address AS address,'in' AS side,from_address AS peer,amount_usd,block_time FROM tx;
CREATE TEMP TABLE sweep AS
SELECT address, block_time,
  (block_time - last_value(in_time IGNORE NULLS) OVER w)/1000.0 AS hold_secs
FROM (SELECT address, side, block_time,
        CASE WHEN side='in' THEN block_time END AS in_time FROM legs) s
WINDOW w AS (PARTITION BY address ORDER BY block_time
             ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
QUALIFY side='out';
""")

base = con.execute(f"""
WITH a AS (
  SELECT l.address,
    COUNT(*) FILTER (WHERE side='in')                      AS in_ct,
    COUNT(*) FILTER (WHERE side='out')                     AS out_ct,
    COUNT(DISTINCT peer) FILTER (WHERE side='in')          AS senders,
    COUNT(DISTINCT peer) FILTER (WHERE side='out')         AS payees,
    COALESCE(SUM(amount_usd) FILTER (WHERE side='in'),0)   AS in_usd,
    COALESCE(SUM(amount_usd) FILTER (WHERE side='out'),0)  AS out_usd,
    MIN(block_time) AS first_seen, MAX(block_time) AS last_seen
  FROM legs l JOIN idx USING (address) GROUP BY l.address
)
SELECT a.*,
  (SELECT COUNT(DISTINCT peer) FROM legs w
     WHERE w.address=a.address AND w.side='in'
       AND w.block_time >= a.last_seen - {7*DAY_MS})       AS senders_final_7d,
  (SELECT COUNT(DISTINCT peer) FROM legs w
     WHERE w.address=a.address AND w.side='in'
       AND w.block_time >= a.last_seen - {30*DAY_MS})      AS senders_final_30d,
  (SELECT MEDIAN(hold_secs) FROM sweep s WHERE s.address=a.address)  AS hold,
  (SELECT MAX(v)/NULLIF(SUM(v),0) FROM (
      SELECT peer, SUM(amount_usd) v FROM legs o
      WHERE o.address=a.address AND o.side='out' GROUP BY peer))     AS consolidation
FROM a
""").pl()
print(f"profiled: {base.height:,}")

entries = []
for r in base.iter_rows(named=True):
    span = max(1.0, (r["last_seen"] - r["first_seen"]) / DAY_MS)
    prof = {
        "age_days": round(span, 1),
        "senders_all": int(r["senders"]),
        "senders_7d": int(r["senders_final_7d"]),
        "senders_30d": int(r["senders_final_30d"]),
        "inbound_count": int(r["in_ct"]), "outbound_count": int(r["out_ct"]),
        "payees_all": int(r["payees"]),
        "inbound_usd": float(r["in_usd"]), "outbound_usd": float(r["out_usd"]),
        "usd_per_sender": (float(r["in_usd"]) / r["senders"]) if r["senders"] else None,
        "forward_ratio": (float(r["out_usd"]) / r["in_usd"]) if r["in_usd"] else None,
        "consolidation_ratio": float(r["consolidation"]) if r["consolidation"] is not None else None,
        "median_hold_secs": float(r["hold"]) if r["hold"] is not None else None,
    }
    addr = r["address"]
    _, notes = collection_verdict(prof)   # observations only; not a verdict
    score = score_map.get(addr)
    band = risk_band(score, THRESHOLDS) if score is not None else None
    entries.append({
        "a": addr,
        "s": round(float(score), 4) if score is not None else None,
        "b": band,
        "n": notes,
        "l": 1 if addr in label_map else 0,
        "ls": label_map.get(addr),
        "lw": label_when.get(addr),
        "p": {k: (round(v, 3) if isinstance(v, float) else v)
              for k, v in prof.items() if v is not None},
    })

entries.sort(key=lambda e: (-(e["s"] or 0), -e["l"]))
out = {"count": len(entries), "entries": entries,
       "model": {"evaluation": model_report["evaluation"],
                 "thresholds": THRESHOLDS}}
# Both consumers get it. This used to write only to demo/, and site/ was
# updated by hand - so the published page kept serving a 3,567-address index
# long after the ingest had fetched thousands more. Nothing downstream can
# notice that kind of staleness, so it must not be possible to forget.
blob = json.dumps(out, separators=(",", ":"))
for path in (DEMO / "address_index.json", SITE / "address_index.json"):
    path.write_text(blob)
    print(f"wrote {path}  ({path.stat().st_size/1024:.0f} KB, {len(entries):,} addresses)")
from collections import Counter
print(Counter(e["b"] for e in entries).most_common())
print(f"on a public list: {sum(e['l'] for e in entries):,}")

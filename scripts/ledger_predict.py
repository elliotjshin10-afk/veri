"""Nightly: flag addresses that look like collection points and are on no list.

Candidates come from the live USDT transfer stream, which is the population the
product actually sits in front of - wallets receiving stablecoins right now -
rather than anything we curated. Anything already on a list we hold is dropped
before scoring: predicting an address Tether has already frozen proves nothing,
and including one would quietly inflate the hit rate.

Each prediction records the score, the features behind it, and the model file's
own hash, so a later reader can check that the number came from the model we
said it did and not one fitted afterwards.
"""
from __future__ import annotations

import asyncio, datetime as dt, hashlib, json, logging, sys
sys.path.insert(0, "src")

import lightgbm as lgb
import polars as pl

from veridis.chain.address import hex_to_base58
from veridis.chain.tron import tron_client, PAGE
from veridis.config import INTERIM, PROCESSED, ROOT, TRONGRID_BASE, USDT_TRON
from veridis.dataset.ingest import fetch_histories
from veridis.features.asof import FeatureEngine
from veridis.ledger import Ledger, predictions_path
from veridis.model.address_risk import DEST_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)

N_CANDIDATES = int(sys.argv[1]) if len(sys.argv) > 1 else 400
MAX_PAGES = int(sys.argv[2]) if len(sys.argv) > 2 else 4
# The "high" band: the top 1% of ordinary wallets by the shipped model's own
# held-out control distribution. A ledger is a public claim, and one made at a
# looser budget would fill with wallets that were never going to be frozen and
# say nothing about the model.


async def sample_recent(client, n: int) -> list[str]:
    """Addresses receiving USDT in the last few hours."""
    url = f"{TRONGRID_BASE}/v1/contracts/{USDT_TRON}/events"
    now = int(dt.datetime.utcnow().timestamp() * 1000)
    seen: list[str] = []
    for hours_back in (1, 4, 10, 20, 36):
        lo = now - hours_back * 3_600_000
        payload = await client.get_json(url, {
            "event_name": "Transfer", "limit": PAGE,
            "min_block_timestamp": lo, "max_block_timestamp": lo + 120_000,
            "order_by": "block_timestamp,asc"})
        for e in (payload or {}).get("data", []):
            to = (e.get("result") or {}).get("to")
            if not to:
                continue
            try:
                seen.append(hex_to_base58(to))
            except Exception:
                continue
        if len(dict.fromkeys(seen)) >= n * 3:
            break
    return list(dict.fromkeys(seen))


def listed_addresses() -> set[str]:
    """Everything already on a list we hold - never predict one of these."""
    out: set[str] = set()
    p = INTERIM / "scam_addresses.parquet"
    if p.exists():
        out |= set(pl.read_parquet(p)["address"].to_list())
    return out


async def main() -> None:
    model_path = PROCESSED / "model_dest_latest.txt"
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()[:16]
    booster = lgb.Booster(model_file=str(model_path))
    # The banding threshold from the shipped model's own held-out control
    # distribution, not from the backtest file - those are different runs and
    # a threshold borrowed across them means nothing.
    report = json.loads((PROCESSED / "address_model_report.json").read_text())
    threshold = report["thresholds"]["high"]

    listed = listed_addresses()
    ledger = Ledger(predictions_path(ROOT))
    already = {e["address"] for e in ledger}
    ok, msg = ledger.verify()
    if not ok:
        sys.exit(f"ledger is broken, refusing to append: {msg}")
    print(f"ledger: {msg}")

    async with tron_client(concurrency=4) as client:
        cands = await sample_recent(client, N_CANDIDATES)
        fresh = [a for a in cands if a not in listed and a not in already][:N_CANDIDATES]
        print(f"sampled {len(cands):,} recent recipients -> {len(fresh):,} unlisted and new")
        if not fresh:
            return
        tx, trunc = await fetch_histories(client, fresh, max_pages=MAX_PAGES,
                                          concurrency=4, label="candidates")

    if not tx.height:
        print("no transfer history fetched; nothing to score")
        return

    now_ms = int(dt.datetime.utcnow().timestamp() * 1000)
    ev = (pl.DataFrame({"destination": fresh})
          .with_columns(pl.lit(now_ms).alias("event_time"), pl.lit("tron").alias("chain"),
                        pl.lit("__probe__").alias("sender"), pl.lit(1000.0).alias("amount_usd"))
          .with_row_index("event_id"))
    engine = FeatureEngine(tx)
    feats = engine.compute(ev)
    engine.close()
    scores = booster.predict(feats.select(DEST_FEATURES).to_numpy())

    stamp = dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    flagged = 0
    for row, score in zip(feats.iter_rows(named=True), scores):
        if score < threshold:
            continue
        addr = row["destination"]
        # Inbound total is not a model feature, but without it a reader cannot
        # tell a call on a wallet holding $10 from one holding $3m, and the
        # report needs that distinction to mean anything.
        senders = row.get("dest_senders_all") or 0
        inbound = float((row.get("dest_usd_per_sender") or 0) * senders)
        ledger.append({
            "address": addr, "chain": "tron", "predicted_at": stamp,
            "predicted_at_ms": now_ms, "score": round(float(score), 6),
            "threshold": round(threshold, 6), "band": "high",
            "model_sha256_16": model_sha,
            "senders": int(senders), "inbound_usd": round(inbound, 2),
            # The evidence, so the call can be audited without our warehouse.
            "features": {f: (None if row[f] is None else round(float(row[f]), 6))
                         for f in DEST_FEATURES},
        })
        flagged += 1

    print(f"scored {len(fresh):,} addresses at the high band threshold "
          f"{threshold:.4f} -> {flagged} flagged")
    print(f"ledger now {len(ledger):,} entries, head {ledger.head[:16]}")


if __name__ == "__main__":
    asyncio.run(main())

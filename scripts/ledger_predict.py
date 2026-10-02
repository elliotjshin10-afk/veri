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

import asyncio, datetime as dt, hashlib, json, logging, os, sys, urllib.parse
sys.path.insert(0, "src")

import lightgbm as lgb
import polars as pl

from veridis.chain.address import hex_to_base58
from veridis.chain.etherscan import client as es_client, token_transfers
from veridis.chain.tron import tron_client, PAGE
from veridis.config import INTERIM, PROCESSED, ROOT, SITE, TRONGRID_BASE, USDT_TRON
from veridis.dataset.ingest import TRANSFER_SCHEMA, fetch_histories
from veridis.features.asof import FeatureEngine
from veridis.chain.etherscan import BASE as ETHERSCAN_BASE
from veridis.ledger import Ledger, predictions_path, utc_now, utc_stamp
from veridis.model import browser_model
from veridis.model.address_risk import DEST_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)

N_CANDIDATES = int(sys.argv[1]) if len(sys.argv) > 1 else 400
MAX_PAGES = int(sys.argv[2]) if len(sys.argv) > 2 else 4
# Tron by default, Ethereum only when asked for explicitly.
#
# The Ethereum arm works - it samples the live USDT stream, scores with the
# model the site serves, and appends. It is not in the nightly run because a
# trial pass flagged 19 of 189 live candidates (10%), where the held-out
# evaluation puts the false-positive rate at 2%, and eleven of the nineteen had
# fewer than five payers and under $1,000 received.
#
# The cause is in the control arm, not the model. Control probes require
# `mark - 180d > first_seen`, so every ordinary address in the evaluation
# predates its pseudo-freeze by six months: median control age 999 days against
# 138 for frozen. Frozen addresses carry no such requirement, because collectors
# are short-lived. The model therefore had "old means ordinary" available as a
# shortcut, which holds in the evaluation and fails on live traffic, where a
# young address is usually just a new wallet.
#
# A ledger entry is a public claim. Until the control arm resembles the
# population the product actually sees, this one would fill the record with
# calls we have no reason to believe. Run it by hand with `... 2000 4 ethereum`
# to reproduce the finding.
CHAINS = (sys.argv[3].split(",") if len(sys.argv) > 3 else ["tron"])

ETHERSCAN_KEY = os.environ.get("ETHERSCAN_API_KEY") or \
    "9QGTZYJ7CW6K4YTWWQCK3I6NHCAN32YXXJ"
USDT_ETH_C = "0xdAC17F958D2ee523a2206206994597C13D831ec7"
TRANSFER_TOPIC = ("0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4"
                  "df523b3ef")
# The "high" band: the top 1% of ordinary wallets by the shipped model's own
# held-out control distribution. A ledger is a public claim, and one made at a
# looser budget would fill with wallets that were never going to be frozen and
# say nothing about the model.


async def sample_recent(client, n: int) -> list[str]:
    """Addresses receiving USDT in the last few hours."""
    url = f"{TRONGRID_BASE}/v1/contracts/{USDT_TRON}/events"
    now = int(utc_now().timestamp() * 1000)
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


async def sample_recent_evm(client, n: int) -> list[str]:
    """Addresses receiving USDT on Ethereum in the last few hours.

    Same idea as the Tron sampler: take the live transfer stream, which is the
    population the product sits in front of, rather than anything we curated.
    Ethereum has no events endpoint, so this reads Transfer logs over a short
    block window at a few points in the recent past - spread out rather than one
    contiguous run, so a single busy minute cannot dominate the sample.
    """
    now = int(utc_now().timestamp())
    seen: list[str] = []
    for hours_back in (2, 6, 14, 26):
        q = urllib.parse.urlencode({
            "chainid": 1, "module": "block", "action": "getblocknobytime",
            "timestamp": now - hours_back * 3600, "closest": "before",
            "apikey": ETHERSCAN_KEY})
        payload = await client.get_json(f"{ETHERSCAN_BASE}?{q}", None)
        try:
            block = int((payload or {}).get("result"))
        except (TypeError, ValueError):
            continue
        q2 = urllib.parse.urlencode({
            "chainid": 1, "module": "logs", "action": "getLogs",
            "address": USDT_ETH_C, "topic0": TRANSFER_TOPIC,
            "fromBlock": block, "toBlock": block + 30,
            "page": 1, "offset": 1000, "apikey": ETHERSCAN_KEY})
        rows = (await client.get_json(f"{ETHERSCAN_BASE}?{q2}", None) or {}).get("result")
        if not isinstance(rows, list):
            continue
        for e in rows:
            t = e.get("topics") or []
            if len(t) > 2:
                seen.append("0x" + t[2][-40:].lower())
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


def append_flagged(ledger, feats, scores, *, chain, threshold, model_sha,
                   features, now_ms, stamp) -> int:
    """Write every candidate at or above the band, with its evidence."""
    flagged = 0
    for row, score in zip(feats.iter_rows(named=True), scores):
        if score < threshold:
            continue
        # Inbound total is not a model feature, but without it a reader cannot
        # tell a call on a wallet holding $10 from one holding $3m, and the
        # report needs that distinction to mean anything.
        senders = row.get("dest_senders_all") or 0
        inbound = float((row.get("dest_usd_per_sender") or 0) * senders)
        ledger.append({
            "address": row["destination"], "chain": chain, "predicted_at": stamp,
            "predicted_at_ms": now_ms, "score": round(float(score), 6),
            "threshold": round(threshold, 6), "band": "high",
            "model_sha256_16": model_sha,
            "senders": int(senders), "inbound_usd": round(inbound, 2),
            # The evidence, so the call can be audited without our warehouse.
            "features": {f: (None if row.get(f) is None else round(float(row[f]), 6))
                         for f in features},
        })
        flagged += 1
    return flagged


async def predict_tron(ledger, listed, already) -> None:
    model_path = PROCESSED / "model_dest_latest.txt"
    model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()[:16]
    booster = lgb.Booster(model_file=str(model_path))
    # The banding threshold from the shipped model's own held-out control
    # distribution, not from the backtest file - those are different runs and
    # a threshold borrowed across them means nothing.
    report = json.loads((PROCESSED / "address_model_report.json").read_text())
    threshold = report["thresholds"]["high"]

    async with tron_client(concurrency=4) as client:
        cands = await sample_recent(client, N_CANDIDATES)
        fresh = [a for a in cands if a not in listed and a not in already][:N_CANDIDATES]
        print(f"  sampled {len(cands):,} recent recipients -> "
              f"{len(fresh):,} unlisted and new")
        if not fresh:
            return
        tx, trunc = await fetch_histories(client, fresh, max_pages=MAX_PAGES,
                                          concurrency=4, label="candidates")
    if not tx.height:
        print("  no transfer history fetched; nothing to score")
        return

    now_ms = int(utc_now().timestamp() * 1000)
    ev = (pl.DataFrame({"destination": fresh})
          .with_columns(pl.lit(now_ms).alias("event_time"), pl.lit("tron").alias("chain"),
                        pl.lit("__probe__").alias("sender"), pl.lit(1000.0).alias("amount_usd"))
          .with_row_index("event_id"))
    engine = FeatureEngine(tx)
    feats = engine.compute(ev)
    engine.close()
    scores = booster.predict(feats.select(DEST_FEATURES).to_numpy())
    n = append_flagged(ledger, feats, scores, chain="tron", threshold=threshold,
                       model_sha=model_sha, features=DEST_FEATURES,
                       now_ms=now_ms, stamp=utc_stamp())
    print(f"  scored {len(fresh):,} addresses at the high band threshold "
          f"{threshold:.4f} -> {n} flagged")


async def predict_ethereum(ledger, listed, already) -> None:
    """The Ethereum arm, scored with the model the SITE serves.

    Deliberately site/model_eth.json rather than the LightGBM booster beside it:
    that booster carries a different feature set, so using it would mean the
    ledger and the product were answering with different models while the ledger
    claimed to be the product's record. The Python walk of that JSON is held to
    LightGBM's own scores in tests/test_browser_model.py.
    """
    model_file = SITE / "model_eth.json"
    if not model_file.exists():
        print("  no site/model_eth.json - skipping Ethereum")
        return
    model_sha = hashlib.sha256(model_file.read_bytes()).hexdigest()[:16]
    model = browser_model.load(model_file)
    features = model["features"]
    threshold = model["thresholds"]["high"]

    rows: list[dict] = []
    complete: list[str] = []
    async with es_client() as client:
        cands = await sample_recent_evm(client, N_CANDIDATES)
        fresh = [a for a in cands if a not in listed and a not in already][:N_CANDIDATES]
        print(f"  sampled {len(cands):,} recent recipients -> "
              f"{len(fresh):,} unlisted and new")
        if not fresh:
            return
        sem = asyncio.Semaphore(3)
        done = [0]

        async def one(a: str) -> None:
            async with sem:
                try:
                    got, cut = await token_transfers(client, a, ETHERSCAN_KEY,
                                                     max_pages=MAX_PAGES)
                except Exception:
                    return
                finally:
                    done[0] += 1
                    if done[0] % 200 == 0:
                        logging.info("  candidates: %d/%d", done[0], len(fresh))
                # The model is fitted on complete histories only. A truncated
                # one has the wrong age and the wrong lifetime totals, and a
                # ledger entry is a public claim - not the place to score a
                # history we know is partial.
                if cut or not got:
                    return
                rows.extend(got)
                complete.append(a)

        await asyncio.gather(*(one(a) for a in fresh))

    if not complete:
        print("  no complete candidate histories; nothing to score")
        return
    tx = pl.DataFrame(rows, schema=TRANSFER_SCHEMA)
    now_ms = int(utc_now().timestamp() * 1000)
    ev = (pl.DataFrame({"destination": complete})
          .with_columns(pl.lit(now_ms).alias("event_time"),
                        pl.lit("ethereum").alias("chain"),
                        pl.lit("__probe__").alias("sender"), pl.lit(1000.0).alias("amount_usd"))
          .with_row_index("event_id"))
    engine = FeatureEngine(tx)
    feats = engine.compute(ev)
    engine.close()
    scores = browser_model.score(
        model, [[None if v is None else float(v) for v in r]
                for r in feats.select(features).iter_rows()])
    n = append_flagged(ledger, feats, scores, chain="ethereum", threshold=threshold,
                       model_sha=model_sha, features=features,
                       now_ms=now_ms, stamp=utc_stamp())
    print(f"  scored {len(complete):,} complete of {len(fresh):,} sampled at the "
          f"high band threshold {threshold:.4f} -> {n} flagged")


async def main() -> None:
    listed = listed_addresses()
    ledger = Ledger(predictions_path(ROOT))
    already = {e["address"] for e in ledger}
    ok, msg = ledger.verify()
    if not ok:
        sys.exit(f"ledger is broken, refusing to append: {msg}")
    print(f"ledger: {msg}")

    for chain in CHAINS:
        print(f"\n[{chain}]")
        if chain == "tron":
            await predict_tron(ledger, listed, already)
        elif chain == "ethereum":
            await predict_ethereum(ledger, listed, already)
        else:
            sys.exit(f"unknown chain {chain!r}")

    print(f"\nledger now {len(ledger):,} entries, head {ledger.head[:16]}")


if __name__ == "__main__":
    asyncio.run(main())

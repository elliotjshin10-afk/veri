"""Bring the Tether freeze lists up to date, incrementally, before resolving.

The nightly ledger predicted and then resolved against freeze lists that were
last fetched by hand - Tron on 2026-09-22, Ethereum on 2026-09-25 - while every
prediction in the ledger was made on 2026-09-27 or later. So each entry was
being checked against a list that could not possibly contain it, and the ledger
would have read "0 resolved" forever: not because nothing had been frozen, but
because nothing newer was ever fetched. A resolution loop that cannot resolve is
worse than none, because it looks like evidence of a low freeze rate.

This closes that. It is incremental on purpose: a full walk is 400 paginated
TronGrid pages and a 21-million-block Blockscout crawl at 0.45 requests a
second, which is a one-off job, not a nightly one. Both arms refetch only from
shortly before the newest event already stored, and MERGE - an address already
known keeps its earliest freeze time, so a short window can add facts but never
remove them.

A failed request is never read as an empty range. Partial results still merge,
because merging cannot lose anything, but the exit code is non-zero so the
nightly says so rather than quietly recording a shorter list.

Run: python scripts/refresh_freezes.py [--tron-only|--eth-only]
"""
from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import sys

sys.path.insert(0, "src")

from veridis.chain.http import CachedClient
from veridis.chain.tron import blacklist_events, tron_client
from veridis.config import RAW, USDT_ETH, USDT_TRON

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)

DAY_MS = 86_400_000
# How far back before the newest stored event to re-ask. A freeze event can be
# indexed slightly after the block it sits in, so starting exactly at the
# newest stored timestamp risks stepping over one.
OVERLAP_MS = 2 * DAY_MS
BLOCKSCOUT = "https://eth.blockscout.com/api"
ADDED_BLACKLIST_TOPIC = \
    "0x42e160154868087d6bfdc0ca23d96a1c1cfa32f1b72ba9ba27b69b98a0d819dc"
ETH_CHUNK = 200_000
ETH_MIN_SPAN = 20_000


def day(ms: int | None) -> str:
    return dt.datetime.utcfromtimestamp(ms / 1000).date().isoformat() if ms else "?"


def load(name: str) -> tuple[list[dict], dict[str, dict]]:
    path = RAW / name
    rows = json.loads(path.read_text()) if path.exists() else []
    by_addr: dict[str, dict] = {}
    for r in rows:
        a = r.get("address")
        if not a:
            continue
        prev = by_addr.get(a)
        # Earliest wins: first_reported_at is the label's timestamp, and a later
        # duplicate event would quietly push every lead time down.
        if prev is None or (r.get("block_time") or 1 << 62) < (prev.get("block_time") or 1 << 62):
            by_addr[a] = r
    return rows, by_addr


def save(name: str, by_addr: dict[str, dict], before: int) -> None:
    rows = sorted(by_addr.values(), key=lambda r: r.get("block_time") or 0)
    (RAW / name).write_text(json.dumps(rows))
    added = len(rows) - before
    newest = max((r.get("block_time") or 0) for r in rows) if rows else None
    logging.info("  %s: %d addresses (%+d), newest %s",
                 name, len(rows), added, day(newest))


async def refresh_tron() -> bool:
    name = "tether_blacklist_tron.json"
    _, by_addr = load(name)
    before = len(by_addr)
    newest = max((r.get("block_time") or 0) for r in by_addr.values()) if by_addr else 0
    since = max(newest - OVERLAP_MS, 0)
    logging.info("tron: stored to %s, refetching from %s", day(newest), day(since))

    async with tron_client(rate_per_sec=10, concurrency=4) as c:
        # blacklist_events walks TronGrid's fingerprint pagination itself; the
        # window keeps that walk to a handful of pages instead of 400.
        rows = await blacklist_events(c, USDT_TRON, max_pages=60,
                                      min_block_timestamp=since)
        ok = c.stats.get("error", 0) == 0

    for r in rows:
        a = r["address"]
        prev = by_addr.get(a)
        if prev is None or r["block_time"] < (prev.get("block_time") or 1 << 62):
            by_addr[a] = r
    logging.info("  %d events in the window", len(rows))
    save(name, by_addr, before)
    return ok


async def eth_block_at(c: CachedClient, ts_ms: int) -> int | None:
    payload = await c.get_json(BLOCKSCOUT, {
        "module": "block", "action": "getblocknobytime",
        "timestamp": ts_ms // 1000, "closest": "before"})
    # Blockscout answers {"result": {"blockNumber": "26109277"}} where Etherscan
    # answers {"result": "26109277"}. Accept either rather than depending on
    # which host is behind the Etherscan-compatible path today.
    res = (payload or {}).get("result")
    if isinstance(res, dict):
        res = res.get("blockNumber")
    try:
        return int(res)
    except (TypeError, ValueError):
        return None


async def refresh_eth() -> bool:
    name = "tether_blacklist_eth.json"
    _, by_addr = load(name)
    before = len(by_addr)
    newest = max((r.get("block_time") or 0) for r in by_addr.values()) if by_addr else 0
    if not newest:
        logging.error("eth: no stored list to extend - run scripts/fetch_eth_blacklist.py")
        return False

    ok = True
    async with CachedClient("blockscout", rate_per_sec=0.45, concurrency=1,
                            backoff_start=3.0, max_attempts=4) as c:
        lo = await eth_block_at(c, newest - OVERLAP_MS)
        tip = await eth_block_at(c, int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000))
        if lo is None or tip is None:
            logging.error("eth: could not resolve a block range - leaving the list alone")
            return False
        logging.info("eth: stored to %s, walking blocks %d-%d", day(newest), lo, tip)

        span = ETH_CHUNK
        found = 0
        while lo < tip:
            hi = min(lo + span, tip)
            payload = await c.get_json(BLOCKSCOUT, {
                "module": "logs", "action": "getLogs",
                "fromBlock": lo, "toBlock": hi,
                "address": USDT_ETH, "topic0": ADDED_BLACKLIST_TOPIC})
            res = (payload or {}).get("result") if payload else None

            # A failed request is not an empty range - the distinction is the
            # whole reason the full fetcher exists in the shape it does.
            if not isinstance(res, list):
                if span > ETH_MIN_SPAN:
                    span = max(ETH_MIN_SPAN, span // 4)
                    logging.warning("  request failed, narrowing to %d blocks", span)
                    await asyncio.sleep(5)
                    continue
                logging.error("  UNRESOLVED GAP %d-%d", lo, hi)
                ok = False
                lo = hi + 1
                continue
            if len(res) >= 1000 and span > ETH_MIN_SPAN:
                span = max(ETH_MIN_SPAN, span // 4)
                logging.info("  capped at %d, narrowing to %d blocks", len(res), span)
                continue

            for e in res:
                h = (e.get("data") or "").removeprefix("0x")
                if len(h) < 40:
                    continue
                a = ("0x" + h[-40:]).lower()
                ts = e.get("timeStamp")
                try:
                    ms = (int(ts, 16) if isinstance(ts, str) and ts.startswith("0x")
                          else int(ts)) * 1000
                except (TypeError, ValueError):
                    ms = None
                prev = by_addr.get(a)
                if prev is None or (ms or 1 << 62) < (prev.get("block_time") or 1 << 62):
                    by_addr[a] = {"address": a, "block_time": ms,
                                  "tx_hash": e.get("transactionHash")}
                found += 1
            lo = hi + 1
            if len(res) < 300:
                span = min(ETH_CHUNK, span * 2)
        logging.info("  %d events in the window", found)

    save(name, by_addr, before)
    return ok


async def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    results = []
    if arg != "--eth-only":
        results.append(await refresh_tron())
    if arg != "--tron-only":
        results.append(await refresh_eth())
    if not all(results):
        sys.exit("freeze lists refreshed with gaps - see above")


asyncio.run(main())

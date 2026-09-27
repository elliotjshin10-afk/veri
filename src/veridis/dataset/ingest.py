"""Bulk address-history ingestion with bounded concurrency and disk caching.

Pagination on TronGrid is fingerprint-chained and therefore sequential *within*
an address, so throughput comes from fetching many addresses concurrently.
Everything is cached, so re-runs are free and the pipeline is resumable.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Iterable, Sequence

import polars as pl

from veridis.chain.http import CachedClient
from veridis.chain.tron import normalise_transfers, trc20_transfers

log = logging.getLogger(__name__)

TRANSFER_SCHEMA = {
    "chain": pl.Utf8,
    "tx_hash": pl.Utf8,
    "from_address": pl.Utf8,
    "to_address": pl.Utf8,
    "amount_usd": pl.Float64,
    "asset": pl.Utf8,
    "block_time": pl.Int64,
}


async def fetch_histories(
    client: CachedClient,
    addresses: Sequence[str],
    max_pages: int = 25,
    concurrency: int = 8,
    label: str = "addresses",
) -> tuple[pl.DataFrame, dict[str, bool]]:
    """Fetch TRC-20 history for many addresses. Returns (transfers, truncated)."""
    unique = list(dict.fromkeys(addresses))
    sem = asyncio.Semaphore(concurrency)
    rows: list[dict] = []
    truncated: dict[str, bool] = {}
    done = 0
    total = len(unique)

    async def one(addr: str) -> None:
        nonlocal done
        async with sem:
            try:
                raw, trunc = await trc20_transfers(client, addr, max_pages=max_pages)
            except Exception as exc:  # noqa: BLE001 - never let one address stall a run
                log.debug("history failed for %s: %s", addr, exc)
                raw, trunc = [], False
            rows.extend(normalise_transfers(raw))
            truncated[addr] = trunc
            done += 1
            if done % 200 == 0 or done == total:
                log.info(
                    "  %s: %d/%d addresses, %d transfers, cache=%s",
                    label,
                    done,
                    total,
                    len(rows),
                    client.stats,
                )

    await asyncio.gather(*(one(a) for a in unique))
    df = (
        pl.DataFrame(rows, schema=TRANSFER_SCHEMA)
        if rows
        else pl.DataFrame(schema=TRANSFER_SCHEMA)
    )
    return df.unique(subset=["tx_hash", "from_address", "to_address", "block_time"]), truncated


def dedupe_transfers(frames: Iterable[pl.DataFrame]) -> pl.DataFrame:
    frames = [f for f in frames if f is not None and f.height]
    if not frames:
        return pl.DataFrame(schema=TRANSFER_SCHEMA)
    return pl.concat(frames, how="vertical_relaxed").unique(
        subset=["tx_hash", "from_address", "to_address", "block_time"]
    )

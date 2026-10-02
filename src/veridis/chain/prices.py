"""Daily ETH/USD, and the one rule that makes it safe to use.

Stablecoins needed none of this: USDT is a dollar, so `amount_usd` was the
transfer amount. Native ETH is not, so every dollar-denominated feature -
`usd_per_sender`, `inbound_usd_7d`, `amount_vs_sender_median` - now depends on
a price, and a price is a time series, which is exactly the shape of thing this
project keeps finding lookahead in.

THE RULE: a transfer at time T is valued at the close of the last daily candle
that had already CLOSED at T. Never the close of the day T falls in - that
price is set hours after the transfer, and using it would let the feature layer
see the future, which is the failure every other guard here exists to prevent.
The cost is at most one day of staleness; the alternative is a backtest that
cannot happen live.

Source is Coinbase Exchange's public candles: free, no key, a real USD pair
rather than a stablecoin proxy, and history back to 2016. Etherscan's daily
price endpoint is PRO-only and CoinGecko's free tier refuses anything older
than a year, so neither covers the span the freeze list does.
"""
from __future__ import annotations

import bisect
import logging
import time
from typing import Sequence

import httpx
import polars as pl

from veridis.config import INTERIM

log = logging.getLogger(__name__)

CANDLES = "https://api.exchange.coinbase.com/products/ETH-USD/candles"
STORE = INTERIM / "eth_usd_daily.parquet"
DAY = 86_400
# Coinbase returns at most 300 candles per request.
CHUNK = 300 * DAY
FIRST = 1_451_606_400          # 2016-01-01, comfortably before USDT on Ethereum


def fetch(start: int = FIRST, end: int | None = None) -> pl.DataFrame:
    """Daily ETH/USD candles, oldest first, as (day_start_s, close)."""
    end = int(end or time.time())
    rows: dict[int, float] = {}
    with httpx.Client(timeout=30.0, headers={"User-Agent": "veridis/1.0"}) as c:
        lo = start
        while lo < end:
            hi = min(lo + CHUNK, end)
            r = c.get(CANDLES, params={
                "granularity": DAY,
                "start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(lo)),
                "end": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(hi))})
            r.raise_for_status()
            got = r.json()
            if not isinstance(got, list):
                raise RuntimeError(f"coinbase refused: {str(got)[:120]!r}")
            for candle in got:
                # [time, low, high, open, close, volume]
                rows[int(candle[0])] = float(candle[4])
            lo = hi
            time.sleep(0.35)       # public endpoint; be a good citizen
    if not rows:
        raise RuntimeError("no candles returned")
    out = pl.DataFrame({"day": sorted(rows), "close": [rows[d] for d in sorted(rows)]})
    return out


def refresh() -> pl.DataFrame:
    df = fetch()
    STORE.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(STORE)
    return df


class EthUsd:
    """Point-in-time ETH/USD lookup.

    Built once and reused: a per-row lookup against a parquet would dominate the
    ingest, and the series is small enough to hold entirely in memory.
    """

    def __init__(self, df: pl.DataFrame | None = None) -> None:
        if df is None:
            if not STORE.exists():
                raise SystemExit(
                    f"no {STORE.name} - run scripts/m12_eth_prices.py first")
            df = pl.read_parquet(STORE)
        self.days: Sequence[int] = df["day"].to_list()
        self.closes: Sequence[float] = df["close"].to_list()
        if not self.days:
            raise SystemExit("price series is empty")

    def at(self, ts_ms: int) -> float | None:
        """USD per ETH for a transfer at `ts_ms`, or None if before our history.

        Strictly the last CLOSED candle: a candle starting at day D closes at
        D + 86400, so it only counts once the transfer time is past that.
        """
        t = int(ts_ms) // 1000
        # rightmost day whose close (day + DAY) is <= t
        i = bisect.bisect_right(self.days, t - DAY) - 1
        if i < 0:
            return None
        return self.closes[i]

    def __len__(self) -> int:
        return len(self.days)

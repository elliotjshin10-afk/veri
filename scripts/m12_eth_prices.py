"""Fetch the daily ETH/USD series the native-ETH features are priced against.

Run once, then again whenever the ingest moves past the last stored day.
"""
import logging, sys
sys.path.insert(0, "src")
import datetime as dt

from veridis.chain.prices import EthUsd, refresh

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")

df = refresh()
p = EthUsd(df)
first = dt.datetime.utcfromtimestamp(df["day"][0]).date()
last = dt.datetime.utcfromtimestamp(df["day"][-1]).date()
print(f"{len(df):,} daily closes, {first} -> {last}")
print(f"  min ${df['close'].min():,.2f}   max ${df['close'].max():,.2f}")

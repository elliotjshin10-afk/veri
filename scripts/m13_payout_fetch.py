"""Fetch the payout wallet for every address the PIT model can probe.

m13_payout_ingest.py covered the 1,747 destinations in the event-level dataset,
which was enough to measure whether the signal is worth having. The shipped
destination model is trained on a different and much larger universe: every
address whose own history we hold, probed at horizons before its mark. Training
it with the feature means holding the second hop for all of them.

Appends to data/interim/payout_transfers.parquet rather than replacing it, and
skips wallets already fetched, so this is resumable. It is a long run against a
rate-limited API and it will be interrupted; being able to start it again
without paying for the same pages twice is the difference between a job and an
ordeal.

Run: python scripts/m13_payout_fetch.py [max_pages] [batch]
"""
from __future__ import annotations

import asyncio
import sys

sys.path.insert(0, "src")

import polars as pl

from veridis.chain.tron import tron_client
from veridis.config import INTERIM
from veridis.dataset.holdings import all_indexed, all_transfers
from veridis.dataset.ingest import fetch_histories

MAX_PAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 15
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 700

TX = INTERIM / "payout_transfers.parquet"
TRUNC = INTERIM / "payout_truncated.parquet"


def payout_targets() -> list[str]:
    """Every indexed address's largest payout wallet, by total value sent."""
    tf = all_transfers()
    idx = all_indexed()
    top = (tf.filter(pl.col("from_address").is_in(list(idx)) & (pl.col("amount_usd") > 0))
           .group_by(["from_address", "to_address"])
           .agg(pl.col("amount_usd").sum().alias("v"))
           .sort("v", descending=True)
           .group_by("from_address").first())
    return sorted(set(top["to_address"].to_list()))


async def main() -> None:
    targets = payout_targets()
    have = set(pl.read_parquet(TRUNC)["address"].to_list()) if TRUNC.exists() else set()
    todo = [a for a in targets if a not in have]
    print(f"{len(targets):,} payout wallets in the PIT universe; "
          f"{len(have):,} already held, {len(todo):,} to fetch")
    if not todo:
        print("nothing to do")
        return

    # Write after every batch. A two-hour run against a rate-limited API gets
    # interrupted, and finishing with nothing to show for the requests already
    # paid for is the avoidable part of that.
    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        async with tron_client(concurrency=4) as c:
            tx, trunc = await fetch_histories(c, chunk, max_pages=MAX_PAGES,
                                              concurrency=4,
                                              label=f"payout {i//BATCH + 1}")
        old_tx = pl.read_parquet(TX) if TX.exists() else None
        merged = (pl.concat([old_tx, tx.select(old_tx.columns)], how="vertical_relaxed")
                  .unique(subset=["tx_hash", "to_address"]) if old_tx is not None else tx)
        merged.write_parquet(TX)
        old_tr = pl.read_parquet(TRUNC) if TRUNC.exists() else None
        add = pl.DataFrame({"address": list(trunc),
                            "truncated": list(trunc.values())})
        tr = (pl.concat([old_tr, add.select(old_tr.columns)], how="vertical_relaxed")
              .unique(subset=["address"]) if old_tr is not None else add)
        tr.write_parquet(TRUNC)
        print(f"  checkpoint: {tr.height:,} wallets held, "
              f"{merged.height:,} transfers")


asyncio.run(main())

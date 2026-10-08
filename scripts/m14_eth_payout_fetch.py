"""The second hop on Ethereum, so the two chains answer the same question.

Tron's destination model reads one address further than the address you are
paying: it finds the wallet that address pays most, reads that wallet's history,
and counts how many other wallets pay it. A low count is the warning, because a
mule wallet is fed by a handful of addresses and an exchange hot wallet by
thousands. Worth +2.5pp of recall at the shipped operating point.

Ethereum could not do it because the second hop was not held, so the two chains
were answering different questions while the page presented one product. This
fetches it: every payout wallet for every Ethereum address whose own history we
hold, 3,900 of them beyond what is already stored.

Resumable and checkpointed, like the Tron side. Etherscan's limit is 2.5
requests a second against a shared key, which makes this an hour rather than a
minute, and an interrupted run that had to start over would make it two.

Run: python scripts/m14_eth_payout_fetch.py [max_pages] [batch]
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys

sys.path.insert(0, "src")

import polars as pl

from veridis.chain.etherscan import client as es_client, normalise, token_transfers
from veridis.config import INTERIM
from veridis.dataset.holdings import eth_complete, eth_fetched, eth_transfers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)

MAX_PAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 4
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 400
KEY = os.environ.get("ETHERSCAN_API_KEY") or "9QGTZYJ7CW6K4YTWWQCK3I6NHCAN32YXXJ"

TX = INTERIM / "eth_payout_transfers.parquet"
TRUNC = INTERIM / "eth_payout_truncated.parquet"


def targets() -> list[str]:
    """Every complete address's largest payout wallet, ties broken on address.

    The same rule as veridis.features.payout.links, and for the same reason:
    the browser has to be able to reach the same wallet, and an arbitrary pick
    among equal sums is not reachable.
    """
    tf = eth_transfers()
    comp = eth_complete()
    top = (tf.filter(pl.col("from_address").is_in(list(comp))
                     & (pl.col("amount_usd") > 0))
           .group_by(["from_address", "to_address"])
           .agg(pl.col("amount_usd").sum().alias("v"))
           .sort(["v", "to_address"], descending=[True, False])
           .group_by("from_address").first())
    return sorted(set(top["to_address"].to_list()))


async def main() -> None:
    want = targets()
    # An address whose own history we already hold needs no second fetch: the
    # inbound edges are already in the warehouse, completely.
    held = eth_fetched()
    have = set(pl.read_parquet(TRUNC)["address"].to_list()) if TRUNC.exists() else set()
    todo = [a for a in want if a not in held and a not in have]
    print(f"{len(want):,} payout wallets; {len(want) - len(todo):,} already held, "
          f"{len(todo):,} to fetch")
    if not todo:
        print("nothing to do")
        return

    for i in range(0, len(todo), BATCH):
        chunk = todo[i:i + BATCH]
        rows: list[dict] = []
        trunc: dict[str, bool] = {}
        failed = 0
        async with es_client() as c:
            sem = asyncio.Semaphore(3)

            async def one(a: str) -> None:
                nonlocal failed
                async with sem:
                    try:
                        raw, cut = await token_transfers(c, a, KEY,
                                                         max_pages=MAX_PAGES)
                        rows.extend(normalise(raw))
                        trunc[a] = cut
                    except Exception as exc:          # noqa: BLE001
                        # A refusal raises rather than returning a short history
                        # marked complete. One wallet lost beats a feature
                        # quietly computed on a fragment.
                        failed += 1
                        logging.debug("  %s failed: %s", a, str(exc)[:90])

            await asyncio.gather(*(one(a) for a in chunk))

        new = pl.DataFrame(rows) if rows else None
        if new is not None:
            old = pl.read_parquet(TX) if TX.exists() else None
            merged = (pl.concat([old, new.select(old.columns)], how="vertical_relaxed")
                      .unique(subset=["tx_hash", "to_address"]) if old is not None else new)
            merged.write_parquet(TX)
        old_tr = pl.read_parquet(TRUNC) if TRUNC.exists() else None
        add = pl.DataFrame({"address": list(trunc),
                            "truncated": [bool(v) for v in trunc.values()]})
        tr = (pl.concat([old_tr, add.select(old_tr.columns)], how="vertical_relaxed")
              .unique(subset=["address"]) if old_tr is not None and add.height else
              (add if old_tr is None else old_tr))
        tr.write_parquet(TRUNC)
        print(f"  checkpoint: {tr.height:,} wallets held, "
              f"{(pl.read_parquet(TX).height if TX.exists() else 0):,} transfers, "
              f"{failed} failed this batch")


asyncio.run(main())

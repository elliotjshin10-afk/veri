"""Rebuild transfer tables from the HTTP cache alone, making no requests.

The chain APIs enforce a depleting quota, so an ingest can stall part-way
through. Everything fetched is already on disk, so this harvests whatever
landed and writes the same parquet files the online stages would - letting the
rest of the pipeline run at the scale actually achieved, rather than waiting
on a quota window.
"""
import asyncio, logging, sys
sys.path.insert(0, "src")
import polars as pl
from veridis.chain.tron import tron_client
from veridis.dataset.ingest import fetch_histories
from veridis.config import INTERIM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)

WHICH = sys.argv[1] if len(sys.argv) > 1 else "victims"
PAGES = int(sys.argv[2]) if len(sys.argv) > 2 else 3

if WHICH == "victims":
    addrs = pl.read_parquet(INTERIM / "victims.parquet")["victim_address"].unique().to_list()
    out_tx, out_tr = "victim_transfers.parquet", "victim_truncated.parquet"
else:
    seeds_p = INTERIM / "control_seeds.parquet"
    if not seeds_p.exists():
        raise SystemExit(
            "no control seeds yet - run `make controls` at least far enough to "
            "sample the firehose, then harvest."
        )
    seeds = pl.read_parquet(seeds_p)["address"].to_list()
    peers_p = INTERIM / "control_peers.parquet"
    peers = pl.read_parquet(peers_p)["address"].to_list() if peers_p.exists() else []
    addrs = list(dict.fromkeys(seeds + peers))
    out_tx, out_tr = "control_transfers.parquet", "control_truncated.parquet"

async def main():
    async with tron_client(concurrency=1, offline=True) as c:
        df, trunc = await fetch_histories(c, addrs, max_pages=PAGES,
                                          concurrency=1, label=f"harvest-{WHICH}")
        # Count addresses we actually fetched, by checking whether their own
        # first-page request is in the cache. Counting addresses that merely
        # appear in the transfer table overstates coverage badly - a victim
        # shows up as a counterparty in the scam address's history without us
        # ever having fetched that victim.
        from veridis.config import TRONGRID_BASE
        from veridis.chain.tron import PAGE
        have = 0
        for a in addrs:
            url = f"{TRONGRID_BASE}/v1/accounts/{a}/transactions/trc20"
            if c.peek(url, {"limit": PAGE, "order_by": "block_timestamp,asc"}) is not None:
                have += 1
        print(f"{WHICH}: {have}/{len(addrs)} addresses actually fetched, "
              f"{df.height:,} transfers  stats={c.stats}")
        if df.height:
            df.write_parquet(INTERIM / out_tx)
            pl.DataFrame({"address": list(trunc), "truncated": list(trunc.values())},
                         schema={"address": pl.Utf8, "truncated": pl.Boolean}
                         ).write_parquet(INTERIM / out_tr)
            print(f"wrote {out_tx}")

asyncio.run(main())

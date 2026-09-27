"""M9c - fetch every accepted Tron scam address we do not already hold.

Coverage is the product here. A live lookup works for any address, but a lookup
that hits the index answers instantly, works with no network, and is the only
thing that works in a sandboxed preview. Every address on Tether's Tron freeze
list should therefore be indexed - if somebody pastes a known-bad address, the
worst outcome is that we go and ask TronGrid about it.

Resumable: it skips what is already on disk and appends, so an interrupted run
costs only the addresses in flight.
"""
import asyncio, logging, sys
sys.path.insert(0, "src")
import polars as pl
from veridis.chain.tron import tron_client
from veridis.dataset.ingest import fetch_histories
from veridis.config import INTERIM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0      # 0 = everything left
MAX_PAGES = int(sys.argv[2]) if len(sys.argv) > 2 else 8

OUT, TRUNC = INTERIM / "coverage_transfers.parquet", INTERIM / "coverage_truncated.parquet"

want = set(pl.read_parquet(INTERIM / "scam_addresses.parquet").filter(
    (pl.col("chain") == "tron") & pl.col("accepted"))["address"].to_list())

have: set[str] = set()
for name in ("scam_truncated", "victim_truncated", "control_truncated",
             "leadtime_truncated", "coverage_truncated"):
    p = INTERIM / f"{name}.parquet"
    if p.exists():
        have |= set(pl.read_parquet(p)["address"].to_list())

todo = sorted(want - have)
if LIMIT:
    todo = todo[:LIMIT]
print(f"accepted tron scam addresses {len(want):,}; already fetched {len(want & have):,}; "
      f"fetching {len(todo):,} now")

async def main():
    if not todo:
        print("nothing to do")
        return
    async with tron_client(concurrency=4) as c:
        df, trunc = await fetch_histories(c, todo, max_pages=MAX_PAGES,
                                          concurrency=4, label="coverage")
        if OUT.exists():
            df = pl.concat([pl.read_parquet(OUT), df], how="vertical_relaxed")
        df.unique(subset=["tx_hash", "to_address"]).write_parquet(OUT)
        new = pl.DataFrame({"address": list(trunc), "truncated": list(trunc.values())})
        if TRUNC.exists():
            new = pl.concat([pl.read_parquet(TRUNC), new], how="vertical_relaxed").unique("address")
        new.write_parquet(TRUNC)
        print(f"transfers now {df.height:,}; addresses indexed {new.height:,}; stats={c.stats}")

asyncio.run(main())

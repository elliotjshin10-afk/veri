"""M9a - fetch histories for every Tron address Tether froze recently.

The lead-time question is: how long before a freeze did this address already
look like a collection point to us? Answering it honestly needs the *whole*
recent-freeze population, not the addresses an earlier ingest happened to pick.
Only 161 of the 2,231 addresses frozen in the last 180 days had history, and
that 161 was chosen by the victim-driven M1 sample - addresses with traceable
victims, which is plainly correlated with being detectable. Reporting a
detection rate on that subset would measure the sampling, not the model.

So this fetches all of them. Writes its own parquet; scam_transfers.parquet is
the trained model's input and is not touched.

Ascending order matters here: scoring an address as of T-30 needs the transfers
*before* that point, so when a history is truncated we want the early end, and
the truncation point is recorded so the scorer can drop horizons it cannot see.
"""
import asyncio, logging, sys, datetime as dt
sys.path.insert(0, "src")
import polars as pl
from veridis.chain.tron import tron_client
from veridis.dataset.ingest import fetch_histories
from veridis.config import INTERIM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)

WINDOW_DAYS = int(sys.argv[1]) if len(sys.argv) > 1 else 180
MAX_PAGES = int(sys.argv[2]) if len(sys.argv) > 2 else 12

labels = pl.read_parquet(INTERIM / "scam_addresses.parquet").filter(
    (pl.col("chain") == "tron") & pl.col("accepted")
    & pl.col("first_reported_at").is_not_null())
cut = (dt.datetime.utcnow() - dt.timedelta(days=WINDOW_DAYS)).timestamp() * 1000
recent = labels.filter(pl.col("first_reported_at") >= cut).sort("first_reported_at")

print(f"tron addresses frozen in the last {WINDOW_DAYS} days: {recent.height:,}")
print(f"  freeze dates {dt.datetime.utcfromtimestamp(recent['first_reported_at'].min()/1000).date()}"
      f" -> {dt.datetime.utcfromtimestamp(recent['first_reported_at'].max()/1000).date()}")

async def main():
    async with tron_client(concurrency=4) as c:
        df, trunc = await fetch_histories(
            c, recent["address"].to_list(), max_pages=MAX_PAGES,
            concurrency=4, label="recent-freezes")
        df.write_parquet(INTERIM / "leadtime_transfers.parquet")
        pl.DataFrame({"address": list(trunc), "truncated": list(trunc.values())}
                     ).write_parquet(INTERIM / "leadtime_truncated.parquet")
        recent.write_parquet(INTERIM / "leadtime_labels.parquet")
        print(f"transfers={df.height:,} addresses={recent.height:,} stats={c.stats}")

asyncio.run(main())

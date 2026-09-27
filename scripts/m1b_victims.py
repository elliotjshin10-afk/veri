"""M1b - extract victims from scam-address inbound flow, then fetch their history.

Victim histories are the expensive half of the ingest: one address-fetch each.
Their destinations (the scam addresses) are already fetched by M1a, so both
sides of a positive event have complete history - which is what lets us give
controls the same completeness in M2.
"""
import asyncio, logging, sys
sys.path.insert(0, "src")
import polars as pl
from veridis.chain.tron import tron_client
from veridis.dataset.ingest import fetch_histories
from veridis.dataset.events import (
    profile_destinations, select_retail_collection, extract_victims,
)
from veridis.config import INTERIM, SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)

MAX_VICTIMS = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
MAX_PAGES = int(sys.argv[2]) if len(sys.argv) > 2 else 6
MAX_SENDS = 40
MAX_TOTAL_USD = 250_000.0
PER_SCAM = 12

transfers = pl.read_parquet(INTERIM / "scam_transfers.parquet")
labels = pl.read_parquet(INTERIM / "scam_addresses.parquet").filter(
    (pl.col("chain") == "tron") & pl.col("accepted")
)
fetched = pl.Series(sorted(set(transfers["to_address"]) | set(transfers["from_address"])))
cands = labels.filter(pl.col("address").is_in(fetched))
print(f"scam addresses with history: {cands.height}")

prof = profile_destinations(transfers, cands["address"])
retail = select_retail_collection(prof)
print(f"retail-collection profile: {retail.height} of {prof.height}")
retail.write_parquet(INTERIM / "scam_profiles.parquet")

keep = labels.filter(pl.col("address").is_in(retail["address"]))
victims = extract_victims(transfers, keep)
print(f"victim relationships: {victims.height}, distinct victims: {victims['victim_address'].n_unique()}")

# Select the retail victim population the product is actually for.
#
# Sorting by send count picks whales: the top 800 relationships averaged $580k
# each, which is OTC/infrastructure flow, not someone being talked into a
# transfer. Bound the population to retail scale instead, and require >= 2
# sends so there is a sequence for early detection to be early within.
retail = victims.filter(
    (pl.col("n_sends") >= 2)
    & (pl.col("n_sends") <= MAX_SENDS)
    & (pl.col("total_sent_usd") <= MAX_TOTAL_USD)
)
print(f"retail-plausible relationships: {retail.height} "
      f"(median ${retail['total_sent_usd'].median():,.0f} over "
      f"{retail['n_sends'].median():.0f} transfers)")
# Spread across scam operations so no single ring dominates the groups, and
# sample *within* each operation rather than taking its largest losses -
# ranking by size inside a ring reproduces the same whale bias at a smaller
# scale and would overstate dollars-protected.
sel = (
    retail.with_columns(
        pl.int_range(pl.len()).shuffle(seed=SEED).over("scam_address").alias("_r")
    )
    .filter(pl.col("_r") < PER_SCAM)
    .drop("_r")
    .sample(fraction=1.0, shuffle=True, seed=SEED)
    .head(MAX_VICTIMS)
)
print(f"selected {sel.height} relationships across "
      f"{sel['scam_address'].n_unique()} scam addresses")
victims.write_parquet(INTERIM / "victims_all.parquet")
sel.write_parquet(INTERIM / "victims.parquet")
print(f"fetching history for {sel['victim_address'].n_unique()} victims")

async def main():
    async with tron_client(concurrency=1, backoff_start=8.0) as c:
        df, trunc = await fetch_histories(
            c, sel["victim_address"].unique().to_list(),
            max_pages=MAX_PAGES, concurrency=1, label="victim")
        df.write_parquet(INTERIM / "victim_transfers.parquet")
        pl.DataFrame({"address": list(trunc), "truncated": list(trunc.values())},
                     schema={"address": pl.Utf8, "truncated": pl.Boolean}
                     ).write_parquet(INTERIM / "victim_truncated.parquet")
        print(f"victim transfers={df.height} stats={c.stats}")

asyncio.run(main())

"""M1a - fetch transfer histories for a stratified sample of scam addresses.

Ascending order is deliberate: the earliest inbound transfers are where victims
first appear, which is exactly the window the early-detection metric cares about.
"""
import asyncio, logging, sys
sys.path.insert(0, "src")
import polars as pl
from veridis.chain.tron import tron_client
from veridis.dataset.ingest import fetch_histories
from veridis.config import INTERIM, SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 600
MAX_PAGES = int(sys.argv[2]) if len(sys.argv) > 2 else 12

labels = pl.read_parquet(INTERIM / "scam_addresses.parquet").filter(
    (pl.col("chain") == "tron") & pl.col("accepted")
    & pl.col("first_reported_at").is_not_null()
)
# Stratify across freeze half-years so the temporal split has material on both
# sides; restrict to 2023+ where pig-butchering volume actually lives.
labels = labels.with_columns(
    pl.from_epoch("first_reported_at", time_unit="ms").alias("frozen_at")
).filter(pl.col("frozen_at").dt.year() >= 2023)
labels = labels.with_columns(
    (pl.col("frozen_at").dt.year().cast(pl.Utf8) + "H"
     + ((pl.col("frozen_at").dt.month() > 6).cast(pl.Int8) + 1).cast(pl.Utf8)).alias("stratum")
)
per = max(1, N // labels["stratum"].n_unique())
sample = (labels.sort("address")
          .group_by("stratum", maintain_order=True)
          .head(per))
print(f"sampling {sample.height} scam addresses across {sample['stratum'].n_unique()} strata")
print(sample.group_by("stratum").len().sort("stratum"))

async def main():
    async with tron_client(concurrency=1, backoff_start=8.0) as c:
        df, trunc = await fetch_histories(
            c, sample["address"].to_list(), max_pages=MAX_PAGES,
            concurrency=1, label="scam")
        df.write_parquet(INTERIM / "scam_transfers.parquet")
        pl.DataFrame({"address": list(trunc), "truncated": list(trunc.values())}
                     ).write_parquet(INTERIM / "scam_truncated.parquet")
        print(f"transfers={df.height} addrs={sample.height} stats={c.stats}")

asyncio.run(main())

"""Fetch native ETH movements for a subset of the Ethereum population.

The question this exists to answer is narrow: does reading native ETH make the
model better? Today we read stablecoins only, and 80% of frozen Ethereum
addresses also receive ether we never look at.

Scoped as an A/B on the SAME addresses rather than an ingest of everything.
That is not just to save time - it is the only way to get a clean answer. If
ETH were fetched for some addresses and not others, "has ETH data" would itself
separate the classes, and the comparison would measure our own sampling instead
of the signal.

So: pick a subset, fetch ether for ALL of it, then train twice on exactly that
subset - once on stablecoins alone, once on stablecoins plus ether. Any
difference is the ether.

Run: python scripts/m12_eth_native.py [N_POS] [N_CTRL] [MAX_PAGES]
"""
from __future__ import annotations

import asyncio, json, logging, random, sys
sys.path.insert(0, "src")

import polars as pl

from veridis.chain.etherscan import client as es_client, native_transfers
from veridis.chain.prices import EthUsd
from veridis.config import INTERIM, PROCESSED
from veridis.dataset.holdings import eth_complete
from veridis.dataset.ingest import TRANSFER_SCHEMA, dedupe_transfers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)

KEY = "9QGTZYJ7CW6K4YTWWQCK3I6NHCAN32YXXJ"
N_POS = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
N_CTRL = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
MAX_PAGES = int(sys.argv[3]) if len(sys.argv) > 3 else 4
SEED = 12

OUT = INTERIM / "eth_native_transfers.parquet"
TRUNC = INTERIM / "eth_native_truncated.parquet"
COHORT = PROCESSED / "eth_native_cohort.json"


def main() -> None:
    price = EthUsd()
    complete = eth_complete()
    sa = pl.read_parquet(INTERIM / "scam_addresses.parquet")
    pos_all = sorted(set(
        sa.filter((pl.col("chain") == "ethereum") & pl.col("accepted")
                  & pl.col("first_reported_at").is_not_null())["address"].to_list()
    ) & complete)
    scam = set(sa.filter(pl.col("accepted"))["address"].to_list())
    c3 = pl.read_parquet(INTERIM / "eth_control3_truncated.parquet")
    ctrl_all = sorted((set(c3.filter(~pl.col("truncated"))["address"].to_list())
                       & complete) - scam)

    rng = random.Random(SEED)
    pos = rng.sample(pos_all, min(N_POS, len(pos_all)))
    ctrl = rng.sample(ctrl_all, min(N_CTRL, len(ctrl_all)))
    logging.info("cohort: %d frozen of %d available, %d ordinary of %d",
                 len(pos), len(pos_all), len(ctrl), len(ctrl_all))
    COHORT.write_text(json.dumps({"frozen": pos, "ordinary": ctrl,
                                  "seed": SEED, "max_pages": MAX_PAGES}, indent=1))

    done_already = set()
    if TRUNC.exists():
        done_already = set(pl.read_parquet(TRUNC)["address"].to_list())
    pool = [a for a in pos + ctrl if a not in done_already]
    logging.info("%d addresses to fetch (%d already held)",
                 len(pool), len(pos) + len(ctrl) - len(pool))
    if not pool:
        print("nothing to fetch")
        return

    rows: list[dict] = []
    trunc: dict[str, bool] = {}
    done = failed = 0

    def flush() -> None:
        if rows:
            df = pl.DataFrame(rows, schema=TRANSFER_SCHEMA)
            if OUT.exists():
                df = pl.concat([pl.read_parquet(OUT), df], how="vertical_relaxed")
            dedupe_transfers([df]).write_parquet(OUT)
        if trunc:
            t = pl.DataFrame({"address": list(trunc), "truncated": list(trunc.values())})
            if TRUNC.exists():
                t = pl.concat([pl.read_parquet(TRUNC), t],
                              how="vertical_relaxed").unique("address")
            t.write_parquet(TRUNC)
        rows.clear()

    async def run() -> None:
        nonlocal done, failed
        async with es_client() as c:
            sem = asyncio.Semaphore(3)

            async def one(a: str) -> None:
                nonlocal done, failed
                async with sem:
                    try:
                        got, cut = await native_transfers(c, a, KEY, price,
                                                          max_pages=MAX_PAGES)
                        rows.extend(got)
                        trunc[a] = cut
                    except Exception:
                        failed += 1
                    done += 1
                    if done % 200 == 0:
                        flush()
                        logging.info("  %d/%d, %d failed, %d ether transfers",
                                     done, len(pool), failed,
                                     pl.read_parquet(OUT).height if OUT.exists() else 0)

            await asyncio.gather(*(one(a) for a in pool))
            flush()

    asyncio.run(run())
    t = pl.read_parquet(TRUNC)
    df = pl.read_parquet(OUT) if OUT.exists() else None
    print(f"done: {t.height:,} addresses, {failed} failed, "
          f"{0 if df is None else df.height:,} native ETH transfers")


if __name__ == "__main__":
    main()

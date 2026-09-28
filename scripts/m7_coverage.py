"""Fetch histories for every Ethereum address on Tether's freeze list.

Ethereum has 3,125 labels with freeze dates and we hold histories for 600 of
them. That gap is the whole reason the cross-chain result was weak: the Tron
model went from 0.770 to 0.951 when it stopped being trained on 117 addresses
and got 8,395, and an Ethereum model built on 600 would be starved the same way.

Blockscout pages NEWEST-first, the opposite of TronGrid, so a truncated history
is missing an address's earliest activity - which corrupts age at every point
in time rather than only late ones. max_pages is therefore generous, and the
truncation flag is recorded so the model build can keep only complete
histories, as m7_evaluate already does.

Resumable: anything already on disk is skipped, and progress is checkpointed,
because Blockscout refuses often enough that a single long run will not finish.
"""
import asyncio, logging, sys
sys.path.insert(0, "src")
import polars as pl
from veridis.chain.evm import erc20_transfers, evm_client
from veridis.dataset.ingest import TRANSFER_SCHEMA, dedupe_transfers
from veridis.config import INTERIM

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)

LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0
MAX_PAGES = int(sys.argv[2]) if len(sys.argv) > 2 else 8
CHECKPOINT = 100

OUT = INTERIM / "eth_coverage_transfers.parquet"
TRUNC = INTERIM / "eth_coverage_truncated.parquet"


def already() -> set[str]:
    out: set[str] = set()
    for n in ("eth_truncated", "eth_coverage_truncated"):
        p = INTERIM / f"{n}.parquet"
        if p.exists():
            out |= set(pl.read_parquet(p)["address"].to_list())
    return out


async def main() -> None:
    labels = pl.read_parquet(INTERIM / "scam_addresses.parquet").filter(
        (pl.col("chain") == "ethereum") & pl.col("accepted")
        & pl.col("first_reported_at").is_not_null())
    have = already()
    todo = sorted(set(labels["address"].to_list()) - have)
    if LIMIT:
        todo = todo[:LIMIT]
    print(f"ethereum labels {labels.height:,}; already held {len(have & set(labels['address'])):,}; "
          f"fetching {len(todo):,}")
    if not todo:
        return

    rows, trunc, failed = [], {}, 0
    def flush():
        if not rows:
            return
        df = pl.DataFrame(rows, schema=TRANSFER_SCHEMA)
        if OUT.exists():
            df = pl.concat([pl.read_parquet(OUT), df], how="vertical_relaxed")
        dedupe_transfers([df]).write_parquet(OUT)
        t = pl.DataFrame({"address": list(trunc), "truncated": list(trunc.values())})
        if TRUNC.exists():
            t = pl.concat([pl.read_parquet(TRUNC), t], how="vertical_relaxed").unique("address")
        t.write_parquet(TRUNC)

    async with evm_client() as c:
        for i, a in enumerate(todo, 1):
            try:
                raw, cut = await erc20_transfers(c, a, max_pages=MAX_PAGES)
                rows.extend(raw)
                trunc[a] = cut
            except Exception as exc:               # a refusal is not an empty history
                failed += 1
                logging.debug("failed %s: %s", a, exc)
            if i % CHECKPOINT == 0:
                flush(); rows.clear()
                logging.info("  %d/%d fetched, %d failed, %d transfers held",
                             i, len(todo), failed, len(trunc))
    flush()
    n = pl.read_parquet(TRUNC).height if TRUNC.exists() else 0
    print(f"done: {n:,} addresses indexed, {failed} refused "
          f"({'rerun to retry those' if failed else 'none to retry'})")

asyncio.run(main())

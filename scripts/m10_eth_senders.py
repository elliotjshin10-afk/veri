"""Fetch the histories of the SENDERS on both Ethereum arms.

The destination-only model answers "what is this address", which is all an
address lookup can ask. The product asks something better: at the moment of
sending, the sender and the amount are visible too, and the relationship
between the two addresses is the strongest signal there is - a first transfer
to a young collector reads very differently from the fortieth to a supplier.

Those relationship features need BOTH sides' histories. We already hold every
destination's history; this fetches the senders'.

Both arms are sampled by the same procedure, which is the part that matters:

  positives  senders who paid one of the frozen RETAIL collection points,
             strictly before its freeze date
  negatives  senders who paid one of the independently-sampled ordinary
             wallets - drawn from historical USDT block windows with no
             knowledge of any scam

An earlier Ethereum control arm was built from counterparties already in our
pool, and 97% of them turned out to be counterparties OF FROZEN ADDRESSES:
victims, mules and cash-out points. Separating a collector from its own
neighbours is a different, easier question than the one the product asks, and
it inflated the result. Hence the independent destinations here.

Senders are taken up to K per destination rather than in bulk, so coverage
spreads across many destinations instead of concentrating on the few busiest.

Run: python scripts/m10_eth_senders.py [K_POS] [K_NEG] [MAX_PAGES]
"""
from __future__ import annotations

import asyncio, logging, sys
sys.path.insert(0, "src")

import polars as pl

from veridis.chain.etherscan import client as es_client, token_transfers
from veridis.config import INTERIM
from veridis.dataset.events import (RETAIL_MAX_SENDERS, RETAIL_MAX_USD_PER_SENDER,
                                    RETAIL_MIN_INBOUND_USD, RETAIL_MIN_SENDERS)
from veridis.dataset.holdings import eth_fetched, eth_transfers
from veridis.dataset.ingest import TRANSFER_SCHEMA, dedupe_transfers

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)

KEY = "9QGTZYJ7CW6K4YTWWQCK3I6NHCAN32YXXJ"
K_POS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
K_NEG = int(sys.argv[2]) if len(sys.argv) > 2 else 2
MAX_PAGES = int(sys.argv[3]) if len(sys.argv) > 3 else 8
SEED = 90

OUT = INTERIM / "eth_sender_transfers.parquet"
TRUNC = INTERIM / "eth_sender_truncated.parquet"


def retail_frozen(tf: pl.DataFrame) -> pl.DataFrame:
    """Frozen addresses that look like retail collection points.

    Tether also freezes laundering and exchange infrastructure. Without this
    filter the "victims" of a $99m address are other criminal wallets, which
    corrupts the very behaviour we are trying to learn. Same thresholds as the
    Tron side, so the two chains mean the same thing by "collection point".
    """
    sa = pl.read_parquet(INTERIM / "scam_addresses.parquet")
    frozen = (sa.filter((pl.col("chain") == "ethereum") & pl.col("accepted")
                        & pl.col("first_reported_at").is_not_null())
              .select("address", pl.col("first_reported_at").alias("frozen_at")))
    prof = (tf.join(frozen, left_on="to_address", right_on="address", how="inner")
            .group_by("to_address", "frozen_at")
            .agg(pl.col("from_address").n_unique().alias("senders"),
                 pl.col("amount_usd").sum().alias("usd")))
    return prof.filter(
        (pl.col("senders") >= RETAIL_MIN_SENDERS)
        & (pl.col("senders") <= RETAIL_MAX_SENDERS)
        & ((pl.col("usd") / pl.col("senders")) <= RETAIL_MAX_USD_PER_SENDER)
        & (pl.col("usd") >= RETAIL_MIN_INBOUND_USD)
    ).select("to_address", "frozen_at")


def take_per_dest(ev: pl.DataFrame, k: int) -> pl.DataFrame:
    """Up to k distinct senders per destination, chosen deterministically."""
    first = (ev.group_by("to_address", "from_address")
             .agg(pl.col("block_time").min().alias("t")))
    return (first.sort(["to_address", "t", "from_address"])
            .with_columns(pl.int_range(pl.len()).over("to_address").alias("i"))
            .filter(pl.col("i") < k).select("to_address", "from_address"))


def main() -> None:
    tf = eth_transfers()
    sa = pl.read_parquet(INTERIM / "scam_addresses.parquet")
    scam = set(sa.filter(pl.col("accepted"))["address"].to_list())

    # ---- positive arm: payers of retail collection points, pre-freeze ----
    retail = retail_frozen(tf)
    pos_ev = (tf.join(retail, on="to_address", how="inner")
              .filter((pl.col("block_time") < pl.col("frozen_at"))
                      & (pl.col("amount_usd") > 0)))
    pos = take_per_dest(pos_ev, K_POS)
    pos_senders = set(pos["from_address"].to_list()) - scam
    logging.info("positive arm: %d retail destinations, %d candidate senders",
                 retail.height, len(pos_senders))

    # ---- negative arm: payers of the independent ordinary wallets ----
    c3 = pl.read_parquet(INTERIM / f"eth_control3_truncated.parquet")
    c3_complete = set(c3.filter(~pl.col("truncated"))["address"].to_list())
    neg_ev = (tf.filter(pl.col("to_address").is_in(list(c3_complete))
                        & (pl.col("amount_usd") > 0)))
    neg = take_per_dest(neg_ev, K_NEG)
    # A sender on the positive arm must not also be a control sender: it is a
    # victim, and its history carries the scam relationship we are learning.
    neg_senders = set(neg["from_address"].to_list()) - scam - pos_senders
    logging.info("negative arm: %d independent destinations, %d candidate senders",
                 len(c3_complete), len(neg_senders))

    have = eth_fetched()
    pool = sorted((pos_senders | neg_senders) - have)
    logging.info("%d senders to fetch (%d already held)",
                 len(pool), len(pos_senders | neg_senders) - len(pool))
    if not pool:
        print("nothing to fetch")
        return

    rows: list[dict] = []
    trunc: dict[str, bool] = {}
    done = failed = 0

    def flush() -> None:
        """Checkpoint. A two-hour ingest that keeps everything in memory and
        dies at 90 minutes has cost two hours and bought nothing."""
        if not rows and not trunc:
            return
        if rows:
            df = pl.DataFrame(rows, schema=TRANSFER_SCHEMA)
            if OUT.exists():
                df = pl.concat([pl.read_parquet(OUT), df], how="vertical_relaxed")
            dedupe_transfers([df]).write_parquet(OUT)
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
                        raw, cut = await token_transfers(c, a, KEY,
                                                         max_pages=MAX_PAGES)
                        rows.extend(raw)
                        trunc[a] = cut
                    except Exception as exc:
                        # A refusal now raises rather than returning a short
                        # history marked complete, so a failure here costs one
                        # sender instead of silently corrupting the features.
                        failed += 1
                        log_exc = str(exc)[:90]
                        logging.debug("  %s failed: %s", a, log_exc)
                    done += 1
                    if done % 200 == 0:
                        flush()
                        logging.info("  %d/%d, %d failed, %d transfers held",
                                     done, len(pool), failed,
                                     pl.read_parquet(OUT).height if OUT.exists() else 0)

            await asyncio.gather(*(one(a) for a in pool))
            flush()

    asyncio.run(run())
    t = pl.read_parquet(TRUNC)
    df = pl.read_parquet(OUT)
    print(f"done: {t.height:,} senders fetched, "
          f"{int((~t['truncated']).sum()):,} complete, {failed} failed, "
          f"{df.height:,} transfers")


if __name__ == "__main__":
    main()

"""Ordinary Ethereum wallets, sampled the way the Tron controls were.

The first two attempts at an Ethereum control arm both failed, differently:

  - The live USDT firehose gave wallets whose median first transfer was three
    weeks old, so a pseudo-freeze drawn from the positives landed before they
    existed and every probe was dropped.
  - Counterparties already in our transfer pool were old enough, but 97% of
    them were counterparties OF FROZEN ADDRESSES - victims, downstream mules,
    cash-out points. Scoring against those measures how well we separate a
    scam collector from its own neighbours, which is not the question.

This samples USDT transfers from historical block ranges spread across the
years the frozen addresses span, and takes the addresses moving money in each
window. That is what M2 did on Tron, and it is why the Tron controls were old
enough to probe. Etherscan's startblock/endblock makes it possible here; the
Blockscout endpoint we were using cannot do it.
"""
import asyncio, json, logging, random, sys, urllib.parse
sys.path.insert(0, "src")
import polars as pl
from veridis.chain.etherscan import client as es_client, token_transfers
from veridis.chain.http import CachedClient
from veridis.dataset.ingest import TRANSFER_SCHEMA, dedupe_transfers
from veridis.config import INTERIM, USDT_ETH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)

KEY = "9QGTZYJ7CW6K4YTWWQCK3I6NHCAN32YXXJ"
ES = "https://api.etherscan.io/v2/api"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 1600
MAX_PAGES = int(sys.argv[2]) if len(sys.argv) > 2 else 8
SEED = 404
# Spread across the years the freeze list spans, so controls are not all of one
# vintage and a pseudo-freeze can sit inside any of their lives.
YEARS = [2021, 2022, 2023, 2024, 2025]

OUT = INTERIM / "eth_control3_transfers.parquet"
TRUNC = INTERIM / "eth_control3_truncated.parquet"


async def es(client, params):
    q = urllib.parse.urlencode({**params, "chainid": 1, "apikey": KEY})
    d = await client.get_json(f"{ES}?{q}", None)
    r = (d or {}).get("result")
    return r if isinstance(r, list) else []


async def block_at(client, ts):
    q = urllib.parse.urlencode({"chainid": 1, "module": "block",
                                "action": "getblocknobytime", "timestamp": ts,
                                "closest": "before", "apikey": KEY})
    d = await client.get_json(f"{ES}?{q}", None)
    try:
        return int((d or {}).get("result"))
    except (TypeError, ValueError):
        return None


def held() -> set[str]:
    out: set[str] = set()
    for n in ("eth_truncated", "eth_coverage_truncated",
              "eth_control2_truncated", "eth_control3_truncated"):
        p = INTERIM / f"{n}.parquet"
        if p.exists():
            out |= set(pl.read_parquet(p)["address"].to_list())
    return out


async def main() -> None:
    import datetime as dt
    rng = random.Random(SEED)
    scam = set(pl.read_parquet(INTERIM / "scam_addresses.parquet")["address"].to_list())
    vic = set(pl.read_parquet(INTERIM / "eth_victims.parquet")["from_address"].to_list())
    have = held()

    async with CachedClient(namespace="etherscan", rate_per_sec=4.0, concurrency=2) as c:
        seeds: list[str] = []
        for year in YEARS:
            for month in (2, 6, 10):
                ts = int(dt.datetime(year, month, 15).timestamp())
                lo = await block_at(c, ts)
                if lo is None:
                    continue
                rows = await es(c, {"module": "account", "action": "tokentx",
                                    "contractaddress": USDT_ETH, "startblock": lo,
                                    "endblock": lo + 400, "page": 1, "offset": 400,
                                    "sort": "asc"})
                for r in rows:
                    for side in ("from", "to"):
                        a = (r.get(side) or "").lower()
                        if a and a.startswith("0x"):
                            seeds.append(a)
                logging.info("  %d-%02d: block %s -> %d rows, %d addresses so far",
                             year, month, lo, len(rows), len(set(seeds)))
        pool = [a for a in dict.fromkeys(seeds) if a not in scam and a not in vic and a not in have]
        rng.shuffle(pool)
        pool = pool[:N]
        print(f"\nhistorical firehose -> {len(pool):,} ordinary wallets across {len(YEARS)} years")

    # Etherscan, not Blockscout: an ingest set to 8 req/s on Blockscout settled
    # at about 0.75 because it throttles, which made this an hour-long job.
    rows, trunc, failed, done = [], {}, 0, 0

    def flush():
        """Checkpoint. The first version wrote only at the end, so a refusal an
        hour in would have cost the whole run."""
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
        rows.clear()

    async with es_client() as c:
        sem = asyncio.Semaphore(3)

        async def one(a):
            nonlocal failed, done
            async with sem:
                try:
                    raw, cut = await token_transfers(c, a, KEY, max_pages=MAX_PAGES)
                    rows.extend(raw); trunc[a] = cut
                except Exception:
                    failed += 1
                done += 1
                if done % 200 == 0:
                    flush()
                    logging.info("  %d/%d, %d failed", done, len(pool), failed)

        await asyncio.gather(*(one(a) for a in pool))
        flush()
        df = pl.read_parquet(OUT) if OUT.exists() else pl.DataFrame(rows, schema=TRANSFER_SCHEMA)
        t = pl.read_parquet(TRUNC)
        print(f"done: {t.height:,} addresses, {int((~t['truncated']).sum()):,} complete, "
              f"{failed} failed, {df.height:,} transfers")

asyncio.run(main())

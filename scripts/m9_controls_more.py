"""M9f - more ordinary wallets, so the boundary has something to learn from.

The destination model was trained against 2,483 controls and just 117 scam
destinations. Coverage has since grown to 8,532 scam addresses, which leaves
the negative side badly outnumbered and the decision boundary under-determined.

Same sampling as M2 - addresses drawn from the live USDT transfer stream, which
is the honest population of "wallets that move stablecoins" - but written to its
own files so the original control set stays intact and the existing evaluation
remains reproducible against it.

Different RNG seed from M2, and anything M2 already took is skipped, so this
adds new wallets rather than re-fetching the same ones.
"""
import asyncio, logging, random, sys
sys.path.insert(0, "src")
import polars as pl
from veridis.chain.tron import tron_client
from veridis.chain.address import hex_to_base58
from veridis.dataset.ingest import fetch_histories
from veridis.config import INTERIM, TRONGRID_BASE, USDT_TRON
from veridis.chain.tron import PAGE

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)

N_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 2200
MAX_PAGES = int(sys.argv[2]) if len(sys.argv) > 2 else 6
SEED = 91

scam = set(pl.read_parquet(INTERIM / "scam_addresses.parquet")["address"].to_list())
victims = set(pl.read_parquet(INTERIM / "victims_all.parquet")["victim_address"].to_list())
already: set[str] = set()
for n in ("control_truncated", "control2_truncated", "scam_truncated",
          "victim_truncated", "leadtime_truncated", "coverage_truncated"):
    p = INTERIM / f"{n}.parquet"
    if p.exists():
        already |= set(pl.read_parquet(p)["address"].to_list())
print(f"excluding {len(scam):,} scam, {len(victims):,} victim, {len(already):,} already held")

vic = pl.read_parquet(INTERIM / "victims_all.parquet")
LO, HI = int(vic["first_send_at"].min()), int(vic["last_send_at"].max())


async def firehose(c, windows):
    url = f"{TRONGRID_BASE}/v1/contracts/{USDT_TRON}/events"
    rows = []
    for lo, hi in windows:
        p = await c.get_json(url, {"event_name": "Transfer", "limit": PAGE,
                                   "min_block_timestamp": lo, "max_block_timestamp": hi,
                                   "order_by": "block_timestamp,asc"})
        if p and p.get("data"):
            rows.extend(p["data"])
    return rows


async def main():
    rng = random.Random(SEED)
    windows = [(t, t + 60_000) for t in
               (rng.randint(LO, max(LO + 1, HI)) for _ in range(200))]
    async with tron_client(concurrency=4) as c:
        ev = await firehose(c, windows)
        seen = []
        for e in ev:
            r = e.get("result") or {}
            for k in ("from", "to"):
                v = r.get(k)
                if not v:
                    continue
                try:
                    a = hex_to_base58(v)
                except Exception:
                    continue
                if a not in scam and a not in victims and a not in already:
                    seen.append(a)
        seeds = list(dict.fromkeys(seen))
        rng.shuffle(seeds)
        seeds = seeds[:N_SEED]
        print(f"firehose {len(ev):,} transfers -> {len(seeds):,} new ordinary wallets")

        df, trunc = await fetch_histories(c, seeds, max_pages=MAX_PAGES,
                                          concurrency=4, label="ctrl2")
        df.write_parquet(INTERIM / "control2_transfers.parquet")
        pl.DataFrame({"address": list(trunc), "truncated": list(trunc.values())}
                     ).write_parquet(INTERIM / "control2_truncated.parquet")
        print(f"transfers={df.height:,} addresses={len(trunc):,} stats={c.stats}")

asyncio.run(main())

"""M2 - build a control population with the same data completeness as positives.

Controls are drawn as a connected subgraph, not as isolated addresses: we seed
from the live USDT transfer firehose, fetch those senders, then fetch their
frequent counterparties. That two-hop expansion means a control event's sender
*and* destination both have fetched history - matching the positives, where the
victim and the scam address are both fully fetched.

That symmetry matters. If control destinations had thinner history than scam
destinations, the model could separate the classes on data completeness alone,
which would look like signal and generalise to nothing.
"""
import asyncio, logging, sys, random
sys.path.insert(0, "src")
import polars as pl
from veridis.chain.tron import tron_client, normalise_transfers, PAGE
from veridis.chain.http import CachedClient
from veridis.dataset.ingest import fetch_histories, dedupe_transfers
from veridis.config import INTERIM, TRONGRID_BASE, USDT_TRON, SEED

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)

N_SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 400
N_EXPAND = int(sys.argv[2]) if len(sys.argv) > 2 else 700
MAX_PAGES = int(sys.argv[3]) if len(sys.argv) > 3 else 6

scam = set(pl.read_parquet(INTERIM / "scam_addresses.parquet")
           .filter(pl.col("chain") == "tron")["address"].to_list())
victims = pl.read_parquet(INTERIM / "victims_all.parquet")
victim_addrs = set(victims["victim_address"].to_list())
print(f"excluding {len(scam)} scam and {len(victim_addrs)} victim addresses")


async def sample_firehose(client: CachedClient, windows: list[tuple[int, int]]) -> list[dict]:
    """Sample ordinary USDT transfers across several time windows."""
    url = f"{TRONGRID_BASE}/v1/contracts/{USDT_TRON}/events"
    rows = []
    for lo, hi in windows:
        payload = await client.get_json(url, {
            "event_name": "Transfer", "limit": PAGE,
            "min_block_timestamp": lo, "max_block_timestamp": hi,
            "order_by": "block_timestamp,asc",
        })
        if payload and payload.get("data"):
            rows.extend(payload["data"])
    return rows


async def main():
    rng = random.Random(SEED)
    # Sample windows spanning the same period the positives cover.
    lo = int(victims["first_send_at"].min())
    hi = int(victims["last_send_at"].max())
    windows = []
    for _ in range(60):
        t = rng.randint(lo, max(lo + 1, hi))
        windows.append((t, t + 60_000))

    async with tron_client(concurrency=1, backoff_start=8.0) as c:
        ev = await sample_firehose(c, windows)
        from veridis.chain.address import hex_to_base58
        seeds = []
        for e in ev:
            r = e.get("result") or {}
            frm = r.get("from") or r.get("0")
            if not frm:
                continue
            try:
                a = hex_to_base58(frm)
            except Exception:
                continue
            if a not in scam and a not in victim_addrs:
                seeds.append(a)
        seeds = list(dict.fromkeys(seeds))
        rng.shuffle(seeds)
        seeds = seeds[:N_SEED]
        print(f"firehose sampled {len(ev)} transfers -> {len(seeds)} seed addresses")

        seed_tx, trunc_s = await fetch_histories(c, seeds, max_pages=MAX_PAGES,
                                           concurrency=1, label="ctrl-seed")
        print(f"seed transfers={seed_tx.height}")

        # Expand to frequent counterparties so both ends of a control event
        # have history, mirroring the positives.
        peers = (
            seed_tx.filter(pl.col("from_address").is_in(seeds))
            .group_by("to_address").len().sort("len", descending=True)
            .filter(~pl.col("to_address").is_in(list(scam)))
            ["to_address"].to_list()
        )
        peers = [p for p in peers if p not in victim_addrs][:N_EXPAND]
        print(f"expanding to {len(peers)} counterparties")
        peer_tx, trunc_p = await fetch_histories(c, peers, max_pages=MAX_PAGES,
                                           concurrency=1, label="ctrl-peer")

        allt = dedupe_transfers([seed_tx, peer_tx])
        tr = {**trunc_s, **trunc_p}
        pl.DataFrame({"address": list(tr), "truncated": list(tr.values())},
                     schema={"address": pl.Utf8, "truncated": pl.Boolean}
                     ).write_parquet(INTERIM / "control_truncated.parquet")
        allt.write_parquet(INTERIM / "control_transfers.parquet")
        pl.DataFrame({"address": seeds}).write_parquet(INTERIM / "control_seeds.parquet")
        pl.DataFrame({"address": peers}).write_parquet(INTERIM / "control_peers.parquet")
        print(f"control transfers={allt.height} stats={c.stats}")

asyncio.run(main())

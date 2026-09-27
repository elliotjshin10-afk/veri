"""Fetch the full Tether TRC-20 freeze list from chain events."""
import asyncio, json, sys, logging
sys.path.insert(0, "src")
from veridis.chain.tron import tron_client, blacklist_events
from veridis.config import USDT_TRON, RAW

logging.basicConfig(level=logging.INFO, format="%(message)s")

async def main():
    async with tron_client(rate_per_sec=10, concurrency=4) as c:
        rows = await blacklist_events(c, USDT_TRON, max_pages=400)
        print(f"events={len(rows)} unique={len({r['address'] for r in rows})} stats={c.stats}")
        if rows:
            import datetime as dt
            f = lambda ms: dt.datetime.utcfromtimestamp(ms/1000).date().isoformat()
            print("range:", f(rows[0]['block_time']), "->", f(rows[-1]['block_time']))
        (RAW/"tether_blacklist_tron.json").write_text(json.dumps(rows))

asyncio.run(main())

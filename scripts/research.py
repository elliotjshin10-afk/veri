"""Research any Tron wallet from the command line.

    make research ADDR=TXYZ...

Fetches the address's TRC-20 history if we have never seen it, then prints what
it does on-chain. No model score: these are checkable facts about the address.
"""
import asyncio, sys, time
sys.path.insert(0, "src")
import datetime as dt
import polars as pl
from veridis.api.live import LiveWarehouse, collection_verdict
from veridis.chain.address import is_tron_address
from veridis.config import INTERIM, PROCESSED

if len(sys.argv) < 2:
    raise SystemExit("usage: research.py <tron address>")
addr = sys.argv[1].strip()
if not is_tron_address(addr):
    raise SystemExit(f"{addr!r} is not a valid Tron address (checksum failed)")

base = pl.read_parquet(PROCESSED / "warehouse_transfers.parquet")
labels = pl.read_parquet(INTERIM / "scam_addresses.parquet").filter(pl.col("accepted"))
listed = labels.filter(pl.col("address") == addr)

indexed = set()
for name in ("scam_truncated.parquet", "victim_truncated.parquet",
             "control_truncated.parquet"):
    path = INTERIM / name
    if path.exists():
        indexed |= set(pl.read_parquet(path)["address"].to_list())
wh = LiveWarehouse(base, indexed=indexed)
known_before = wh.knows(addr)

async def main():
    t0 = time.time()
    fetch = await wh.ensure([addr])
    prof = wh.profile(addr, int(time.time() * 1000))
    print(f"\n{addr}")
    print(f"{'already indexed' if known_before else 'fetched from chain'}"
          f"  ({time.time()-t0:.1f}s, {wh.n_transfers:,} transfers in warehouse)")
    if listed.height:
        r = listed.row(0, named=True)
        when = (dt.datetime.utcfromtimestamp(r["first_reported_at"]/1000).date()
                if r["first_reported_at"] else "undated")
        print(f"\n  ON A PUBLIC LIST  source={r['sources']}  first reported {when}")
    if prof is None:
        print("\n  No TRC-20 stablecoin activity found for this address.")
        return
    verdict, notes = collection_verdict(prof)
    print(f"\n  verdict: {verdict}")
    for n in notes:
        print(f"    - {n}")
    print("\n  profile:")
    order = ["age_days","senders_all","senders_7d","senders_30d","inbound_count",
             "outbound_count","payees_all","inbound_usd","outbound_usd",
             "usd_per_sender","forward_ratio","consolidation_ratio","median_hold_secs"]
    for k in order:
        v = prof.get(k)
        if v is None: continue
        print(f"    {k:<22} {v:,.2f}" if isinstance(v,float) else f"    {k:<22} {v:,}")
    print(f"\n  verify: https://tronscan.org/#/address/{addr}\n")

asyncio.run(main())

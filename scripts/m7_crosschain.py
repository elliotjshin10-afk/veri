"""M7 - train on Tron, test on Ethereum.

The long-term claim is that this is pre-send fraud detection for irreversible
push payments in general, not a Tron trick. The way to test that is to take the
model trained on Tron, change nothing, and score Ethereum transfers with it. If
the signal transfers, it is behavioural; if it collapses, we learned chain
artifacts.

Nothing is refitted here - no retraining, no recalibration, no new threshold.
The Tron model and the Tron operating point are applied as-is.
"""
import asyncio, json, logging, sys
sys.path.insert(0, "src")
import polars as pl
from veridis.chain.evm import erc20_transfers, evm_client, BLOCKSCOUT_BASE
from veridis.chain.http import CachedClient
from veridis.dataset.ingest import TRANSFER_SCHEMA, dedupe_transfers
from veridis.config import INTERIM, MIN_VICTIM_SEND_USD, SEED, USDT_ETH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)

N_SCAM = int(sys.argv[1]) if len(sys.argv) > 1 else 470
N_VICTIMS = int(sys.argv[2]) if len(sys.argv) > 2 else 420
N_SEED = int(sys.argv[3]) if len(sys.argv) > 3 else 150
N_PEER = int(sys.argv[4]) if len(sys.argv) > 4 else 260
PAGES = 2

labels = pl.read_parquet(INTERIM / "scam_addresses.parquet").filter(
    (pl.col("chain") == "ethereum") & pl.col("accepted"))
# Prefer Tether freezes: same label generator as Tron, and unlike phishing
# lists these addresses actually move stablecoins.
frozen = labels.filter(pl.col("sources").str.contains("tether_freeze_eth"))
if frozen.height >= 100:
    labels = frozen.sort("first_reported_at", descending=True)
    print(f"ethereum labels: {labels.height} (Tether freezes, same source as Tron)")
else:
    print(f"ethereum labels: {labels.height} (mixed sources; "
          f"only {frozen.height} Tether freezes available)")


async def fetch_many(client, addrs, label, max_pages=PAGES):
    rows, trunc = [], {}
    for i, a in enumerate(addrs, 1):
        try:
            r, t = await erc20_transfers(client, a, max_pages=max_pages)
        except Exception as exc:  # noqa: BLE001
            logging.debug("fetch failed %s: %s", a, exc)
            r, t = [], False
        rows.extend(r); trunc[a] = t
        if i % 100 == 0 or i == len(addrs):
            logging.info("  %s: %d/%d, %d transfers, %s", label, i, len(addrs),
                         len(rows), client.stats)
    df = (pl.DataFrame(rows, schema=TRANSFER_SCHEMA) if rows
          else pl.DataFrame(schema=TRANSFER_SCHEMA))
    return df.unique(subset=["tx_hash", "from_address", "to_address", "block_time"]), trunc


async def main():
    async with evm_client() as c:
        scam_addrs = [a.lower() for a in labels["address"].to_list()[:N_SCAM]]
        scam_tx, scam_tr = await fetch_many(c, scam_addrs, "eth-scam")
        scam_tx.write_parquet(INTERIM / "eth_scam_transfers.parquet")
        print(f"eth scam transfers: {scam_tx.height:,}")

        # Victims: wallets that paid a labelled address. Most Ethereum labels
        # carry no report date, so there is no freeze cutoff to apply - every
        # inbound transfer counts.
        sends = scam_tx.filter(pl.col("to_address").is_in(scam_addrs))
        victims = (sends.group_by(["from_address", "to_address"])
                   .agg(pl.col("amount_usd").sum().alias("total_sent_usd"),
                        pl.len().alias("n_sends"),
                        pl.col("block_time").min().alias("first_send_at"))
                   .filter((pl.col("total_sent_usd") >= MIN_VICTIM_SEND_USD)
                           & (~pl.col("from_address").is_in(scam_addrs))))
        retail = victims.filter(
            (pl.col("n_sends") >= 2) & (pl.col("n_sends") <= 40)
            & (pl.col("total_sent_usd") <= 250_000))
        print(f"victim relationships: {victims.height:,} "
              f"(retail-plausible: {retail.height:,})")
        pool = retail if retail.height >= 60 else victims
        sel = pool.sample(fraction=1.0, shuffle=True, seed=SEED).head(N_VICTIMS)
        sel.write_parquet(INTERIM / "eth_victims.parquet")
        vaddrs = sel["from_address"].unique().to_list()
        print(f"fetching {len(vaddrs)} victim histories")
        vic_tx, vic_tr = await fetch_many(c, vaddrs, "eth-victim")
        vic_tx.write_parquet(INTERIM / "eth_victim_transfers.parquet")

        # Controls: sample the USDT firehose, then expand to counterparties so
        # both ends of a control event have fetched history, as on Tron.
        url = f"{BLOCKSCOUT_BASE}/tokens/{USDT_ETH}/transfers"
        seeds, params = [], {}
        for _ in range(14):
            payload = await c.get_json(url, params)
            if not payload:
                break
            for it in payload.get("items") or []:
                f = (it.get("from") or {}).get("hash")
                if f:
                    seeds.append(f.lower())
            nxt = payload.get("next_page_params")
            if not nxt:
                break
            params = {k: v for k, v in nxt.items() if v is not None}
        excl = set(scam_addrs) | set(vaddrs)
        seeds = [a for a in dict.fromkeys(seeds) if a not in excl][:N_SEED]
        print(f"firehose -> {len(seeds)} control seeds")
        seed_tx, seed_tr = await fetch_many(c, seeds, "eth-ctrl-seed")

        peers = (seed_tx.filter(pl.col("from_address").is_in(seeds))
                 .group_by("to_address").len().sort("len", descending=True)
                 ["to_address"].to_list())
        peers = [p for p in peers if p not in excl and p not in set(seeds)][:N_PEER]
        print(f"expanding to {len(peers)} counterparties")
        peer_tx, peer_tr = await fetch_many(c, peers, "eth-ctrl-peer")

        ctl = dedupe_transfers([seed_tx, peer_tx])
        ctl.write_parquet(INTERIM / "eth_control_transfers.parquet")
        pl.DataFrame({"address": seeds}).write_parquet(INTERIM / "eth_control_seeds.parquet")
        pl.DataFrame({"address": peers}).write_parquet(INTERIM / "eth_control_peers.parquet")
        trunc = {**scam_tr, **vic_tr, **seed_tr, **peer_tr}
        pl.DataFrame({"address": list(trunc), "truncated": list(trunc.values())},
                     schema={"address": pl.Utf8, "truncated": pl.Boolean}
                     ).write_parquet(INTERIM / "eth_truncated.parquet")
        print(f"eth control transfers: {ctl.height:,}  stats={c.stats}")
        print("\nnow run: scripts/m7_evaluate.py")

asyncio.run(main())

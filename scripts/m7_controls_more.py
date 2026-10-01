"""More ordinary Ethereum wallets, so the negative arm can carry an evaluation.

Ethereum now has 2,584 complete-history scam addresses and 195 complete-history
controls. That asymmetry is fatal twice over: a model trained against 195
negatives learns that particular sample, and an evaluation against 195 cannot
set a false-alarm threshold at all. The Tron side taught this the expensive
way - a retrained model looked like 0.969 ROC-AUC until it was tested against
controls drawn by a procedure it had not trained on, and came back 0.903.

NOT sampled from the live firehose. That was the first attempt and it failed in
a way worth recording: firehose addresses are young (median first transfer three
weeks ago) while the frozen addresses span 2017-2026, so a pseudo-freeze drawn
from the positives landed before the control existed and every probe was
dropped - 24 of 195 controls survived. The Tron controls escaped this only
because M2 happened to sample across historical windows.

So controls are drawn from counterparties already present in the Ethereum
transfer pool, oldest first: age is the binding property, because a control is
only usable if a pseudo-freeze date can sit after its first transfer.

Blockscout pages newest-first, so max_pages is generous and the truncation flag
is recorded - only complete histories are usable on this chain, and more than a
third of them come back capped.
"""
import asyncio, logging, sys
sys.path.insert(0, "src")
import polars as pl
from veridis.chain.evm import erc20_transfers, evm_client, BLOCKSCOUT_BASE
from veridis.dataset.ingest import TRANSFER_SCHEMA, dedupe_transfers
from veridis.config import INTERIM, USDT_ETH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
MAX_PAGES = int(sys.argv[2]) if len(sys.argv) > 2 else 8

OUT = INTERIM / "eth_control2_transfers.parquet"
TRUNC = INTERIM / "eth_control2_truncated.parquet"


def held() -> set[str]:
    out: set[str] = set()
    for n in ("eth_truncated", "eth_coverage_truncated", "eth_control2_truncated"):
        p = INTERIM / f"{n}.parquet"
        if p.exists():
            out |= set(pl.read_parquet(p)["address"].to_list())
    return out


async def main() -> None:
    scam = set(pl.read_parquet(INTERIM / "scam_addresses.parquet")["address"].to_list())
    vic = set(pl.read_parquet(INTERIM / "eth_victims.parquet")["from_address"].to_list())
    have = held()
    cand = pl.read_parquet(INTERIM / "eth_ctrl_candidates.parquet")["address"].to_list()
    excl = scam | vic | have
    seeds = [a for a in cand if a not in excl][:N]
    print(f"{len(seeds):,} ordinary ethereum wallets to fetch (oldest first)")
    # 0.9 req/s with no concurrency was the default and it is far below what
    # Blockscout tolerates: 12 simultaneous requests returned 12 x HTTP 200 in
    # 0.7s. At the old rate this fetch was a five-hour job for no reason.
    async with evm_client(rate_per_sec=8.0, concurrency=4) as c:

        rows, trunc, failed, done = [], {}, 0, 0
        sem = asyncio.Semaphore(4)

        async def one(a: str) -> None:
            nonlocal failed, done
            async with sem:
                try:
                    raw, cut = await erc20_transfers(c, a, max_pages=MAX_PAGES)
                    rows.extend(raw)
                    trunc[a] = cut
                except Exception:
                    failed += 1
                done += 1
                if done % 100 == 0:
                    logging.info("  %d/%d, %d failed", done, len(seeds), failed)

        await asyncio.gather(*(one(a) for a in seeds))
        df = pl.DataFrame(rows, schema=TRANSFER_SCHEMA)
        if OUT.exists():
            df = pl.concat([pl.read_parquet(OUT), df], how="vertical_relaxed")
        dedupe_transfers([df]).write_parquet(OUT)
        t = pl.DataFrame({"address": list(trunc), "truncated": list(trunc.values())})
        if TRUNC.exists():
            t = pl.concat([pl.read_parquet(TRUNC), t], how="vertical_relaxed").unique("address")
        t.write_parquet(TRUNC)
        comp = int((~t["truncated"]).sum())
        print(f"done: {t.height:,} addresses, {comp:,} with complete history, {failed} failed")

asyncio.run(main())

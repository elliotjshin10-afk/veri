"""Fetch Tether's Ethereum TRC-20... ERC-20 freeze list.

Same label generator as the Tron side - a Tether `AddedBlackList` event - which
is what makes the cross-chain test clean: the chain changes and the labelling
process does not. Using Ethereum phishing lists instead would confound the two,
and in practice those addresses drain ETH and NFTs rather than stablecoins, so
they carry almost no USDT history to learn from (470 of them yielded ~2,800
stablecoin transfers).

Blockscout's Etherscan-compatible endpoint caps at 1000 logs per call, so this
walks block ranges.
"""
import asyncio, json, logging, sys
sys.path.insert(0, "src")
from veridis.chain.http import CachedClient
from veridis.config import RAW, USDT_ETH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)

BASE = "https://eth.blockscout.com/api"
TOPIC = "0x42e160154868087d6bfdc0ca23d96a1c1cfa32f1b72ba9ba27b69b98a0d819dc"
FIRST_BLOCK = 4634748          # USDT deployment
CHUNK = 900_000
MIN_SPAN = 20_000


def _write(out: dict, gaps: list, partial: bool = False) -> None:
    rows = sorted(out.values(), key=lambda r: r["block_time"] or 0)
    (RAW / "tether_blacklist_eth.json").write_text(json.dumps(rows))
    if gaps:
        (RAW / "tether_blacklist_eth_gaps.json").write_text(json.dumps(gaps))
    if partial:
        logging.info("  checkpoint: %d addresses written", len(rows))


def _addr(data: str) -> str | None:
    h = (data or "").removeprefix("0x")
    return "0x" + h[-40:] if len(h) >= 40 else None


async def main():
    tip = int(sys.argv[1]) if len(sys.argv) > 1 else 26_100_000
    out: dict[str, dict] = {}
    gaps: list[tuple[int, int]] = []
    last_write = 0
    # The Etherscan-compatible endpoint is rate-limited harder than v2.
    async with CachedClient("blockscout", rate_per_sec=0.45, concurrency=1,
                            backoff_start=3.0, max_attempts=4) as c:
        lo, span = FIRST_BLOCK, CHUNK
        while lo < tip:
            hi = min(lo + span, tip)
            payload = await c.get_json(BASE, {
                "module": "logs", "action": "getLogs",
                "fromBlock": lo, "toBlock": hi,
                "address": USDT_ETH, "topic0": TOPIC,
            })
            res = (payload or {}).get("result") if payload else None

            # A failed request is NOT an empty range. Treating it as one drops
            # every freeze in that window without a trace, which is exactly the
            # kind of silent hole that makes a label set quietly wrong.
            if not isinstance(res, list):
                if span > MIN_SPAN:
                    span = max(MIN_SPAN, span // 4)
                    logging.warning("  request failed, narrowing to %d blocks and retrying",
                                    span)
                    await asyncio.sleep(5)
                    continue
                gaps.append((lo, hi))
                logging.error("  UNRESOLVED GAP %d-%d after narrowing to %d blocks",
                              lo, hi, span)
                lo = hi + 1
                continue

            n = len(res)
            # A full page means the range was capped, so events inside it were
            # dropped. Narrow the window and retry the SAME range - advancing
            # here would silently skip whatever did not fit.
            if n >= 1000 and span > MIN_SPAN:
                span = max(MIN_SPAN, span // 4)
                logging.info("  capped at %d, narrowing window to %d blocks", n, span)
                continue

            if isinstance(res, list):
                for e in res:
                    a = _addr(e.get("data"))
                    if not a:
                        continue
                    ts = e.get("timeStamp")
                    try:
                        ms = (int(ts, 16) if isinstance(ts, str) and ts.startswith("0x")
                              else int(ts)) * 1000
                    except (TypeError, ValueError):
                        ms = None
                    out.setdefault(a.lower(), {
                        "address": a.lower(), "block_time": ms,
                        "tx_hash": e.get("transactionHash")})
            logging.info("blocks %d-%d -> %d events (total %d)", lo, hi, n, len(out))
            lo = hi + 1
            # Checkpoint as we go. A long walk against a throttled endpoint
            # will be interrupted sooner or later, and writing only at the end
            # means an interruption throws away every request already paid for.
            if len(out) - last_write >= 100:
                _write(out, gaps, partial=True)
                last_write = len(out)
            # Widen again once a window comes back comfortably under the cap.
            if n < 300:
                span = min(CHUNK, span * 2)

    rows = sorted(out.values(), key=lambda r: r["block_time"] or 0)
    if gaps:
        logging.error("%d unresolved block ranges - the label set is INCOMPLETE: %s",
                      len(gaps), gaps[:5])
    _write(out, gaps)
    import datetime as dt
    if rows:
        f = lambda ms: dt.datetime.utcfromtimestamp(ms / 1000).date().isoformat()
        dated = [r for r in rows if r["block_time"]]
        print(f"\n{len(rows)} unique frozen Ethereum addresses")
        if dated:
            print(f"range: {f(dated[0]['block_time'])} -> {f(dated[-1]['block_time'])}")

asyncio.run(main())

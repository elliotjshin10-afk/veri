"""Ethereum ERC-20 history via Blockscout.

Cross-chain validation needs full pre-loss history for labelled Ethereum
addresses, which is archive data. Free public JSON-RPC endpoints refuse
archive-range `eth_getLogs` ("Archive requests require a personal token"), and
Etherscan, BigQuery and Dune all require an account - which is why this test sat
unrun.

Blockscout's public v2 API serves an address's token transfers with no key and
tolerates ~1 req/s, which is enough. Same normalised schema as the Tron client,
so the feature layer does not know or care which chain a transfer came from -
the whole point of the exercise.
"""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any

from veridis.chain.http import CachedClient
from veridis.config import STABLE_SYMBOLS

log = logging.getLogger(__name__)

BLOCKSCOUT_BASE = "https://eth.blockscout.com/api/v2"


def evm_client(rate_per_sec: float = 0.9, concurrency: int = 1) -> CachedClient:
    return CachedClient(
        "blockscout", rate_per_sec=rate_per_sec, concurrency=concurrency,
        backoff_start=4.0,
        headers={"accept": "application/json"},
    )


def available() -> bool:
    """No credentials required."""
    return True


def _epoch_ms(ts: str) -> int | None:
    try:
        return int(dt.datetime.fromisoformat(
            ts.replace("Z", "+00:00")).timestamp() * 1000)
    except (ValueError, AttributeError):
        return None


async def erc20_transfers(
    client: CachedClient, address: str, max_pages: int = 4
) -> tuple[list[dict], bool]:
    """Token transfers for `address`, newest-first, in the canonical schema.

    Blockscout paginates newest-first, the opposite of the Tron client. That is
    recorded rather than reversed: `truncated` means we hold the address's
    *recent* history and not its oldest, so the coverage guard that drops
    events outside fetched history still applies, just from the other end.
    """
    url = f"{BLOCKSCOUT_BASE}/addresses/{address}/token-transfers"
    params: dict[str, Any] = {"type": "ERC-20"}
    rows: list[dict] = []
    for _page in range(max_pages):
        payload = await client.get_json(url, params)
        if not payload or not isinstance(payload, dict):
            break
        items = payload.get("items") or []
        rows.extend(items)
        nxt = payload.get("next_page_params")
        if not nxt:
            return _normalise(rows, address), False
        params = {"type": "ERC-20", **{k: v for k, v in nxt.items() if v is not None}}
    return _normalise(rows, address), True


def _normalise(rows: list[dict], address: str) -> list[dict]:
    out = []
    for r in rows:
        token = r.get("token") or {}
        symbol = (token.get("symbol") or "").upper()
        if symbol not in STABLE_SYMBOLS:
            continue
        total = r.get("total") or {}
        try:
            decimals = int(total.get("decimals") or token.get("decimals") or 18)
            value = int(total.get("value"))
        except (TypeError, ValueError):
            continue
        frm = (r.get("from") or {}).get("hash")
        to = (r.get("to") or {}).get("hash")
        ts = _epoch_ms(r.get("timestamp") or "")
        if not frm or not to or ts is None:
            continue
        out.append({
            "chain": "ethereum",
            "tx_hash": r.get("transaction_hash") or r.get("tx_hash"),
            "from_address": frm.lower(),
            "to_address": to.lower(),
            "amount_usd": value / (10 ** decimals),
            "asset": symbol,
            "block_time": ts,
        })
    return out

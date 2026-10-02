"""Etherscan V2 reader, used wherever Blockscout is too slow or cannot sort.

Blockscout needs no key and was the right default, but it throttles hard - an
ingest configured for 8 req/s settled at about 0.75 - and it cannot page
ascending, so a busy address has no recoverable age. Etherscan does both, and
one key covers Ethereum, Arbitrum and Polygon on the free tier.

The two sources were compared on the data the model trains on: 150 transfers
across six addresses, zero disagreement, once the feature layer's own
`amount_usd > 0` filter is applied to each. The rows that differed before that
filter were zero-value USDT the feature SQL drops anyway.
"""
from __future__ import annotations

import asyncio
import logging
import urllib.parse
from typing import Any

from veridis.chain.http import CachedClient
from veridis.config import STABLE_SYMBOLS

log = logging.getLogger(__name__)

BASE = "https://api.etherscan.io/v2/api"
CHAINS = {"ethereum": 1, "arbitrum": 42161, "polygon": 137}
# 1,000 is the per-request maximum Etherscan honours; asking for more returns
# 1,000 anyway. Five times fewer requests against a 3/sec cap.
PAGE = 1000


def is_refusal(payload: Any) -> bool:
    """True when a 200 response is really a rate-limit refusal.

    Etherscan does not use 429 for this. It answers HTTP 200 with
    {"status": "0", "result": "Max calls per sec rate limit reached (3/sec)"},
    which is why this has to be matched on the body.
    """
    if not isinstance(payload, dict):
        return False
    text = str(payload.get("result") or payload.get("message") or "")
    low = text.lower()
    return "rate limit" in low or "max calls" in low or "too many" in low


def client(rate_per_sec: float = 2.5, concurrency: int = 1) -> CachedClient:
    """The free tier is 3 req/s, not the 5 the docs advertise - the refusal body
    states the real figure. An earlier default of 4.5/s with concurrency 3 was
    refused constantly; 2.5/s serial leaves margin. Sequential matters as much
    as the rate: three concurrent workers each pacing themselves still put three
    requests into the same second.
    """
    return CachedClient(namespace="etherscan", rate_per_sec=rate_per_sec,
                        concurrency=concurrency, is_refusal=is_refusal, burst=1)


def _url(chain: str, params: dict[str, Any], key: str) -> str:
    q = urllib.parse.urlencode({**params, "chainid": CHAINS[chain], "apikey": key})
    return f"{BASE}?{q}"


def _is_empty(payload: Any) -> bool:
    """Etherscan's documented empty-history answer."""
    text = str((payload or {}).get("result") or (payload or {}).get("message") or "")
    return "no transactions found" in text.lower()


def normalise(rows: list[dict], chain: str = "ethereum") -> list[dict]:
    """Same shape and same filters as the Blockscout normaliser.

    The symbol match is exact deliberately: Ethereum carries impersonation
    tokens named with a Cyrillic S, with mathematical-bold characters, and with
    whole sentences ('Visit ... to claim rewards'). A case-folded or fuzzy match
    would feed the model counterfeit USDT.
    """
    out = []
    for r in rows:
        sym = (r.get("tokenSymbol") or "").upper()
        if sym not in STABLE_SYMBOLS:
            continue
        try:
            dec = int(r.get("tokenDecimal") or 18)
            val = int(r["value"]) / (10 ** dec)
            ts = int(r["timeStamp"]) * 1000
        except (TypeError, ValueError, KeyError):
            continue
        frm, to = (r.get("from") or "").lower(), (r.get("to") or "").lower()
        if not frm or not to:
            continue
        out.append({"chain": chain, "tx_hash": r.get("hash"),
                    "from_address": frm, "to_address": to,
                    "amount_usd": val, "asset": sym, "block_time": ts})
    return out


def normalise_native(rows: list[dict], address: str, price, chain: str = "ethereum",
                     internal: bool = False) -> list[dict]:
    """Native ETH movements, shaped exactly like a stablecoin transfer.

    Two things differ from the token path and both matter.

    A value, not a dollar. USDT is a dollar; ETH is not, so every row needs
    ETH/USD AS OF THAT TRANSFER - see veridis.chain.prices for why the day's own
    close is the wrong number. A row older than the price series is dropped
    rather than valued at zero: a scam collector that took 40 ETH is not a
    collector that took nothing, and feeding the model a zero would say exactly
    that.

    Most rows are not payments. An external transaction list is mostly contract
    calls carrying no value, and a failed transaction moved nothing at all.
    Both are dropped, or the feature layer would count a token approval as
    somebody paying this address.
    """
    out = []
    for r in rows:
        try:
            wei = int(r.get("value") or 0)
        except (TypeError, ValueError):
            continue
        if wei <= 0:
            continue
        # A reverted call moved no ether. Internal rows report this as
        # `isError`; external rows carry `isError` and `txreceipt_status`.
        if str(r.get("isError") or "0") != "0":
            continue
        if not internal and str(r.get("txreceipt_status") or "1") == "0":
            continue
        try:
            ts = int(r["timeStamp"]) * 1000
        except (TypeError, ValueError, KeyError):
            continue
        usd_per_eth = price.at(ts)
        if usd_per_eth is None:
            continue
        frm = (r.get("from") or "").lower()
        to = (r.get("to") or "").lower()
        if not frm or not to:
            continue
        out.append({"chain": chain, "tx_hash": r.get("hash"),
                    "from_address": frm, "to_address": to,
                    "amount_usd": (wei / 1e18) * usd_per_eth,
                    "asset": "ETH", "block_time": ts})
    return out


async def native_transfers(c: CachedClient, address: str, key: str, price, *,
                           chain: str = "ethereum",
                           max_pages: int = 8) -> tuple[list[dict], bool]:
    """Native ETH in and out of `address`, newest first, plus truncation.

    Two endpoints, because ether moves two ways and reading only the first
    misses most of what an exchange or a contract-mediated collector does:
    `txlist` is transactions the address sent or received directly, and
    `txlistinternal` is ether moved by a contract on someone's behalf.
    """
    rows: list[dict] = []
    hit_cap = False
    for action, internal in (("txlist", False), ("txlistinternal", True)):
        for page in range(1, max_pages + 1):
            payload = await c.get_json(_url(chain, {
                "module": "account", "action": action, "address": address,
                "startblock": 0, "endblock": 99999999,
                "page": page, "offset": PAGE, "sort": "desc"}, key), None)
            got = (payload or {}).get("result")
            if not isinstance(got, list):
                if _is_empty(payload):
                    break
                raise RuntimeError(
                    f"etherscan refused {action} page {page} for {address}: "
                    f"{str((payload or {}).get('result'))[:120]!r}"
                )
            rows.extend(normalise_native(got, address, price, chain, internal))
            if len(got) < PAGE:
                break
            if page == max_pages:
                hit_cap = True
    return rows, hit_cap


async def token_transfers(c: CachedClient, address: str, key: str, *,
                          chain: str = "ethereum",
                          max_pages: int = 8) -> tuple[list[dict], bool]:
    """Newest-first history, plus whether the page budget was exhausted."""
    rows: list[dict] = []
    hit_cap = True
    for page in range(1, max_pages + 1):
        payload = await c.get_json(_url(chain, {
            "module": "account", "action": "tokentx", "address": address,
            "page": page, "offset": PAGE, "sort": "desc"}, key), None)
        got = (payload or {}).get("result")
        if not isinstance(got, list):
            # Only a documented empty history may come back as empty. Any
            # other non-list result is a failure, and must not be mistaken for
            # the end of the list: treating a refusal as end-of-history is how
            # 932 cut-short histories were once recorded as complete, which
            # corrupts every lifetime feature built on them.
            if _is_empty(payload):
                hit_cap = False
                break
            raise RuntimeError(
                f"etherscan refused page {page} for {address}: "
                f"{str((payload or {}).get('result'))[:120]!r}"
            )
        rows.extend(got)
        if len(got) < PAGE:
            hit_cap = False
            break
    return normalise(rows, chain), hit_cap

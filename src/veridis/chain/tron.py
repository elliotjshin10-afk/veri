"""TronGrid client: TRC-20 transfer history, account metadata, contract events."""
from __future__ import annotations

import logging
import os
from typing import Any, Iterable

from veridis.chain.address import hex_to_base58, is_tron_address
from veridis.chain.http import CachedClient
from veridis.config import STABLE_SYMBOLS, TRONGRID_API_KEY, TRONGRID_BASE

log = logging.getLogger(__name__)

PAGE = 200  # TronGrid hard maximum

# Sustainable request rates, measured rather than taken from the docs.
KEYLESS_RATE = 0.85   # zero retries at this rate; above it the quota depletes
KEYED_RATE = 8.0      # 12/12 with no delay at all; kept well under the ceiling


def tron_client(
    rate_per_sec: float | None = None,
    concurrency: int = 1,
    backoff_start: float = 1.0,
    offline: bool = False,
) -> CachedClient:
    """TronGrid client.

    Without an API key TronGrid enforces a depleting quota, not just a
    per-second rate: after sustained use even 0.5 req/s returns mostly 429.
    Measured with a key, 12 consecutive requests with no delay all succeeded,
    so the default rate lifts substantially when one is present - that single
    change is the difference between an overnight ingest and a coffee break.

    TRON_RATE still overrides, so a run can be slowed without editing code.
    """
    if rate_per_sec is None:
        default = KEYED_RATE if TRONGRID_API_KEY else KEYLESS_RATE
        rate_per_sec = float(os.getenv("TRON_RATE", str(default)))
    headers = {"TRON-PRO-API-KEY": TRONGRID_API_KEY} if TRONGRID_API_KEY else {}
    return CachedClient(
        "trongrid", rate_per_sec=rate_per_sec, concurrency=concurrency,
        headers=headers, backoff_start=backoff_start, offline=offline,
    )


async def _paginate(
    client: CachedClient,
    url: str,
    params: dict[str, Any],
    max_pages: int,
) -> tuple[list[dict], bool]:
    """Walk TronGrid's fingerprint pagination. Returns (rows, truncated)."""
    rows: list[dict] = []
    page_params = dict(params)
    for page in range(max_pages):
        payload = await client.get_json(url, page_params)
        if not payload or not payload.get("success"):
            break
        data = payload.get("data") or []
        rows.extend(data)
        fingerprint = (payload.get("meta") or {}).get("fingerprint")
        if not fingerprint or len(data) < page_params.get("limit", PAGE):
            return rows, False
        page_params = dict(params)
        page_params["fingerprint"] = fingerprint
    return rows, True


async def trc20_transfers(
    client: CachedClient,
    address: str,
    max_pages: int = 25,
    min_timestamp: int | None = None,
    max_timestamp: int | None = None,
) -> tuple[list[dict], bool]:
    """All TRC-20 transfers touching `address`, oldest-first.

    Returns (transfers, truncated). `truncated` matters: a capped history makes
    account-age and lifetime-volume features unreliable, so the flag is carried
    into the feature table rather than silently ignored.
    """
    url = f"{TRONGRID_BASE}/v1/accounts/{address}/transactions/trc20"
    params: dict[str, Any] = {"limit": PAGE, "order_by": "block_timestamp,asc"}
    if min_timestamp is not None:
        params["min_timestamp"] = min_timestamp
    if max_timestamp is not None:
        params["max_timestamp"] = max_timestamp
    rows, truncated = await _paginate(client, url, params, max_pages)
    return [r for r in rows if _usable(r)], truncated


def _usable(row: dict) -> bool:
    if row.get("type") != "Transfer":
        return False
    info = row.get("token_info") or {}
    if not info.get("symbol"):
        return False
    frm, to = row.get("from"), row.get("to")
    return is_tron_address(frm or "") and is_tron_address(to or "")


def to_usd(row: dict) -> float | None:
    """Stablecoin transfers only; we deliberately avoid price oracles."""
    info = row.get("token_info") or {}
    symbol = (info.get("symbol") or "").upper()
    if symbol not in STABLE_SYMBOLS:
        return None
    try:
        decimals = int(info.get("decimals", 6))
        return int(row["value"]) / (10**decimals)
    except (KeyError, ValueError, TypeError):
        return None


def normalise_transfers(rows: Iterable[dict], chain: str = "tron") -> list[dict]:
    """Flatten TronGrid rows into the canonical transfer schema."""
    out = []
    for r in rows:
        usd = to_usd(r)
        if usd is None:
            continue
        out.append(
            {
                "chain": chain,
                "tx_hash": r.get("transaction_id"),
                "from_address": r["from"],
                "to_address": r["to"],
                "amount_usd": usd,
                "asset": (r.get("token_info") or {}).get("symbol", "").upper(),
                "block_time": int(r["block_timestamp"]),
            }
        )
    return out


async def account_info(client: CachedClient, address: str) -> dict | None:
    url = f"{TRONGRID_BASE}/v1/accounts/{address}"
    payload = await client.get_json(url)
    if not payload or not payload.get("data"):
        return None
    return payload["data"][0]


async def blacklist_events(
    client: CachedClient,
    contract: str,
    event_name: str = "AddedBlackList",
    max_pages: int = 200,
) -> list[dict]:
    """Tether freeze events, oldest-first, decoded to base58 addresses.

    These are the strongest public Tron fraud label available: an on-chain,
    timestamped action Tether takes on law-enforcement request. The timestamp
    doubles as `first_reported_at` for the temporal split.
    """
    url = f"{TRONGRID_BASE}/v1/contracts/{contract}/events"
    params = {
        "event_name": event_name,
        "limit": PAGE,
        "order_by": "block_timestamp,asc",
    }
    rows, _ = await _paginate(client, url, params, max_pages)
    out = []
    for r in rows:
        raw = (r.get("result") or {}).get("_user") or (r.get("result") or {}).get("0")
        if not raw:
            continue
        try:
            addr = hex_to_base58(raw)
        except ValueError:
            continue
        out.append(
            {
                "address": addr,
                "block_time": int(r["block_timestamp"]),
                "tx_hash": r.get("transaction_id") or r.get("transaction"),
            }
        )
    return out

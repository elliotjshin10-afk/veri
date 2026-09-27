"""Public scam-label sources, each emitting reports with explicit provenance.

Every record carries the source it came from and how strong that source is, so
label quality can be varied in evaluation rather than assumed. No source here
is licensed or proprietary; the whole point is that free lists are late.
"""
from __future__ import annotations

import csv
import io
import json
import logging
import re
from dataclasses import dataclass, asdict
from typing import Iterator

import httpx

from veridis.chain.address import is_evm_address, is_tron_address
from veridis.config import RAW

log = logging.getLogger(__name__)

TRON_RE = re.compile(r"T[1-9A-HJ-NP-Za-km-z]{33}")
EVM_RE = re.compile(r"0x[0-9a-fA-F]{40}")


@dataclass(frozen=True)
class ScamReport:
    address: str
    chain: str
    source: str
    reported_at: int | None  # epoch ms; None when the source is undated
    category: str
    # How much this source counts toward corroboration. On-chain freezes and
    # sanctions designations are authoritative; scraped community lists are not.
    weight: float


def _fetch_text(url: str, cache_name: str) -> str | None:
    path = RAW / cache_name
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")
    try:
        r = httpx.get(url, timeout=60, follow_redirects=True)
        r.raise_for_status()
    except Exception as exc:
        log.warning("source fetch failed %s: %s", url, exc)
        return None
    path.write_text(r.text, encoding="utf-8")
    return r.text


def tether_freezes() -> Iterator[ScamReport]:
    """Tether's on-chain TRC-20 freeze list (fetched separately, on-chain)."""
    path = RAW / "tether_blacklist_tron.json"
    if not path.exists():
        log.warning("tether blacklist not fetched yet")
        return
    for row in json.loads(path.read_text()):
        if is_tron_address(row["address"]):
            yield ScamReport(
                address=row["address"],
                chain="tron",
                source="tether_freeze",
                reported_at=row["block_time"],
                category="frozen_by_issuer",
                weight=1.0,
            )


def tether_freezes_eth() -> Iterator[ScamReport]:
    """Tether's on-chain ERC-20 freeze list.

    The same label generator as the Tron side. That is what makes the
    cross-chain test clean: the chain changes and the labelling process does
    not. Ethereum phishing lists would confound the two, and in practice those
    addresses drain ETH and NFTs rather than stablecoins - 470 of them yielded
    roughly 2,800 stablecoin transfers between them, far too thin to learn from.
    """
    path = RAW / "tether_blacklist_eth.json"
    if not path.exists():
        log.warning("ethereum tether blacklist not fetched yet")
        return
    for row in json.loads(path.read_text()):
        addr = (row.get("address") or "").lower()
        if is_evm_address(addr):
            yield ScamReport(
                address=addr, chain="ethereum", source="tether_freeze_eth",
                reported_at=row.get("block_time"), category="frozen_by_issuer",
                weight=1.0,
            )


def ofac_sdn() -> Iterator[ScamReport]:
    """OFAC SDN digital-currency addresses. Authoritative but small and slow."""
    text = _fetch_text("https://www.treasury.gov/ofac/downloads/sdn.csv", "ofac_sdn.csv")
    if not text:
        return
    seen: set[tuple[str, str]] = set()
    for m in re.finditer(
        r"Digital Currency Address - (\w+)\s+([A-Za-z0-9]{25,60})", text
    ):
        ticker, addr = m.group(1), m.group(2)
        if is_tron_address(addr):
            chain = "tron"
        elif is_evm_address(addr):
            chain = "ethereum"
        else:
            continue
        key = (addr, chain)
        if key in seen:
            continue
        seen.add(key)
        yield ScamReport(addr, chain, "ofac_sdn", None, "sanctioned", 1.0)


def cryptoscamdb() -> Iterator[ScamReport]:
    text = _fetch_text(
        "https://raw.githubusercontent.com/CryptoScamDB/blacklist/master/data/urls.yaml",
        "cryptoscamdb_urls.yaml",
    )
    if not text:
        return
    for addr in set(TRON_RE.findall(text)):
        if is_tron_address(addr):
            yield ScamReport(addr, "tron", "cryptoscamdb", None, "reported_scam", 0.5)
    for addr in set(EVM_RE.findall(text)):
        yield ScamReport(
            addr.lower(), "ethereum", "cryptoscamdb", None, "reported_scam", 0.5
        )


def scamsniffer() -> Iterator[ScamReport]:
    text = _fetch_text(
        "https://raw.githubusercontent.com/scamsniffer/scam-database/main/blacklist/address.json",
        "scamsniffer_addresses.json",
    )
    if not text:
        return
    try:
        rows = json.loads(text)
    except ValueError:
        return
    for addr in rows:
        if isinstance(addr, str) and is_evm_address(addr):
            yield ScamReport(
                addr.lower(), "ethereum", "scamsniffer", None, "phishing", 0.5
            )


def mew_darklist() -> Iterator[ScamReport]:
    text = _fetch_text(
        "https://raw.githubusercontent.com/MyEtherWallet/ethereum-lists/master/src/addresses/addresses-darklist.json",
        "mew_darklist.json",
    )
    if not text:
        return
    try:
        rows = json.loads(text)
    except ValueError:
        return
    for row in rows:
        addr = (row.get("address") or "").lower()
        if is_evm_address(addr):
            yield ScamReport(
                addr,
                "ethereum",
                "mew_darklist",
                None,
                row.get("comment", "")[:60] or "reported_scam",
                0.5,
            )


SOURCES = {
    "tether_freeze": tether_freezes,
    "tether_freeze_eth": tether_freezes_eth,
    "ofac_sdn": ofac_sdn,
    "cryptoscamdb": cryptoscamdb,
    "scamsniffer": scamsniffer,
    "mew_darklist": mew_darklist,
}


def collect_all() -> list[dict]:
    out: list[dict] = []
    for name, fn in SOURCES.items():
        rows = [asdict(r) for r in fn()]
        log.info("source %-14s -> %d reports", name, len(rows))
        out.extend(rows)
    return out

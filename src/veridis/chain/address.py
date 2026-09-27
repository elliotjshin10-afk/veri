"""Tron address encoding.

TronGrid returns addresses in two shapes depending on the endpoint:
  * transfer endpoints return base58check ("T...")
  * event logs return bare 20-byte hex ("0x6f56...") with no 0x41 network prefix

Everything downstream keys on the base58 form, so normalise at the edge.
"""
from __future__ import annotations

import hashlib

import base58

TRON_MAINNET_PREFIX = 0x41


def hex_to_base58(addr_hex: str) -> str:
    """Convert a Tron hex address to base58check. Accepts 20- or 21-byte hex."""
    h = addr_hex.lower().removeprefix("0x")
    if len(h) == 40:  # bare 20-byte word from an event log
        h = f"{TRON_MAINNET_PREFIX:02x}" + h
    if len(h) != 42:
        raise ValueError(f"unexpected tron hex address length: {addr_hex!r}")
    payload = bytes.fromhex(h)
    checksum = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    return base58.b58encode(payload + checksum).decode()


def base58_to_hex(addr_b58: str) -> str:
    """Convert base58check back to 21-byte hex (0x41-prefixed, no 0x)."""
    raw = base58.b58decode(addr_b58)
    payload, checksum = raw[:-4], raw[-4:]
    expect = hashlib.sha256(hashlib.sha256(payload).digest()).digest()[:4]
    if checksum != expect:
        raise ValueError(f"bad tron address checksum: {addr_b58!r}")
    return payload.hex()


def is_tron_address(value: str) -> bool:
    if not isinstance(value, str) or not value.startswith("T") or len(value) != 34:
        return False
    try:
        base58_to_hex(value)
    except Exception:
        return False
    return True


def is_evm_address(value: str) -> bool:
    if not isinstance(value, str) or not value.lower().startswith("0x"):
        return False
    body = value[2:]
    return len(body) == 40 and all(c in "0123456789abcdefABCDEF" for c in body)


def normalise(value: str, chain: str) -> str:
    """Canonical on-disk form for an address on a given chain."""
    if chain == "tron":
        v = value.strip()
        if v.startswith("0x") or (len(v) in (40, 42) and not v.startswith("T")):
            return hex_to_base58(v)
        return v
    return value.strip().lower()

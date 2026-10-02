"""Central configuration: paths, chain constants, and pipeline parameters."""
from __future__ import annotations

import os
from pathlib import Path

from veridis.env import load as _load_env

_load_env()  # .env before anything reads the environment

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"
CACHE = DATA / "cache"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
REPORTS = ROOT / "reports"
DEMO = ROOT / "demo"
SITE = ROOT / "site"

for _p in (RAW, CACHE, INTERIM, PROCESSED, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- chain infra
TRONGRID_BASE = "https://api.trongrid.io"
TRONGRID_API_KEY = os.getenv("TRONGRID_API_KEY")  # optional; raises free-tier limits
ETHERSCAN_BASE = "https://api.etherscan.io/v2/api"
ETHERSCAN_API_KEY = os.getenv("ETHERSCAN_API_KEY")

# TRC-20 USDT (Tether) on Tron. Also the contract whose blacklist we mine for labels.
USDT_TRON = "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t"
USDT_TRON_DECIMALS = 6
# ERC-20 USDT on Ethereum, for cross-chain validation (M7).
USDT_ETH = "0xdac17f958d2ee523a2206206994597c13d831ec7"
USDT_ETH_DECIMALS = 6

# Stablecoins we treat as ~$1. Keeps us off price oracles entirely, which is a
# deliberate simplification: the product scores stablecoin sends.
STABLE_SYMBOLS = {"USDT", "USDC", "TUSD", "USDD", "DAI", "FDUSD", "PYUSD"}

# The real contracts, because a symbol is not an identity.
#
# Anyone can deploy a token whose symbol is exactly "USDT" - not a Cyrillic
# lookalike, the actual ASCII string - and send it to whoever they like. Matching
# on the symbol let 996 such transfers into the Ethereum warehouse, one of them
# claiming to move $9e39, which is more money than has ever existed. All 368
# addresses that received one would clear the $100m institutional guard on that
# fake inflow alone, so the page would call a scam collection point "an exchange,
# bridge or trading desk". That is not noise, it is an evasion vector: mint a
# fake USDT, send yourself a trillion of it, and the risk check vouches for you.
#
# Each address below was verified against Etherscan - symbol and decimals both.
STABLE_CONTRACTS_ETH = {
    "0xdac17f958d2ee523a2206206994597c13d831ec7": "USDT",
    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48": "USDC",
    "0x6b175474e89094c44da98b954eedeac495271d0f": "DAI",
    "0x0000000000085d4780b73119b644ae5ecd22b376": "TUSD",
    "0x6c3ea9036406852006290770bedfcaba0e23a0e8": "PYUSD",
    "0xc5f0f7b66764f6ec8c8dff7ba683102295e16409": "FDUSD",
}

# ------------------------------------------------------------ pipeline params
# Corroboration: an address needs this much independent support to be a positive.
MIN_CORROBORATION = 2
# Victim = an address that sent >= this much to a confirmed scam address.
MIN_VICTIM_SEND_USD = 50.0
# Controls sampled per positive victim, matched on behavioural strata.
CONTROL_RATIO = 10
# Feature windows (days) for trailing aggregates.
TRAILING_WINDOWS = (1, 7, 30)
# Operating point: the FPR a wallet will tolerate on a blocking interstitial.
TARGET_FPR = 0.01

# Temporal split: scams whose label timestamp precedes this go to train.
# Set at runtime from the label distribution; see dataset.split.
SPLIT_QUANTILE = 0.70

SEED = 17

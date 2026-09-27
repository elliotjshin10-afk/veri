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

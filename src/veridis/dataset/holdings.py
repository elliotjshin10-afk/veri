"""Everything we have actually fetched, in one place.

Three scripts used to carry their own hardcoded list of which parquets count as
"our data", and they had drifted: the address index was built from three files
while the ingest had since written two more, so thousands of fetched addresses
were scored by nobody and answered no lookups. Coverage is the product - an
address in the index answers instantly, offline, and in a sandboxed preview -
so the list of what we hold belongs in one place that every consumer reads.

WHO SHOULD USE THIS, AND WHO SHOULD NOT. Anything scoring, indexing or
backtesting against current data reads these functions - three scripts kept
their own lists and each one distorted a published number before it was found
(m9_roc lost 3,000 controls, build_pit_model shipped an index scored by the
previous model, check_parity compared the browser against a feature layer fed
different transfers).

Two kinds of caller are right to read a specific file instead:

  * Ingest steps that mean one dataset. m1b_victims derives victims FROM scam
    transfers; handing it everything would change what it computes.
  * check_pair_parity, which compares against events_features.parquet. Those
    events were computed from warehouse_transfers, so the browser must be given
    that same warehouse or the comparison is meaningless. Widening it there
    would break the check rather than fix it.

The test is whether the script is asking "what do we hold now" or "what was
this artefact built from".

`indexed` means an address whose OWN history we fetched, not one glimpsed as
somebody else's counterparty. That distinction is load-bearing: an address seen
once as a peer looks, from our data, like it has a single transfer, and a
profile built from it would confidently describe an exchange as a wallet that
forwards 100% of its money to one place.
"""
from __future__ import annotations

import polars as pl

from veridis.config import INTERIM, PROCESSED

# Transfer histories, in the order they were ingested.
TRANSFER_SETS = (
    (PROCESSED, "warehouse_transfers"),     # scam + victim + control, from M3
    (INTERIM, "leadtime_transfers"),        # every recently frozen address
    (INTERIM, "coverage_transfers"),        # the rest of the Tron freeze list
    (INTERIM, "control2_transfers"),        # a second, independently sampled control set
    # payout_transfers.parquet is deliberately NOT here. The consolidation
    # wallets were fetched so the payout fan-in could be computed from real
    # data, and putting them in the warehouse does compute it correctly - but it
    # also hands every OTHER feature a lopsided helping of new legs, because a
    # destination gains history only where it happens to touch a payout wallet.
    # Measured: the same 17 features on the same events drop from 24.4% recall
    # at a 1% false-alarm budget to 14.5%, and ROC-AUC from 0.9209 to 0.9008.
    # The fan-in is computed from that file on its own instead, in
    # scripts/m13_payout_ingest.py, and joined to the matrix as two columns.
)

# One row per address whose own history was fetched, written beside each ingest.
TRUNCATED_SETS = (
    "scam_truncated", "victim_truncated", "control_truncated",
    "leadtime_truncated", "coverage_truncated", "control2_truncated",
)


# The Ethereum side, kept here for the same reason as the Tron side: the list
# was already copied into m7_pit_model and then into a second script, which is
# how three Tron scripts came to disagree about what we hold. One list.
ETH_TRANSFER_SETS = (
    "eth_scam_transfers",         # frozen addresses, first pass
    "eth_victim_transfers",       # their payers, first pass
    "eth_control_transfers",      # the first control attempt
    "eth_coverage_transfers",     # the rest of the Ethereum freeze list
    "eth_control2_transfers",     # counterparties already in the pool
    "eth_control3_transfers",     # independently sampled ordinary wallets
    "eth_sender_transfers",       # senders on both arms, for the pair model
)

ETH_TRUNCATED_SETS = (
    "eth_truncated", "eth_coverage_truncated", "eth_control2_truncated",
    "eth_control3_truncated", "eth_sender_truncated",
)

# The control arm the headline is measured against: ordinary wallets drawn from
# historical USDT block windows with no knowledge of any scam. The other control
# sets are counterparties of frozen addresses - victims, mules, cash-out points -
# and scoring against those measures how well we separate a collector from its
# own neighbours, which is not the question the product asks.
ETH_INDEPENDENT_SET = "eth_control3_truncated"


# A single real USDT transfer has never approached this. Tether's entire supply
# is on the order of $140bn, and the largest genuine transfers are around $1bn.
COUNTERFEIT_USD = 1e10


def eth_transfers() -> pl.DataFrame:
    """Every Ethereum transfer we hold, deduplicated, counterfeits removed.

    The ingest matched tokens on their SYMBOL, and anyone can deploy a contract
    whose symbol is the real ASCII "USDT". 996 such transfers reached the
    warehouse, one claiming to move $9e39. Both normalisers now match on the
    contract address instead, so nothing new arrives this way - but the rows
    already stored carry no contract address to filter on, and re-deriving them
    means re-reading every cached payload.

    So they are dropped here by amount, which is crude and is stated rather than
    hidden: a handful of genuine whale transfers may go with them. That costs a
    little recall on addresses nobody mistakes for retail; keeping the fakes
    costs the institutional guard, which 368 addresses would have cleared on
    counterfeit inflow alone.
    """
    frames = []
    for name in ETH_TRANSFER_SETS:
        p = INTERIM / f"{name}.parquet"
        if p.exists():
            df = pl.read_parquet(p)
            frames.append(df if not frames else df.select(frames[0].columns))
    if not frames:
        raise SystemExit("no Ethereum transfer data found - run the ingest first")
    out = pl.concat(frames, how="vertical_relaxed").unique(
        subset=["tx_hash", "to_address"])
    return out.filter(pl.col("amount_usd") < COUNTERFEIT_USD)


def eth_complete() -> set[str]:
    """Addresses whose Ethereum history we fetched IN FULL.

    Completeness is not a nicety here: every lifetime feature - age, payer
    count, inbound total - is wrong on a partial history, and the models are
    fitted only on complete ones. A truncated history must therefore be
    excluded, not scored, which is only possible because the ingest records
    truncation honestly. It did not always: a rate-limit refusal used to be
    recorded as the end of the list, so 932 cut-short histories were marked
    complete and trained on as though they were whole.
    """
    out: set[str] = set()
    for name in ETH_TRUNCATED_SETS:
        p = INTERIM / f"{name}.parquet"
        if p.exists():
            d = pl.read_parquet(p)
            out |= set(d.filter(~pl.col("truncated"))["address"].to_list())
    return out


def eth_fetched() -> set[str]:
    """Ethereum addresses whose own history we fetched, complete or not."""
    out: set[str] = set()
    for name in ETH_TRUNCATED_SETS:
        p = INTERIM / f"{name}.parquet"
        if p.exists():
            out |= set(pl.read_parquet(p)["address"].to_list())
    return out


def all_transfers() -> pl.DataFrame:
    """Every Tron transfer we hold, deduplicated on (tx_hash, to_address).

    Counterfeits are dropped by amount here as they are on Ethereum. Tron's
    normaliser matches on the token's SYMBOL too, so the same vector exists -
    deploy a TRC-20 calling itself USDT, send yourself a trillion, and the
    institutional guard vouches for you. In practice only one stored row exceeds
    the bound, because Tron carries far fewer impersonation tokens than Ethereum
    does, but the hole is the same shape. Filtering TRC-20 transfers on their
    contract address is the proper fix and wants each contract verified first;
    only USDT's is pinned in config today, and it is 99.98% of what we hold.

    A single transaction can carry several transfers, so the hash alone is not
    a key; the pair is what the ingest has always deduplicated on.
    """
    frames = []
    for root, name in TRANSFER_SETS:
        p = root / f"{name}.parquet"
        if p.exists():
            df = pl.read_parquet(p)
            frames.append(df if not frames else df.select(frames[0].columns))
    if not frames:
        raise SystemExit("no transfer data found - run the ingest first")
    out = pl.concat(frames, how="vertical_relaxed").unique(
        subset=["tx_hash", "to_address"])
    return out.filter(pl.col("amount_usd") < COUNTERFEIT_USD)


def all_indexed() -> set[str]:
    """Addresses whose own history we fetched."""
    out: set[str] = set()
    for name in TRUNCATED_SETS:
        p = INTERIM / f"{name}.parquet"
        if p.exists():
            out |= set(pl.read_parquet(p)["address"].to_list())
    return out

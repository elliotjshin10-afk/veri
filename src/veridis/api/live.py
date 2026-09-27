"""Live address lookup: score or research a wallet we have never seen.

The warehouse a wallet would run in production is an indexer. Here it is a
parquet file, so an address outside it has no history to read. Rather than
refuse, this fetches that address's TRC-20 history on demand, folds it into the
warehouse and rebuilds the feature engine.

Two paths, deliberately different in cost:

* **warm** - both addresses already indexed: pure compute, tens of milliseconds.
* **cold** - at least one address unknown: a chain fetch first, seconds, because
  TronGrid is rate-limited. Production replaces this with an indexer and the
  cold path disappears; the API reports which path it took so the number is
  never mistaken for the warm one.
"""
from __future__ import annotations

import asyncio
import logging
import time

import polars as pl

from veridis.chain.tron import tron_client, normalise_transfers, trc20_transfers
from veridis.dataset.ingest import TRANSFER_SCHEMA
from veridis.features.asof import FeatureEngine

log = logging.getLogger(__name__)

# Above either of these an address is moving institutional money, whatever
# shape its flows have.
INSTITUTIONAL_USD_PER_SENDER = 1_000_000.0
INSTITUTIONAL_INBOUND_USD = 100_000_000.0


class LiveWarehouse:
    """Warehouse plus an overlay of addresses fetched since start-up."""

    def __init__(
        self,
        base: pl.DataFrame,
        max_pages: int = 3,
        indexed: set[str] | None = None,
    ) -> None:
        """`indexed` is the set of addresses whose OWN history we fetched.

        Appearing in the warehouse is not the same as being indexed: most
        addresses in it are counterparties seen through somebody else's
        history, so we hold one or two of their transfers and none of the
        rest. Profiling those as though the history were complete produces
        confident nonsense - a wallet with one observed transfer reads as
        "100% of outflow to a single address". Anything not in `indexed` gets
        fetched before it is described.
        """
        self.base = base
        self.max_pages = max_pages
        self.overlay: list[pl.DataFrame] = []
        self._known: set[str] = set(indexed) if indexed else (
            set(base["from_address"].unique()) | set(base["to_address"].unique()))
        self._fetched: set[str] = set()
        self.engine = FeatureEngine(base)
        self.n_transfers = base.height

    def knows(self, address: str) -> bool:
        return address in self._known

    async def ensure(self, addresses: list[str]) -> dict:
        """Fetch any address we have no history for. Returns a fetch report."""
        missing = [
            a for a in dict.fromkeys(addresses)
            if a and a not in self._known and a not in self._fetched
        ]
        if not missing:
            return {"fetched": [], "cold": False, "fetch_ms": 0.0}

        t0 = time.perf_counter()
        rows: list[dict] = []
        async with tron_client(concurrency=1) as client:
            for addr in missing:
                try:
                    raw, _ = await trc20_transfers(
                        client, addr, max_pages=self.max_pages)
                    rows.extend(normalise_transfers(raw))
                except Exception as exc:  # noqa: BLE001
                    log.warning("live fetch failed for %s: %s", addr, exc)
                self._fetched.add(addr)
        fetch_ms = (time.perf_counter() - t0) * 1000

        if rows:
            df = pl.DataFrame(rows, schema=TRANSFER_SCHEMA).unique(
                subset=["tx_hash", "from_address", "to_address", "block_time"])
            self.overlay.append(df)
            self._known |= set(df["from_address"].unique()) | set(
                df["to_address"].unique())
            self._rebuild()
        return {"fetched": missing, "cold": True, "fetch_ms": round(fetch_ms, 1)}

    def _rebuild(self) -> None:
        """Rebuild the causal tables over base + overlay."""
        combined = pl.concat([self.base, *self.overlay], how="vertical_relaxed").unique(
            subset=["tx_hash", "from_address", "to_address", "block_time"])
        old = self.engine
        self.engine = FeatureEngine(combined)
        self.n_transfers = combined.height
        try:
            old.close()
        except Exception:  # noqa: BLE001
            pass

    def profile(self, address: str, as_of_ms: int) -> dict | None:
        """Destination-side facts about an address, as of a moment.

        This is the 'research a wallet' view: everything we would tell someone
        about where their money is going, with no sender involved.
        """
        con = self.engine.con
        q = """
        SELECT
          COUNT(*) FILTER (WHERE side='in')                       AS inbound_count,
          COUNT(*) FILTER (WHERE side='out')                      AS outbound_count,
          COUNT(DISTINCT peer) FILTER (WHERE side='in')           AS senders_all,
          COUNT(DISTINCT peer) FILTER (WHERE side='out')          AS payees_all,
          COALESCE(SUM(amount_usd) FILTER (WHERE side='in'),0)    AS inbound_usd,
          COALESCE(SUM(amount_usd) FILTER (WHERE side='out'),0)   AS outbound_usd,
          MIN(block_time)                                         AS first_seen,
          MAX(block_time)                                         AS last_seen,
          COUNT(DISTINCT peer) FILTER (
            WHERE side='in' AND block_time >= ? - 604800000)      AS senders_7d,
          COUNT(DISTINCT peer) FILTER (
            WHERE side='in' AND block_time >= ? - 2592000000)     AS senders_30d
        FROM legs WHERE address = ? AND block_time < ?
        """
        row = con.execute(q, [as_of_ms, as_of_ms, address, as_of_ms]).fetchone()
        if not row or not row[0] and not row[1]:
            return None
        cols = ["inbound_count", "outbound_count", "senders_all", "payees_all",
                "inbound_usd", "outbound_usd", "first_seen", "last_seen",
                "senders_7d", "senders_30d"]
        p = dict(zip(cols, row))
        hold = con.execute(
            "SELECT MEDIAN(hold_secs) FROM sweep WHERE address=? AND block_time<?",
            [address, as_of_ms]).fetchone()
        p["median_hold_secs"] = float(hold[0]) if hold and hold[0] is not None else None
        top = con.execute(
            """SELECT MAX(v)/NULLIF(SUM(v),0) FROM (
                 SELECT peer, SUM(amount_usd) v FROM legs
                 WHERE address=? AND side='out' AND block_time<? GROUP BY peer)""",
            [address, as_of_ms]).fetchone()
        p["consolidation_ratio"] = float(top[0]) if top and top[0] is not None else None
        p["age_days"] = round((as_of_ms - p["first_seen"]) / 86_400_000, 1) if p["first_seen"] else None
        p["forward_ratio"] = (
            round(p["outbound_usd"] / p["inbound_usd"], 3) if p["inbound_usd"] else None)
        p["usd_per_sender"] = (
            round(p["inbound_usd"] / p["senders_all"], 2) if p["senders_all"] else None)
        return p


def collection_verdict(p: dict) -> tuple[str, list[str]]:
    """Plain-language read of a destination profile.

    Deliberately rule-based and separate from the model: these are statements
    about what an address *does*, each checkable on a block explorer, not a
    prediction. The model scores transfers; this describes addresses.

    Scale is checked before shape. An exchange hot wallet has the same *shape*
    as a scam collector - many payers in, a couple of payees out, everything
    forwarded immediately - and a rule set that ignores size cheerfully calls
    Binance a collection address. (Observed: a real hot wallet with $20.4bn
    inbound tripped four of the six shape rules.) Retail collection points move
    thousands per victim, not hundreds of millions.
    """
    notes: list[str] = []

    inbound_usd = p.get("inbound_usd") or 0.0
    per_sender = p.get("usd_per_sender") or 0.0
    age = p.get("age_days") or 0.0

    # --- scale gate, before any shape rule ---
    if per_sender >= INSTITUTIONAL_USD_PER_SENDER or inbound_usd >= INSTITUTIONAL_INBOUND_USD:
        notes.append(
            f"averages ${per_sender:,.0f} per paying wallet across "
            f"${inbound_usd:,.0f} received")
        notes.append(
            "that is institutional scale - an exchange, bridge or desk, not a "
            "retail collection address")
        if age:
            notes.append(f"operating for {age/365:.1f} years")
        return "institutional infrastructure, not a retail collector", notes

    score = 0
    if (p.get("senders_all") or 0) >= 5:
        score += 1
        notes.append(f"{p['senders_all']} unrelated wallets have paid it")
    if age and age <= 60:
        score += 1
        notes.append(f"first seen {age:.0f} day{'' if round(age) == 1 else 's'} ago")
    fr = p.get("forward_ratio")
    if fr is not None and fr >= 0.85:
        score += 1
        notes.append(f"{min(fr,1)*100:.0f}% of what it receives is forwarded straight out")
    cr = p.get("consolidation_ratio")
    if cr is not None and cr >= 0.6:
        score += 1
        notes.append(f"{cr*100:.0f}% of its outflow goes to a single address")
    hold = p.get("median_hold_secs")
    if hold is not None and 0 <= hold <= 3600:
        score += 1
        mins = max(1, round(hold / 60))
        notes.append(
            f"funds leave within about {mins} minute{'' if mins == 1 else 's'} of arriving")
    inout = (p.get("inbound_count") or 0) / max(p.get("outbound_count") or 1, 1)
    if inout >= 2:
        score += 1
        notes.append(f"{inout:.0f}x more payments in ({p.get('inbound_count')}) "
                     f"than out ({p.get('outbound_count')})")

    # An address old enough to have a long benign history needs more than the
    # shape to be called a collector.
    if age >= 365 and score < 5:
        notes.append(f"has been active for {age/365:.1f} years, which retail "
                     f"collection addresses rarely are")
        return ("some collection-like behaviour, but long-lived" if score >= 3
                else "no clear collection pattern"), notes

    if score >= 4:
        return "behaves like a collection address", notes
    if score >= 2:
        return "some collection-like behaviour", notes
    return "no collection-like pattern", notes

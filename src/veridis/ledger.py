"""An append-only, tamper-evident log of predictions made BEFORE the fact.

Every number this project reports is retrospective, and a backtest is the
weakest form of evidence there is: the reader has to take on trust that the
protocol was not adjusted until it flattered the model. Twice in this project's
history a headline had to be walked back for exactly that reason.

A prediction published before the event answers that objection instead of
arguing with it. So: each night, score addresses that are on no list anywhere,
write down the ones that look like collection points, and wait. When Tether
later freezes one, the gap between our line and theirs is a measured lead time
that nobody has to trust us about - both ends are public and on-chain.

WHAT THE HASH CHAIN DOES AND DOES NOT PROVE
Each entry carries the hash of the entry before it, so the log cannot be
reordered or edited without every later hash changing. That is integrity, not
timestamping: on its own it proves the ORDER of entries, not when they were
written. The external anchor is publication - each night's head hash goes out
with the published page, and the host records when that happened. Anchoring to
a public chain would be stronger and is the obvious upgrade; until then the
claim is exactly this much and no more.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

GENESIS = "0" * 64


def _canonical(payload: dict) -> bytes:
    """Stable bytes for hashing: sorted keys, no incidental whitespace."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()


def entry_hash(prev: str, payload: dict) -> str:
    return hashlib.sha256(prev.encode() + _canonical(payload)).hexdigest()


@dataclass
class Ledger:
    path: Path
    _entries: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.path.exists():
            self._entries = [json.loads(l) for l in
                             self.path.read_text().splitlines() if l.strip()]

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self) -> Iterator[dict]:
        return iter(self._entries)

    @property
    def head(self) -> str:
        return self._entries[-1]["hash"] if self._entries else GENESIS

    def append(self, payload: dict) -> dict:
        """Add one entry, chained to the current head."""
        prev = self.head
        rec = {"prev": prev, "hash": entry_hash(prev, payload), **payload}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a") as fh:
            fh.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
        self._entries.append(rec)
        return rec

    def verify(self) -> tuple[bool, str]:
        """Recompute the whole chain. Any edit anywhere breaks it from there on."""
        prev = GENESIS
        for i, rec in enumerate(self._entries):
            payload = {k: v for k, v in rec.items() if k not in ("prev", "hash")}
            if rec["prev"] != prev:
                return False, f"entry {i} ({rec.get('address')}): prev does not match chain"
            want = entry_hash(prev, payload)
            if rec["hash"] != want:
                return False, f"entry {i} ({rec.get('address')}): content does not match its hash"
            prev = rec["hash"]
        return True, f"{len(self._entries)} entries, chain intact, head {prev[:16]}"


def utc_now() -> "dt.datetime":
    """Timezone-aware UTC now.

    Not datetime.utcnow(). That returns a NAIVE datetime holding UTC wall-clock,
    and .timestamp() then reinterprets it as LOCAL time - so on a machine four
    hours behind UTC it yields an epoch four hours in the FUTURE. The ledger
    records that number as predicted_at_ms, which anchors the point-in-time
    feature computation and is the baseline every lead-time figure is measured
    from. CI runs in UTC so the bug never surfaced there; it made the script
    unrunnable anywhere else, because the chain APIs refuse a future timestamp.
    """
    import datetime as dt
    return dt.datetime.now(dt.timezone.utc)


def utc_stamp() -> str:
    """UTC, to the second, in the form the ledger has always written."""
    return utc_now().strftime("%Y-%m-%dT%H:%M:%SZ")


def predictions_path(root: Path) -> Path:
    return root / "data" / "ledger" / "predictions.jsonl"


def resolutions_path(root: Path) -> Path:
    return root / "data" / "ledger" / "resolutions.jsonl"

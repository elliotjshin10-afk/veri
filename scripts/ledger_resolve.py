"""Check open predictions against the current freeze list and record outcomes.

A prediction is resolved when Tether freezes the address. The lead time is the
gap between our entry and their freeze - both public, both timestamped, neither
requiring anyone to trust us.

Resolutions go in their own chained log rather than mutating predictions. An
append-only file that gets edited in place is not append-only, and the whole
point of the ledger is that earlier entries cannot be touched.
"""
from __future__ import annotations

import datetime as dt, json, pathlib, sys
sys.path.insert(0, "src")
import polars as pl
from veridis.config import INTERIM, ROOT
from veridis.ledger import Ledger, predictions_path, resolutions_path, utc_stamp

DAY_MS = 86_400_000


def freeze_times() -> dict[str, int]:
    out: dict[str, int] = {}
    # Both chains. The Ethereum list used to be missing here, so the 13
    # Ethereum entries in the ledger could only ever resolve through the
    # labels parquet - a file rebuilt by hand, and therefore never in time.
    for name in ("tether_blacklist_tron.json", "tether_blacklist_eth.json"):
        raw = ROOT / "data" / "raw" / name
        if not raw.exists():
            continue
        for r in json.loads(raw.read_text()):
            a, t = r.get("address"), r.get("block_time")
            if a and t:
                out[a] = min(int(t), out.get(a, 1 << 62))
    p = INTERIM / "scam_addresses.parquet"
    if p.exists():
        d = pl.read_parquet(p).filter(pl.col("first_reported_at").is_not_null())
        for a, t in zip(d["address"].to_list(), d["first_reported_at"].to_list()):
            out.setdefault(a, int(t))
    return out


def main() -> None:
    preds = Ledger(predictions_path(ROOT))
    res = Ledger(resolutions_path(ROOT))
    for name, led in (("predictions", preds), ("resolutions", res)):
        ok, msg = led.verify()
        if not ok:
            sys.exit(f"{name} ledger broken: {msg}")
        print(f"{name}: {msg}")

    frozen = freeze_times()
    done = {e["address"] for e in res}
    stamp = utc_stamp()

    new = 0
    for p in preds:
        a = p["address"]
        if a in done or a not in frozen:
            continue
        ft = frozen[a]
        # A freeze that predates our call is not a prediction we got right; it
        # means the address was already listed when we flagged it and the
        # unlisted filter let it through. Recorded, never counted as a hit.
        lead_days = (ft - p["predicted_at_ms"]) / DAY_MS
        res.append({
            "address": a, "predicted_at": p["predicted_at"],
            "prediction_hash": p["hash"], "resolved_at": stamp,
            "frozen_at_ms": ft,
            "frozen_at": dt.datetime.utcfromtimestamp(ft / 1000).replace(
                microsecond=0).isoformat() + "Z",
            "lead_days": round(lead_days, 2),
            "valid": bool(lead_days > 0),
        })
        done.add(a)
        new += 1

    hits = [e for e in res if e["valid"]]
    print(f"\nchecked {len(preds):,} predictions against {len(frozen):,} known freezes")
    print(f"  newly resolved this run : {new}")
    print(f"  resolved in total       : {len(list(res))}  ({len(hits)} ahead of the freeze)")
    if hits:
        lead = sorted(e["lead_days"] for e in hits)
        print(f"  lead time: median {lead[len(lead)//2]:.1f}d, "
              f"min {lead[0]:.1f}d, max {lead[-1]:.1f}d")
    print(f"  resolutions head {res.head[:16]}")


if __name__ == "__main__":
    main()

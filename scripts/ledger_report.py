"""Summarise the ledger for the site, without flattering it.

The trap in reporting a live prediction log is treating every unresolved entry
as a miss. A prediction made yesterday has not failed - it has not been given
time to be judged. Counting it against us understates the model; quietly
dropping it overstates it. Both are wrong in the same way: they ignore that the
data is right-censored.

So outcomes are reported by cohort age. Entries old enough to have plausibly
been frozen carry a resolution rate; younger ones are reported as still open
and excluded from that rate, with their count shown so nothing is hidden.

The cohort boundary is not arbitrary: the median lead time in the backtest was
90 days, so a prediction younger than that has not had a median amount of time
to come true.

There is a second way the same trap bites. Some calls land on addresses holding
almost nothing - one payer, ten dollars, nothing forwarded. Those will never be
frozen, because nothing ever happened there, so counting them against the model
measures the sampling rather than the model.

The obvious fix is the wrong one. A `senders >= 3` floor on predictions removes
that noise but costs 29 of 248 real detections at T-90 - addresses that WERE
eventually frozen despite looking thin three months out, which is exactly the
early catch this product exists for. Measured, not assumed.

So nothing is floored and nothing is dropped. Substance is reported alongside
age, both tiers shown, and the headline rate comes from the calls with real
money behind them while the thin ones stay visible and counted separately.
"""
from __future__ import annotations

import datetime as dt, json, sys
sys.path.insert(0, "src")
from veridis.config import PROCESSED, ROOT, SITE
from veridis.ledger import Ledger, predictions_path, resolutions_path, utc_now, utc_stamp

DAY_MS = 86_400_000
COHORTS = [(90, None, "90+ days old"), (30, 90, "30-90 days old"), (0, 30, "under 30 days")]
# A call with real money behind it. At T-90 this tier held 213 of 248 true
# detections, so it is where a hit rate is worth quoting.
MIN_SENDERS, MIN_INBOUND = 5, 1000.0


def substance(p: dict) -> tuple[int, float]:
    """Senders and inbound, from the record or reconstructed from its features."""
    if "senders" in p:
        return int(p["senders"]), float(p.get("inbound_usd") or 0.0)
    f = p.get("features") or {}
    n = int(f.get("dest_senders_all") or 0)
    return n, float((f.get("dest_usd_per_sender") or 0.0) * n)


def main() -> None:
    preds, res = Ledger(predictions_path(ROOT)), Ledger(resolutions_path(ROOT))
    for name, led in (("predictions", preds), ("resolutions", res)):
        ok, msg = led.verify()
        if not ok:
            sys.exit(f"{name} ledger broken: {msg}")

    hits = {e["address"]: e for e in res if e["valid"]}
    now = utc_now().timestamp() * 1000
    rows = []
    for p in preds:
        age = (now - p["predicted_at_ms"]) / DAY_MS
        n, usd = substance(p)
        rows.append({"address": p["address"], "score": p["score"],
                     "predicted_at": p["predicted_at"], "age_days": age,
                     "senders": n, "inbound_usd": usd,
                     "substantive": n >= MIN_SENDERS and usd >= MIN_INBOUND,
                     "frozen": p["address"] in hits,
                     "lead_days": hits.get(p["address"], {}).get("lead_days")})

    out = {
        "generated_at": utc_stamp(),
        "predictions": len(rows),
        "head": preds.head,
        "resolutions_head": res.head,
        "chain_verified": True,
        "cohorts": [],
        # The newest open calls, for the page to show as live claims.
        "min_senders": MIN_SENDERS, "min_inbound": MIN_INBOUND,
        "substantive": sum(1 for r in rows if r["substantive"]),
        # Only calls with money behind them are offered as live claims; a
        # public prediction on a ten-dollar wallet is not worth anyone's time.
        "open_recent": [r for r in sorted(rows, key=lambda r: -r["score"])
                        if not r["frozen"] and r["substantive"]][:10],
        "confirmed": sorted([r for r in rows if r["frozen"]],
                            key=lambda r: -(r["lead_days"] or 0))[:10],
    }
    for lo, hi, label in COHORTS:
        c = [r for r in rows if r["age_days"] >= lo and (hi is None or r["age_days"] < hi)]
        if not c:
            continue
        frozen = [r for r in c if r["frozen"]]
        sub = [r for r in c if r["substantive"]]
        sub_frozen = [r for r in sub if r["frozen"]]
        entry = {"label": label, "n": len(c), "frozen": len(frozen),
                 "n_sub": len(sub), "frozen_sub": len(sub_frozen)}
        # Only cohorts old enough to be judged carry a rate, and the rate that
        # leads is the one from calls with substance behind them.
        if lo >= 30:
            entry["rate"] = len(frozen) / len(c)
            if sub:
                entry["rate_sub"] = len(sub_frozen) / len(sub)
            if frozen:
                lead = sorted(r["lead_days"] for r in frozen)
                entry["median_lead_days"] = lead[len(lead) // 2]
        else:
            entry["too_early"] = True
        out["cohorts"].append(entry)

    (PROCESSED / "ledger_report.json").write_text(json.dumps(out, indent=2))
    (SITE / "ledger.json").write_text(json.dumps(out, separators=(",", ":")))

    print(f"predictions {out['predictions']:,}  "
          f"({out['substantive']:,} with >={MIN_SENDERS} payers and >=${MIN_INBOUND:,.0f})"
          f"   head {out['head'][:16]}")
    for c in out["cohorts"]:
        if c.get("too_early"):
            print(f"  {c['label']:16s} {c['n']:5,} ({c['n_sub']:,} substantive)"
                  f"  still open - too early to judge")
        else:
            print(f"  {c['label']:16s} {c['n']:5,} ({c['n_sub']:,} substantive)"
                  f"  frozen: {c['frozen']} overall ({c['rate']:.0%})"
                  + (f", {c['frozen_sub']} of the substantive ({c['rate_sub']:.0%})"
                     if c.get("rate_sub") is not None else "")
                  + (f", median lead {c['median_lead_days']:.0f}d" if c.get("median_lead_days") else ""))
    print(f"\nwrote {PROCESSED/'ledger_report.json'} and {SITE/'ledger.json'}")


if __name__ == "__main__":
    main()

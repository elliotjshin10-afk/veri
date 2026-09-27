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
"""
from __future__ import annotations

import datetime as dt, json, sys
sys.path.insert(0, "src")
from veridis.config import PROCESSED, ROOT, SITE
from veridis.ledger import Ledger, predictions_path, resolutions_path

DAY_MS = 86_400_000
COHORTS = [(90, None, "90+ days old"), (30, 90, "30-90 days old"), (0, 30, "under 30 days")]


def main() -> None:
    preds, res = Ledger(predictions_path(ROOT)), Ledger(resolutions_path(ROOT))
    for name, led in (("predictions", preds), ("resolutions", res)):
        ok, msg = led.verify()
        if not ok:
            sys.exit(f"{name} ledger broken: {msg}")

    hits = {e["address"]: e for e in res if e["valid"]}
    now = dt.datetime.utcnow().timestamp() * 1000
    rows = []
    for p in preds:
        age = (now - p["predicted_at_ms"]) / DAY_MS
        rows.append({"address": p["address"], "score": p["score"],
                     "predicted_at": p["predicted_at"], "age_days": age,
                     "frozen": p["address"] in hits,
                     "lead_days": hits.get(p["address"], {}).get("lead_days")})

    out = {
        "generated_at": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
        "predictions": len(rows),
        "head": preds.head,
        "resolutions_head": res.head,
        "chain_verified": True,
        "cohorts": [],
        # The newest open calls, for the page to show as live claims.
        "open_recent": [r for r in sorted(rows, key=lambda r: -r["score"]) if not r["frozen"]][:10],
        "confirmed": sorted([r for r in rows if r["frozen"]],
                            key=lambda r: -(r["lead_days"] or 0))[:10],
    }
    for lo, hi, label in COHORTS:
        c = [r for r in rows if r["age_days"] >= lo and (hi is None or r["age_days"] < hi)]
        if not c:
            continue
        frozen = [r for r in c if r["frozen"]]
        entry = {"label": label, "n": len(c), "frozen": len(frozen)}
        # Only cohorts old enough to be judged carry a rate.
        if lo >= 30:
            entry["rate"] = len(frozen) / len(c)
            if frozen:
                lead = sorted(r["lead_days"] for r in frozen)
                entry["median_lead_days"] = lead[len(lead) // 2]
        else:
            entry["too_early"] = True
        out["cohorts"].append(entry)

    (PROCESSED / "ledger_report.json").write_text(json.dumps(out, indent=2))
    (SITE / "ledger.json").write_text(json.dumps(out, separators=(",", ":")))

    print(f"predictions {out['predictions']:,}   head {out['head'][:16]}")
    for c in out["cohorts"]:
        if c.get("too_early"):
            print(f"  {c['label']:16s} {c['n']:5,}  still open - too early to judge")
        else:
            print(f"  {c['label']:16s} {c['n']:5,}  frozen since: {c['frozen']} "
                  f"({c['rate']:.0%})"
                  + (f", median lead {c['median_lead_days']:.0f}d" if c.get("median_lead_days") else ""))
    print(f"\nwrote {PROCESSED/'ledger_report.json'} and {SITE/'ledger.json'}")


if __name__ == "__main__":
    main()

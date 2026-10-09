"""What the catch rate means for somebody who is not in our dataset.

"97% of transfers into an address that was later frozen met a warning" is a
recall figure, and recall does not depend on how common fraud is. Precision
does, completely. A person using this does not experience recall; they
experience how often a stop was right. Those two numbers can be 97% and 4% at
the same time, and publishing only the first is the most misleading honest
thing this project could do.

Three parts, in order of how much they can hurt the claim.

PREVALENCE. Measured, not assumed: sample addresses receiving USDT right now
and count how many are already on Tether's freeze list. That is a floor rather
than the true rate, because an address frozen next year is not frozen today,
and it is the right floor to reason from.

PRECISION. From the held-out confusion at the shipped bands, re-weighted to
that prevalence. The arithmetic is only Bayes, but the result is the number a
wallet integrating this would actually argue about.

SELECTION. Our positives are transfers into addresses Tether LATER FROZE.
Tether freezes on law-enforcement request, which selects for large, reported,
egregious fraud. The catch rate is therefore conditional on a scam being the
kind that eventually gets frozen, and a small scam that never does is invisible
to us in training and in evaluation alike. No amount of held-out discipline
fixes that; it is a property of the only labels that exist.

Writes data/processed/prevalence_report.json.
Run: python scripts/m15_prevalence.py [n_sampled]
"""
from __future__ import annotations

import asyncio
import json
import sys

sys.path.insert(0, "src")

import polars as pl

from veridis.chain.tron import tron_client, PAGE
from veridis.config import INTERIM, PROCESSED, ROOT, SITE, TRONGRID_BASE, USDT_TRON
from veridis.ledger import utc_now

N = int(sys.argv[1]) if len(sys.argv) > 1 else 4000


async def recent_recipients(client, n: int) -> list[str]:
    """Addresses receiving USDT across the last day and a half.

    The same population the nightly ledger samples, and the same one a wallet
    would sit in front of: whoever is being paid right now, not whoever we
    chose to study.
    """
    url = f"{TRONGRID_BASE}/v1/contracts/{USDT_TRON}/events"
    now = int(utc_now().timestamp() * 1000)
    seen: list[str] = []
    # Spread over a week rather than a day. A single day is one market's worth
    # of behaviour, and with a rate this low the sample size is the whole
    # precision of the estimate.
    for hours_back in (1, 3, 6, 10, 16, 24, 30, 36, 48, 60, 72, 96, 120, 144,
                       168, 192, 216, 240, 264, 288, 312, 336):
        lo = now - hours_back * 3_600_000
        payload = await client.get_json(url, {
            "event_name": "Transfer", "limit": PAGE,
            "min_block_timestamp": lo, "max_block_timestamp": lo + 900_000,
            "order_by": "block_timestamp,asc"})
        for e in (payload or {}).get("data", []):
            to = (e.get("result") or {}).get("to")
            if to:
                seen.append(to)
        if len(seen) >= n:
            break
    return seen[:n]


def frozen_set() -> set[str]:
    rows = json.loads((ROOT / "data" / "raw" / "tether_blacklist_tron.json").read_text())
    return {r["address"] for r in rows if r.get("address")}


def confusion() -> dict:
    """The shipped operating point, from the decided-cases log."""
    return json.loads((SITE / "history.json").read_text())["summary"]


def precision_at(recall: float, fpr: float, p: float) -> float:
    tp, fp = recall * p, fpr * (1 - p)
    return tp / (tp + fp) if (tp + fp) else 0.0


async def main() -> None:
    from veridis.chain.address import hex_to_base58

    async with tron_client(concurrency=2) as c:
        raw = await recent_recipients(c, N)
    addrs = []
    for a in raw:
        try:
            addrs.append(hex_to_base58(a) if a.startswith("41") else a)
        except ValueError:
            continue
    uniq = sorted(set(addrs))
    frozen = frozen_set()
    hit = [a for a in uniq if a in frozen]
    # Transfer-weighted as well as address-weighted: a collection point receives
    # many payments, so the share of PAYMENTS going somewhere already frozen is
    # the number that matters to a payer, and it is the larger of the two.
    by_transfer = sum(1 for a in addrs if a in frozen)
    prev_addr = len(hit) / max(len(uniq), 1)
    prev_tx = by_transfer / max(len(addrs), 1)
    print(f"sampled {len(addrs):,} USDT receipts, {len(uniq):,} distinct addresses")
    print(f"  already on the freeze list: {len(hit):,} addresses "
          f"({prev_addr:.3%}), {by_transfer:,} receipts ({prev_tx:.3%})")

    # Rule of three: zero events in n samples puts the 95% upper bound at 3/n.
    # With a rate this low that bound is the only honest summary, and it is the
    # number to carry into the precision arithmetic.
    upper = 3.0 / max(len(uniq), 1)
    if not hit:
        print(f"  none found, so prevalence < {upper:.3%} at 95% confidence "
              f"(rule of three)")

    s = confusion()
    rec_hi = s["scam_high"] / s["n_scam"]
    rec_any = (s["scam_high"] + s["scam_elevated"]) / s["n_scam"]
    fpr_hi = s["control_high"] / s["n_control"]
    fpr_any = (s["control_high"] + s["control_elevated"]) / s["n_control"]
    print(f"\nheld-out operating point: stop {rec_hi:.1%} of scam transfers at "
          f"{fpr_hi:.1%} of ordinary; any warning {rec_any:.1%} at {fpr_any:.1%}")

    # The dataset's own base rate, for contrast. It is a design choice, not a
    # measurement of the world, and it is 20x to 200x what the world looks like.
    study = s["n_scam"] / s["n"]

    # The likelihood ratio is the honest headline, because it is the only one
    # of these that does not move with prevalence. "A stop makes it N times
    # more likely you are paying a collection point" is true for every reader,
    # where a precision figure is true only for a reader whose prevalence we
    # guessed right.
    lr_hi, lr_any = rec_hi / fpr_hi, rec_any / fpr_any
    print(f"\nlikelihood ratio, which does not depend on prevalence:")
    print(f"  a STOP    multiplies the odds by {lr_hi:>5.1f}x")
    print(f"  a WARNING multiplies the odds by {lr_any:>5.1f}x")
    print(f"\nour dataset is {study / max(upper, 1e-12):,.0f}x more fraud-dense "
          f"than the live stream's upper bound")

    grid = sorted({0.0005, 0.001, 0.005, 0.01, 0.05, round(study, 4)})
    print(f"\n  {'prevalence':<14}{'precision of a STOP':>22}{'of any WARNING':>18}")
    rows = []
    for p in grid:
        a, b = precision_at(rec_hi, fpr_hi, p), precision_at(rec_any, fpr_any, p)
        tag = ""
        if abs(p - study) < 1e-9:
            tag = "  <- our dataset"
        print(f"  {p:<14.3%}{a:>21.1%}{b:>18.1%}{tag}")
        rows.append({"prevalence": p, "precision_stop": a, "precision_warn": b})

    doc = {
        "sampled_receipts": len(addrs), "distinct_addresses": len(uniq),
        "already_frozen_addresses": len(hit),
        "prevalence_address": prev_addr, "prevalence_transfer": prev_tx,
        "prevalence_upper_95": upper, "lr_stop": lr_hi, "lr_warn": lr_any,
        "study_base_rate": study,
        "recall_stop": rec_hi, "recall_any": rec_any,
        "fpr_stop": fpr_hi, "fpr_any": fpr_any,
        "precision_curve": rows,
    }
    (PROCESSED / "prevalence_report.json").write_text(json.dumps(doc, indent=2))
    print(f"\nwrote {PROCESSED / 'prevalence_report.json'}")
    print("\nThis is a floor on prevalence: an address frozen next year is not "
          "frozen today.\nIt is also conditional on the only labels that exist. "
          "Tether freezes on\nlaw-enforcement request, so a scam too small to be "
          "reported never enters\neither the training set or this measurement.")


asyncio.run(main())

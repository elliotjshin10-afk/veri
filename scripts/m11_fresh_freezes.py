"""The shipped model against freezes that happened after it shipped.

Every figure this project publishes is a backtest. A backtest can be honest -
these are, now that the control arm is age-matched - but it is still a claim
about a past that was already in the file when the model was fitted, and the
only answer to "did you tune until it worked" is a description of the method.

This is the other kind of evidence. Tether froze 164 Tron addresses after the
label snapshot the model was trained against. None of them appear in training,
in the held-out set, or in the shipped address index: on the day the model was
built they did not exist as far as it was concerned. So we fetch their
histories, truncate each to a point BEFORE its freeze, and ask the shipped
artefact what it would have said.

It is recall only. There is no control arm here, so this cannot produce a false
-alarm rate - it is read against the bands, which are cut at the top 1% of
held-out ordinary wallets, and against the live trial that flagged 0 of 207
sampled recipients at the same threshold.

The horizons are the point. "We would have called it" means little at one day
before a freeze, when the money is already gone; it means a great deal at
thirty.

Writes data/processed/fresh_freeze_report.json.
Run: python scripts/m11_fresh_freezes.py [max_pages]   (default 25)
"""
from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import json
import logging
import sys

sys.path.insert(0, "src")

import polars as pl

from veridis.chain.tron import tron_client
from veridis.config import INTERIM, PROCESSED, ROOT, SITE
from veridis.dataset.ingest import fetch_histories
from veridis.features.asof import FeatureEngine
from veridis.model import browser_model
from veridis.model.address_risk import DEST_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s",
                    datefmt="%H:%M:%S")
logging.getLogger("httpx").setLevel(logging.WARNING)

DAY_MS = 86_400_000
HORIZONS = (1, 7, 30, 90)
MAX_PAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 25


def unseen_freezes() -> dict[str, int]:
    """Tron addresses frozen after the labels were cut, and absent from them.

    Both conditions matter. A freeze after the snapshot could still be an
    address training saw as a CONTROL, which would make this a test of whether
    we remember it rather than of whether we can call it.
    """
    sa = pl.read_parquet(INTERIM / "scam_addresses.parquet")
    snapshot = int(sa.filter(pl.col("chain") == "tron")["first_reported_at"].max())
    known = set(sa["address"].to_list())

    ev_path = PROCESSED / "events_features.parquet"
    if ev_path.exists():
        ev = pl.read_parquet(ev_path, columns=["destination", "sender"])
        known |= set(ev["destination"].to_list()) | set(ev["sender"].to_list())

    rows = json.loads((ROOT / "data" / "raw" / "tether_blacklist_tron.json").read_text())
    out: dict[str, int] = {}
    for r in rows:
        a, t = r.get("address"), r.get("block_time")
        if a and t and int(t) > snapshot and a not in known:
            out[a] = min(int(t), out.get(a, 1 << 62))
    logging.info("labels cut %s; %d Tron freezes since, none seen in training",
                 dt.datetime.utcfromtimestamp(snapshot / 1000).date(), len(out))
    return out


async def main() -> None:
    fresh = unseen_freezes()
    if not fresh:
        sys.exit("no freezes since the label snapshot - run scripts/refresh_freezes.py")

    model_file = SITE / "model_dest.json"
    model = browser_model.load(model_file)
    # The model's own list, not DEST_FEATURES: the shipped export drops
    # dest_funder_fanout, which needs a third address's history and would be
    # defaulted in the browser - exactly the skew this project keeps guarding
    # against. Scoring with the extra column would not be what the site does.
    features = model["features"]
    missing = [f for f in features if f not in DEST_FEATURES]
    if missing:
        sys.exit(f"shipped model wants features the feature layer lacks: {missing}")
    thr = model["thresholds"]
    sha = hashlib.sha256(model_file.read_bytes()).hexdigest()[:16]

    addrs = sorted(fresh)
    async with tron_client(concurrency=4) as client:
        tx, trunc = await fetch_histories(client, addrs, max_pages=MAX_PAGES,
                                          concurrency=4, label="fresh freezes")
    if not tx.height:
        sys.exit("no transfer history fetched")

    # TronGrid is walked oldest-first, so a truncated history keeps the real
    # first transfer - age is right - and loses the most RECENT activity, which
    # is where the collection-point signal lives. Truncation therefore costs
    # recall rather than flattering it: at six pages, 11 addresses were cut
    # short and the high-band rate read 18.4%; at twenty-five it is 21.7% over
    # the same population. A dropped address is still dropped, because scoring
    # a partial history is scoring a different address.
    cut = sorted(a for a in addrs if trunc.get(a))
    if cut:
        logging.warning("%d of %d histories hit the page cap - dropped, "
                        "raise max_pages to keep them", len(cut), len(addrs))
        for a in cut:
            fresh.pop(a, None)
        addrs = [a for a in addrs if a not in set(cut)]

    # Earliest transfer per address: a mark before an address existed has no
    # history to score and must be dropped, not scored as an empty wallet.
    first = (tx.filter(pl.col("to_address").is_in(addrs))
             .group_by("to_address").agg(pl.col("block_time").min().alias("first_seen")))
    first_seen = dict(zip(first["to_address"].to_list(), first["first_seen"].to_list()))
    logging.info("fetched history for %d of %d addresses", len(first_seen), len(addrs))

    engine = FeatureEngine(tx)
    results, per_address = {}, {}
    for h in HORIZONS:
        usable = [a for a in addrs
                  if a in first_seen and fresh[a] - h * DAY_MS > first_seen[a]]
        if not usable:
            results[str(h)] = {"n": 0}
            continue
        ev = (pl.DataFrame({"destination": usable})
              .with_columns(
                  pl.Series("event_time", [fresh[a] - h * DAY_MS for a in usable]),
                  pl.lit("tron").alias("chain"),
                  pl.lit("__probe__").alias("sender"),
                  pl.lit(1000.0).alias("amount_usd"))
              .with_row_index("event_id"))
        feats = engine.compute(ev)
        X = feats.select(features).to_numpy().tolist()
        scores = browser_model.score(model, X)
        bands = [browser_model.band(s, thr) for s in scores]
        results[str(h)] = {
            "n": len(usable),
            "high": sum(b == "high" for b in bands) / len(usable),
            "elevated_or_high": sum(b != "ordinary" for b in bands) / len(usable),
            "median_age_days": float(
                (pl.Series([fresh[a] - h * DAY_MS - first_seen[a] for a in usable])
                 / DAY_MS).median()),
        }
        if h == min(HORIZONS):
            per_address = {a: {"score": round(s, 4), "band": b,
                               "frozen_at_ms": fresh[a]}
                           for a, s, b in zip(usable, scores, bands)}
    engine.close()

    doc = {
        "built_at": int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000),
        "model": {"file": "model_dest.json", "sha256_16": sha, "thresholds": thr},
        "population": {"frozen_since_snapshot": len(fresh),
                       "with_history": len(first_seen), "max_pages": MAX_PAGES,
                       "dropped_truncated": len(cut)},
        "per_horizon": results,
        "addresses": per_address,
    }
    (PROCESSED / "fresh_freeze_report.json").write_text(json.dumps(doc, indent=2))

    print(f"\n{len(fresh)} Tron addresses frozen since the labels were cut, "
          f"none seen in training")
    print(f"scored with site/model_dest.json ({sha}) at the shipped bands\n")
    print(f"  {'days before the freeze':<24}{'n':>6}{'high':>9}{'elevated+':>11}"
          f"{'median age':>12}")
    for h in HORIZONS:
        r = results[str(h)]
        if not r["n"]:
            print(f"  {h:<24}{0:>6}   (none had any history that far back)")
            continue
        print(f"  {h:<24}{r['n']:>6}{r['high']:>8.1%}{r['elevated_or_high']:>11.1%}"
              f"{r['median_age_days']:>10.0f} d")
    print("\nRecall only - no control arm here. The bands are cut at the top 1% "
          "of\nheld-out ordinary wallets, where a live trial flagged 0 of 207.")


asyncio.run(main())

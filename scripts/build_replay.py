"""Build the replay: one real victim, scored at each transfer as it happened.

A lookup page shows what an address IS. It cannot show the thing the product is
for - the minutes in which somebody is being talked out of their money - because
there is no "before" on a page you visit after the fact.

So this replays one relationship. Every score is computed from the transfers
that existed BEFORE that moment and nothing else, using the shipped models, and
the case is chosen from the held-out half: the destination never appeared in
training. The first case considered was a better story - a $10 test transfer
followed by $181,714 inside two hours - and was discarded because the model had
seen it. A demo that cannot survive "was this in your training set?" is worth
less than no demo.

Writes site/replay.json. Run: python scripts/build_replay.py
"""
from __future__ import annotations

import hashlib, json, pathlib, sys
sys.path.insert(0, "src")

import lightgbm as lgb
import polars as pl

from veridis.config import INTERIM, PROCESSED, SITE
from veridis.dataset.holdings import all_transfers
from veridis.features.asof import FEATURE_COLUMNS, FEATURE_FAMILY, FeatureEngine
from veridis.model.address_risk import DEST_FEATURES
from veridis.model.train import temporal_split

VICTIM = "TQJbQwiK2SY4hg7t3zapLcz9T8H5FbG2Pj"
DEST = "TAGjwGdZJ2UGHMjXyQeowFSEa1BKFCRnsE"

PAIR = [c for c in FEATURE_COLUMNS
        if FEATURE_FAMILY[c] in ("destination", "context", "relationship")
        and c != "dest_funder_fanout"]
BROWSER = [c for c in DEST_FEATURES if c != "dest_funder_fanout"]


def sha16(p: pathlib.Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:16]


def main() -> None:
    # Prove the claim rather than assert it: the destination must be absent
    # from the training split, or this is the model recalling an answer.
    ev = pl.read_parquet(PROCESSED / "events_features.parquet")
    sp = temporal_split(ev)
    if DEST in set(sp.train["destination"].to_list()):
        sys.exit(f"{DEST} is in the training split - pick another case")

    tf = all_transfers()
    hits = (tf.filter((pl.col("from_address") == VICTIM)
                      & (pl.col("to_address") == DEST))
            .sort("block_time").select("block_time", "amount_usd"))
    if not hits.height:
        sys.exit("no transfers found for that pair")

    probes = (hits.select(pl.col("block_time").alias("event_time"), "amount_usd")
              .with_columns(pl.lit(DEST).alias("destination"),
                            pl.lit("tron").alias("chain"),
                            pl.lit(VICTIM).alias("sender"))
              .with_row_index("event_id"))
    engine = FeatureEngine(tf)
    feats = engine.compute(probes)
    engine.close()

    dest_m = PROCESSED / "model_browser.txt"
    pair_m = PROCESSED / "model_pair.txt"
    dest_s = lgb.Booster(model_file=str(dest_m)).predict(
        feats.select(BROWSER).to_numpy())
    pair_s = lgb.Booster(model_file=str(pair_m)).predict(
        feats.select(PAIR).to_numpy())

    dthr = json.loads((PROCESSED / "address_model_report.json").read_text())["thresholds"]
    pthr = json.loads((SITE / "model_pair.json").read_text())["thresholds"]

    def band(p, t):
        return "high" if p >= t["high"] else "elevated" if p >= t["elevated"] else "ordinary"

    sa = pl.read_parquet(INTERIM / "scam_addresses.parquet")
    frozen_at = dict(zip(sa["address"].to_list(),
                         sa["first_reported_at"].to_list())).get(DEST)

    steps, running = [], 0.0
    for (bt, amt), d, p, row in zip(hits.iter_rows(), dest_s, pair_s,
                                    feats.iter_rows(named=True)):
        running += amt
        # How much history existed at this instant - the honest denominator
        # behind every number on the row.
        seen = tf.filter(pl.col("block_time") < bt).filter(
            (pl.col("to_address") == DEST) | (pl.col("from_address") == DEST)).height
        why = []
        if row.get("is_first_send_to_dest"):
            why.append("first time this wallet has paid this address")
        esc = row.get("escalation_ratio") or 0
        if esc > 5:
            why.append(f"{esc:,.0f}x their previous transfer to it")
        if (row.get("shared_counterparties") or 0) == 0:
            why.append("no counterparty in common with the sender")
        fwd = row.get("dest_forward_ratio")
        if fwd is not None and fwd > 0.9:
            why.append(f"forwards {min(fwd,1)*100:.0f}% of what it receives")
        steps.append({
            "at": int(bt), "amount": round(float(amt), 2),
            "running": round(running, 2),
            "dest": {"score": round(float(d), 4), "band": band(d, dthr)},
            "pair": {"score": round(float(p), 4), "band": band(p, pthr)},
            "evidence": {
                "transfers_known": int(seen),
                "payers": int(row.get("dest_senders_all") or 0),
                "first_send": bool(row.get("is_first_send_to_dest")),
            },
            "why": why[:3],
        })

    out = {
        "victim": VICTIM, "destination": DEST,
        "frozen_at": int(frozen_at) if frozen_at else None,
        "total": round(running, 2),
        "steps": steps,
        "thresholds": {"dest": dthr, "pair": pthr},
        # Provenance, so a reader can check the numbers came from the models we
        # say they did and not ones fitted afterwards.
        "provenance": {
            "held_out": True,
            "dest_model_sha256_16": sha16(dest_m),
            "pair_model_sha256_16": sha16(pair_m),
            "point_in_time": "every score uses only transfers before its own timestamp",
        },
    }
    (SITE / "replay.json").write_text(json.dumps(out, separators=(",", ":")))
    lead = (frozen_at - steps[-1]["at"]) / 86_400_000 if frozen_at else None
    print(f"wrote site/replay.json - {len(steps)} transfers, ${running:,.0f} lost"
          + (f", frozen {lead:.0f} days later" if lead else ""))


if __name__ == "__main__":
    main()

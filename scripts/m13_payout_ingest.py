"""Ingest the consolidation wallets themselves, and test the inverted signal.

payout_fanin asks: the wallet your money is forwarded to, how many OTHER wallets
pay it? The version computed from the warehouse answers a different question,
because the warehouse holds histories only for addresses we deliberately
fetched, so a consolidation wallet's inbound edges are whichever ones happen to
pass through the sample. Every one of the 1,363 payout wallets below already
appears in the warehouse as somebody's counterparty, and not one of them has its
own history there. That is the artefact.

So this fetches them. Then it recomputes the feature point-in-time, as the count
of distinct wallets that had paid the consolidation address BEFORE the moment
being scored, and retrains the destination model with and without it.

The direction is the opposite of the one the code originally claimed, and that
is the point of building it. A scam collection point forwards to a mule wallet
fed by a handful of addresses. An ordinary wallet forwards to an exchange, and
an exchange hot wallet is fed by thousands. "Your money is going somewhere
almost nobody else pays" is the signal, and no blocklist can see it.

Two columns come out, because one would be dishonest:

  payout_fanin_pit      distinct payers before the scoring moment
  payout_fanin_exact    1 when the fetch covers that moment, 0 when the count
                        is a lower bound because the history hit the page cap

Truncation is not noise here, it is informative and it is one-directional: only
busy wallets truncate, so a lower bound always understates legitimacy. The flag
lets the model treat the two cases differently instead of us inventing a number.

Writes data/interim/payout_transfers.parquet and
data/processed/payout_fanin_eval.json.
Run: python scripts/m13_payout_ingest.py [max_pages]
"""
from __future__ import annotations

import asyncio
import json
import sys

sys.path.insert(0, "src")

import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from veridis.chain.tron import tron_client
from veridis.config import INTERIM, PROCESSED
from veridis.dataset.holdings import all_transfers
from veridis.dataset.ingest import fetch_histories
from veridis.model.address_risk import DEST_FEATURES
from veridis.model.train import temporal_split

MAX_PAGES = int(sys.argv[1]) if len(sys.argv) > 1 else 15
BASE = [f for f in DEST_FEATURES if f != "dest_funder_fanout"]
NEW = ["payout_fanin_pit", "payout_fanin_exact"]


def top_payees(tf: pl.DataFrame, dests: list[str]) -> pl.DataFrame:
    """Each destination's largest payout address, by total value sent."""
    return (tf.filter(pl.col("from_address").is_in(dests) & (pl.col("amount_usd") > 0))
            .group_by(["from_address", "to_address"])
            .agg(pl.col("amount_usd").sum().alias("v"))
            .sort("v", descending=True)
            .group_by("from_address").first()
            .select(pl.col("from_address").alias("destination"),
                    pl.col("to_address").alias("payout")))


async def ingest(payouts: list[str]) -> tuple[pl.DataFrame, dict[str, bool]]:
    path = INTERIM / "payout_transfers.parquet"
    async with tron_client(concurrency=4) as c:
        tx, trunc = await fetch_histories(c, payouts, max_pages=MAX_PAGES,
                                          concurrency=4, label="payout wallets")
    tx.write_parquet(path)
    pl.DataFrame({"address": list(trunc), "truncated": list(trunc.values())}) \
        .write_parquet(INTERIM / "payout_truncated.parquet")
    print(f"  wrote {path} ({tx.height:,} transfers)")
    return tx, trunc


def fanin_features(ev: pl.DataFrame, link: pl.DataFrame,
                   ptx: pl.DataFrame, trunc: dict[str, bool]) -> pl.DataFrame:
    """Point-in-time fan-in at each event's payout address."""
    # Inbound edges of the payout wallets, deduplicated to (wallet, payer, first
    # time that payer appeared). First time is what a "distinct payers before T"
    # count needs, and it collapses a chatty payer to one row.
    edges = (ptx.filter(pl.col("to_address").is_in(link["payout"].unique().to_list()))
             .group_by(["to_address", "from_address"])
             .agg(pl.col("block_time").min().alias("first_paid"))
             .rename({"to_address": "payout", "from_address": "payer"}))
    # How far each fetch actually reached.
    covered = (ptx.group_by("to_address").agg(pl.col("block_time").max().alias("covered_until"))
               .rename({"to_address": "payout"}))
    tr = pl.DataFrame({"payout": list(trunc), "cut": list(trunc.values())})

    probes = ev.join(link, on="destination", how="inner")
    # A join then a filter then a count: distinct payers strictly before the
    # event, excluding the destination's own payments into its payout wallet.
    counted = (probes.select("event_id", "payout", "event_time", "destination")
               .join(edges, on="payout", how="left")
               .filter((pl.col("first_paid") < pl.col("event_time"))
                       & (pl.col("payer") != pl.col("destination")))
               .group_by("event_id").agg(pl.col("payer").n_unique().alias("payout_fanin_pit")))

    out = (probes.join(counted, on="event_id", how="left")
           .join(covered, on="payout", how="left")
           .join(tr, on="payout", how="left")
           .with_columns(
               pl.col("payout_fanin_pit").fill_null(0),
               # Exact when the fetch reached past the moment we are scoring, or
               # when it never hit the cap at all.
               ((~pl.col("cut").fill_null(False))
                | (pl.col("covered_until") >= pl.col("event_time")))
               .cast(pl.Int8).alias("payout_fanin_exact")))
    return out.select("event_id", *NEW)


def arm(train: pl.DataFrame, test: pl.DataFrame, cols: list[str], name: str) -> dict:
    ytr, yte = train["label"].to_numpy(), test["label"].to_numpy()
    b = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 31,
                   "min_data_in_leaf": 30, "feature_fraction": 0.8,
                   "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l2": 1.0,
                   "scale_pos_weight": (ytr == 0).sum() / max((ytr == 1).sum(), 1),
                   "verbose": -1, "seed": 17, "num_threads": 4},
                  lgb.Dataset(train.select(cols).to_numpy(), label=ytr,
                              feature_name=cols), num_boost_round=400)
    s = b.predict(test.select(cols).to_numpy())
    thr = np.quantile(np.sort(s[yte == 0]), 0.99)
    r = {"arm": name, "n_features": len(cols),
         "roc_auc": float(roc_auc_score(yte, s)),
         "pr_auc": float(average_precision_score(yte, s)),
         "tpr_at_1pct_fpr": float((s[yte == 1] >= thr).mean())}
    print(f"  {name:<24}{len(cols):>4} feats  ROC-AUC {r['roc_auc']:.4f}  "
          f"PR-AUC {r['pr_auc']:.4f}  TPR@1%FPR {r['tpr_at_1pct_fpr']:.1%}")
    return r, b


async def main() -> None:
    ev = pl.read_parquet(PROCESSED / "events_features.parquet")
    tf = all_transfers()
    link = top_payees(tf, ev["destination"].unique().to_list())
    payouts = link["payout"].unique().to_list()
    print(f"{ev.height:,} events over {ev['destination'].n_unique():,} destinations; "
          f"{link.height:,} have a payout address, {len(payouts):,} wallets to fetch")

    ptx, trunc = await ingest(payouts)
    cut = sum(1 for a in payouts if trunc.get(a))
    print(f"  {cut:,} of {len(payouts):,} hit the {MAX_PAGES}-page cap "
          f"({cut/max(len(payouts),1):.0%})")

    feats = fanin_features(ev, link, ptx, trunc)
    ev = ev.join(feats, on="event_id", how="left").with_columns(
        pl.col("payout_fanin_pit").fill_null(0),
        pl.col("payout_fanin_exact").fill_null(0))

    sp = temporal_split(ev)
    trained = set(sp.train["destination"].to_list())
    test = sp.test.filter(~pl.col("destination").is_in(list(trained)))
    print(f"\ntrain {sp.train.height:,}  test {test.height:,} "
          f"(destinations the model never saw)")

    for lab, nm in ((1, "scam   "), (0, "control")):
        g = test.filter(pl.col("label") == lab)
        print(f"  {nm} payout_fanin_pit median {g['payout_fanin_pit'].median():>7.0f}  "
              f"p75 {g['payout_fanin_pit'].quantile(.75):>7.0f}  "
              f"exact on {g['payout_fanin_exact'].mean():.0%}")

    print()
    base, _ = arm(sp.train, test, BASE, "shipped 17")
    plus, b = arm(sp.train, test, BASE + NEW, "+ true payout fan-in")

    imp = sorted(zip(BASE + NEW, b.feature_importance("gain")), key=lambda x: -x[1])
    tot = sum(v for _, v in imp) or 1
    print("\n  gain share, top 6:")
    for f, v in imp[:6]:
        print(f"    {f:<26}{v/tot:>7.1%}" + ("   <- new" if f in NEW else ""))

    out = {"max_pages": MAX_PAGES, "payout_wallets": len(payouts), "truncated": cut,
           "arms": [base, plus]}
    (PROCESSED / "payout_fanin_eval.json").write_text(json.dumps(out, indent=2))
    d = plus["tpr_at_1pct_fpr"] - base["tpr_at_1pct_fpr"]
    print(f"\nrecall at the shipped operating point: {d:+.1%}")
    print("wrote data/processed/payout_fanin_eval.json")


asyncio.run(main())

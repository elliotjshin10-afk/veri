"""Is payout_fanin a real signal, or an artefact of which addresses we fetched?

The feature asks a good question: the wallet your money is forwarded to, is it
also collecting from many other fresh addresses? That is a mule network's
signature and it needs no labels. It has been computed all along and excluded
from every model, and the exclusion turns out to be right for a reason nobody
wrote down.

As computed from the warehouse it is a lower bound, because the warehouse holds
histories only for addresses we deliberately fetched and a consolidation wallet
is not one of them. Its inbound edges are therefore only those that happen to
pass through the sample. Measured on the held-out set, 64% of scam destinations
and 59% of controls show a fanin of 1, and the CONTROLS have the heavier tail
(p95 437 against 14). The feature currently separates on which addresses we
chose to fetch, not on what the chain did.

So this fetches the truth for a sample: find each destination's top payout
address, pull that address's own inbound history, and count the distinct wallets
that had paid it before the moment being scored. If the honest version separates
the classes, the feature is worth a full ingest and a serving path. If it does
not, the exclusion stands and we stop paying it attention.

Writes data/processed/payout_fanin_probe.json.
Run: python scripts/m13_payout_fanin.py [n_per_arm]
"""
from __future__ import annotations

import asyncio, json, statistics as st, sys
sys.path.insert(0, "src")

import polars as pl

from veridis.chain.tron import tron_client
from veridis.config import PROCESSED
from veridis.dataset.holdings import all_transfers
from veridis.dataset.ingest import fetch_histories
from veridis.model.train import temporal_split

N_PER_ARM = int(sys.argv[1]) if len(sys.argv) > 1 else 60
MAX_PAGES = 10


def top_payees(tf: pl.DataFrame, dests: list[str]) -> dict[str, str]:
    """Each destination's largest payout address, from the warehouse."""
    out = (tf.filter(pl.col("from_address").is_in(dests) & (pl.col("amount_usd") > 0))
           .group_by(["from_address", "to_address"])
           .agg(pl.col("amount_usd").sum().alias("v"))
           .sort("v", descending=True)
           .group_by("from_address").first())
    return dict(zip(out["from_address"].to_list(), out["to_address"].to_list()))


async def main() -> None:
    ev = pl.read_parquet(PROCESSED / "events_features.parquet")
    sp = temporal_split(ev)
    trained = set(sp.train["destination"].to_list())
    test = sp.test.filter(~pl.col("destination").is_in(list(trained)))

    # One row per destination, with the moment we would have scored it.
    per_dest = (test.sort("event_time")
                .group_by("destination").agg(pl.col("label").first(),
                                             pl.col("event_time").first()))
    arms = {}
    for lab in (1, 0):
        rows = per_dest.filter(pl.col("label") == lab).sample(
            n=min(N_PER_ARM, per_dest.filter(pl.col("label") == lab).height), seed=17)
        arms[lab] = dict(zip(rows["destination"].to_list(), rows["event_time"].to_list()))
    print(f"sample: {len(arms[1])} scam destinations, {len(arms[0])} controls")

    tf = all_transfers()
    payee = top_payees(tf, list(arms[1]) + list(arms[0]))
    print(f"found a payout address for {len(payee)} of "
          f"{len(arms[1]) + len(arms[0])} destinations")

    targets = sorted(set(payee.values()))
    print(f"{len(targets)} distinct consolidation wallets to fetch")
    async with tron_client(concurrency=4) as c:
        ptx, trunc = await fetch_histories(c, targets, max_pages=MAX_PAGES,
                                           concurrency=4, label="payout wallets")

    # Distinct wallets that had paid the consolidation address BEFORE the moment
    # we would have scored the destination. Point in time, like everything else.
    inbound = ptx.filter(pl.col("to_address").is_in(targets)).select(
        "to_address", "from_address", "block_time")
    by_target: dict[str, list[tuple[str, int]]] = {}
    for t, f, bt in inbound.iter_rows():
        by_target.setdefault(t, []).append((f, bt))

    out = {"scam": [], "control": [], "truncated": sum(1 for a in targets if trunc.get(a))}
    for lab, name in ((1, "scam"), (0, "control")):
        for dest, when in arms[lab].items():
            p = payee.get(dest)
            if p is None or p not in by_target:
                continue
            senders = {f for f, bt in by_target[p] if bt < when and f != dest}
            out[name].append({"dest": dest, "payout": p, "fanin": len(senders)})

    print(f"\n{out['truncated']} of {len(targets)} payout histories hit the page cap")
    for name in ("scam", "control"):
        v = sorted(r["fanin"] for r in out[name])
        if not v:
            continue
        q = lambda p: v[min(int(p * len(v)), len(v) - 1)]
        print(f"  {name:<8} n={len(v):<4} median {st.median(v):>7.0f}  "
              f"p75 {q(.75):>7.0f}  p95 {q(.95):>7.0f}  max {v[-1]:>7.0f}  "
              f"share<=1 {sum(1 for x in v if x <= 1)/len(v):.0%}")

    a = [r["fanin"] for r in out["scam"]]
    b = [r["fanin"] for r in out["control"]]
    if a and b:
        # Univariate AUC: P(a random scam scores above a random control).
        wins = sum((x > y) + 0.5 * (x == y) for x in a for y in b)
        auc = wins / (len(a) * len(b))
        print(f"\n  univariate AUC of true payout_fanin: {auc:.3f} "
              f"(0.5 is no signal)")
    (PROCESSED / "payout_fanin_probe.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {PROCESSED / 'payout_fanin_probe.json'}")


asyncio.run(main())

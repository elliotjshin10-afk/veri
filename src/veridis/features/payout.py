"""The second hop: how many other wallets pay the address this one pays.

Every other feature in this project is computed from a single address's own
history. These two need the history of the wallet that address forwards to,
which is why they were excluded for so long: computing them in Python while
defaulting them in the browser is exactly the train/serve skew everything else
here guards against. They are in now because the browser computes them too,
from the same definition, held to this module by scripts/check_parity.py.

The signal runs the opposite way to the name. A scam collection point forwards
to a mule wallet fed by a handful of addresses. An ordinary wallet forwards to
an exchange, and an exchange hot wallet is fed by thousands. So a LOW fan-in is
the warning. Measured on the shipped destination model, worth +2.5pp of recall
at a 1% false-alarm budget, winning on 7 of 7 seeds.

One rule is doing more work than it looks. A truncated history always
understates fan-in, and only busy wallets truncate, so the error is
one-directional: it makes an exchange look like a mule wallet and never the
reverse. `payout_fanin_exact` carries that so the model can tell a real count
from a floor, rather than being handed a silently biased number.
"""
from __future__ import annotations

from typing import Iterable, Mapping

import polars as pl

FEATURES = ["payout_fanin_pit", "payout_fanin_exact"]


def top_payout(transfers: pl.DataFrame, address: str, as_of_ms: int) -> str | None:
    """The address this one pays the most, before `as_of_ms`.

    Ties break on the peer address, ascending. SQL's ROW_NUMBER picks
    arbitrarily among equal sums and the browser cannot reproduce an arbitrary
    pick, so the rule is stated rather than inherited. site/scorer.js breaks
    them the same way.
    """
    out = (transfers.filter((pl.col("from_address") == address)
                            & (pl.col("block_time") < as_of_ms)
                            & (pl.col("amount_usd") > 0))
           .group_by("to_address").agg(pl.col("amount_usd").sum().alias("v"))
           .sort(["v", "to_address"], descending=[True, False]))
    return out["to_address"][0] if out.height else None


def payout_fanin(payout_transfers: pl.DataFrame, payout_address: str | None,
                 as_of_ms: int, self_address: str,
                 truncated: bool) -> dict[str, int]:
    """Distinct wallets that had paid `payout_address` before `as_of_ms`."""
    if payout_address is None or payout_transfers is None:
        return {"payout_fanin_pit": 0, "payout_fanin_exact": 0}
    inbound = payout_transfers.filter((pl.col("to_address") == payout_address)
                                      & (pl.col("amount_usd") > 0))
    if not inbound.height:
        return {"payout_fanin_pit": 0, "payout_fanin_exact": 0 if truncated else 1}
    covered_until = inbound["block_time"].max()
    payers = inbound.filter((pl.col("block_time") < as_of_ms)
                            & (pl.col("from_address") != self_address))
    return {
        "payout_fanin_pit": int(payers["from_address"].n_unique()),
        "payout_fanin_exact": int((not truncated) or covered_until >= as_of_ms),
    }


def links(transfers: pl.DataFrame, addresses: Iterable[str]) -> pl.DataFrame:
    """Every address's top payout wallet over all of its history.

    Used to decide WHICH wallets to fetch, never to compute a feature. The
    bounded form lives in `batch` and in the browser, and the two can disagree
    for an address whose largest payee changed; fetching the union costs one
    extra wallet and keeps the feature honest.
    """
    addrs = list(addresses)
    return (transfers.filter(pl.col("from_address").is_in(addrs)
                             & (pl.col("amount_usd") > 0))
            .group_by(["from_address", "to_address"])
            .agg(pl.col("amount_usd").sum().alias("v"))
            .sort(["v", "to_address"], descending=[True, False])
            .group_by("from_address").first()
            .select(pl.col("from_address").alias("address"),
                    pl.col("to_address").alias("payout")))


def batch(probes: pl.DataFrame, transfers: pl.DataFrame,
          payout_tx: pl.DataFrame, truncated: Mapping[str, bool],
          key: str = "event_id", addr: str = "destination",
          when: str = "event_time") -> pl.DataFrame:
    """Both features for a whole probe table.

    The link is derived PER PROBE, bounded by that probe's moment, because that
    is what the browser does and the two have to agree exactly or the model is
    served a feature it was not trained on. Deriving it once over all history
    would be cheaper and would quietly differ for any address whose largest
    payee changed after the moment being scored.

    Rows whose address has no payout wallet yet, or whose payout wallet was
    never fetched, come back as a zero count and not-exact. That is the honest
    encoding of "we do not know", and it is what the browser reports when its
    second fetch fails.
    """
    p = probes.select(pl.col(key), pl.col(addr).alias("address"),
                      pl.col(when).alias("t"))
    out_legs = (transfers.filter(pl.col("amount_usd") > 0)
                .select(pl.col("from_address").alias("address"),
                        pl.col("to_address").alias("peer"),
                        "amount_usd", "block_time"))
    link = (p.join(out_legs, on="address", how="inner")
            .filter(pl.col("block_time") < pl.col("t"))
            .group_by([key, "peer"]).agg(pl.col("amount_usd").sum().alias("v"))
            .sort(["v", "peer"], descending=[True, False])
            .group_by(key).first()
            .select(key, pl.col("peer").alias("payout")))

    inbound = payout_tx.filter(pl.col("amount_usd") > 0)
    edges = (inbound.group_by(["to_address", "from_address"])
             .agg(pl.col("block_time").min().alias("first_paid"))
             .rename({"to_address": "payout", "from_address": "payer"}))
    covered = (inbound.group_by("to_address")
               .agg(pl.col("block_time").max().alias("covered_until"))
               .rename({"to_address": "payout"}))
    tr = pl.DataFrame({"payout": list(truncated),
                       "cut": [bool(v) for v in truncated.values()]})

    q = p.join(link, on=key, how="left")
    counted = (q.join(edges, on="payout", how="inner")
               .filter((pl.col("first_paid") < pl.col("t"))
                       & (pl.col("payer") != pl.col("address")))
               .group_by(key).agg(pl.col("payer").n_unique()
                                  .alias("payout_fanin_pit")))
    return (q.join(counted, on=key, how="left")
            .join(covered, on="payout", how="left")
            .join(tr, on="payout", how="left")
            .with_columns(
                pl.col("payout_fanin_pit").fill_null(0).cast(pl.Int64),
                pl.when(pl.col("payout").is_null() | pl.col("covered_until").is_null())
                .then(0)
                .otherwise(((~pl.col("cut").fill_null(False))
                            | (pl.col("covered_until") >= pl.col("t"))).cast(pl.Int64))
                .alias("payout_fanin_exact"))
            .select(key, *FEATURES))

"""M3 - the as-of-time feature layer.

The single rule this module exists to enforce: a feature for an event at time
`t` may only read transfers with `block_time < t`. Anything else counts the
victims who arrived later and leaks the label.

Two design choices keep that rule cheap to honour:

1. Every aggregate is expressed as a SQL predicate `x.block_time < e.event_time`.
   There is no "whole history" aggregate anywhere in this file.
2. Per-transfer quantities that would otherwise need a correlated subquery per
   event (sweep gap, running balance) are precomputed *causally* - each row
   looks only at rows strictly before it - so they can then be aggregated under
   the same strict time filter without re-deriving them per event.
"""
from __future__ import annotations

import logging

import duckdb
import polars as pl

log = logging.getLogger(__name__)

DAY_MS = 86_400_000
HOUR_MS = 3_600_000
# Counterparties considered when measuring how connected a sender's world is.
PEER_CAP = 25

# Precompute, per transfer, quantities that only look backward in that
# address's own stream. Safe to aggregate later under a strict time filter.
_CAUSAL_SQL = """
CREATE OR REPLACE TEMP TABLE tx AS
SELECT chain, tx_hash, from_address, to_address, amount_usd, asset, block_time
FROM transfers
WHERE amount_usd IS NOT NULL AND amount_usd > 0;

-- Long form: one row per (address, side) so address-level scans are a filter,
-- not a UNION at query time.
CREATE OR REPLACE TEMP TABLE legs AS
SELECT from_address AS address, 'out' AS side, to_address AS peer,
       amount_usd, block_time, tx_hash FROM tx
UNION ALL
SELECT to_address AS address, 'in' AS side, from_address AS peer,
       amount_usd, block_time, tx_hash FROM tx;

CREATE INDEX IF NOT EXISTS legs_addr ON legs(address, block_time);

-- Causal sweep gap: for each outbound leg, seconds since the most recent
-- inbound leg strictly before it. Looks only backward, so aggregating these
-- under `block_time < t` stays as-of-time correct.
CREATE OR REPLACE TEMP TABLE sweep AS
SELECT address, block_time, amount_usd,
       (block_time - last_value(in_time IGNORE NULLS) OVER w) / 1000.0 AS hold_secs
FROM (
    SELECT address, side, block_time, amount_usd,
           CASE WHEN side = 'in' THEN block_time END AS in_time
    FROM legs
) s
-- Order ties explicitly. Where an inbound and an outbound share a timestamp,
-- an unspecified tie order makes the hold time depend on the engine's sort,
-- which is both non-deterministic and impossible to reproduce in the browser
-- scorer. 'in' sorts before 'out', so money that arrives and leaves in the
-- same block reads as a zero-second hold - which is the sweep behaviour the
-- feature is meant to capture.
WINDOW w AS (PARTITION BY address ORDER BY block_time, side
             ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)
QUALIFY side = 'out';
"""


def _register(con: duckdb.DuckDBPyConnection, transfers: pl.DataFrame) -> None:
    con.register("transfers", transfers.to_arrow())
    con.execute(_CAUSAL_SQL)


# --------------------------------------------------------------- destination
_DEST_SQL = f"""
SELECT
  e.event_id,
  -- age of the destination at scoring time
  (e.event_time - MIN(l.block_time)) / {DAY_MS}.0                        AS dest_age_days,
  COUNT(*) FILTER (WHERE l.side = 'in')                                  AS dest_inbound_count,
  COUNT(*) FILTER (WHERE l.side = 'out')                                 AS dest_outbound_count,
  COUNT(DISTINCT l.peer) FILTER (WHERE l.side = 'in')                    AS dest_senders_all,
  COUNT(DISTINCT l.peer) FILTER (
      WHERE l.side = 'in' AND l.block_time >= e.event_time - {DAY_MS})    AS dest_senders_1d,
  COUNT(DISTINCT l.peer) FILTER (
      WHERE l.side = 'in' AND l.block_time >= e.event_time - {7*DAY_MS})  AS dest_senders_7d,
  COUNT(DISTINCT l.peer) FILTER (
      WHERE l.side = 'in' AND l.block_time >= e.event_time - {30*DAY_MS}) AS dest_senders_30d,
  COALESCE(SUM(l.amount_usd) FILTER (
      WHERE l.side = 'in' AND l.block_time >= e.event_time - {7*DAY_MS}), 0)  AS dest_inbound_usd_7d,
  COALESCE(SUM(l.amount_usd) FILTER (
      WHERE l.side = 'in' AND l.block_time >= e.event_time - {30*DAY_MS}), 0) AS dest_inbound_usd_30d,
  COUNT(*) FILTER (WHERE l.side = 'in'
      AND l.block_time >= e.event_time - {7*DAY_MS})                     AS dest_inbound_count_7d,
  COALESCE(SUM(l.amount_usd) FILTER (WHERE l.side = 'in'), 0)            AS dest_inbound_usd_all,
  COALESCE(SUM(l.amount_usd) FILTER (WHERE l.side = 'out'), 0)           AS dest_outbound_usd_all,
  COUNT(DISTINCT l.peer) FILTER (WHERE l.side = 'out')                   AS dest_distinct_out
FROM events e
JOIN legs l
  ON l.address = e.destination
 AND l.block_time < e.event_time          -- the leakage guard
GROUP BY e.event_id, e.event_time
"""

# Consolidation is a second pass: share of outbound value to the single most
# used destination, again only over pre-event transfers.
_CONSOLIDATION_SQL = f"""
WITH out_legs AS (
  SELECT e.event_id, l.peer, SUM(l.amount_usd) AS v
  FROM events e
  JOIN legs l
    ON l.address = e.destination
   AND l.side = 'out'
   AND l.block_time < e.event_time
  GROUP BY e.event_id, l.peer
)
SELECT event_id,
       MAX(v) / NULLIF(SUM(v), 0) AS dest_consolidation_ratio
FROM out_legs GROUP BY event_id
"""

_SWEEP_SQL = f"""
SELECT e.event_id,
       MEDIAN(s.hold_secs)                          AS dest_median_hold_secs,
       QUANTILE_CONT(s.hold_secs, 0.25)             AS dest_p25_hold_secs
FROM events e
JOIN sweep s
  ON s.address = e.destination
 AND s.block_time < e.event_time
WHERE s.hold_secs IS NOT NULL
GROUP BY e.event_id
"""


# Ring signal: who first funded this destination, and how many *other* fresh
# addresses that same funder has seeded. Scam operations spin up collection
# addresses in batches from a common parent, so a funder with a wide fanout is
# evidence of infrastructure rather than a person. Computed as-of-time: both
# the "first funder" and the fanout are restricted to pre-event transfers.
_FUNDER_SQL = """
WITH first_in AS (
  SELECT e.event_id, e.event_time, l.peer,
         ROW_NUMBER() OVER (PARTITION BY e.event_id ORDER BY l.block_time ASC) AS rn
  FROM events e
  JOIN legs l
    ON l.address = e.destination
   AND l.side = 'in'
   AND l.block_time < e.event_time
),
funder AS (
  SELECT event_id, event_time, peer AS funder_address
  FROM first_in WHERE rn = 1
)
SELECT f.event_id,
       COUNT(DISTINCT l.peer) AS dest_funder_fanout
FROM funder f
JOIN legs l
  ON l.address = f.funder_address
 AND l.side = 'out'
 AND l.block_time < f.event_time
GROUP BY f.event_id
"""


# ---------------------------------------------------------------- graph / 2-hop
#
# COMPUTED BUT DELIBERATELY EXCLUDED FROM THE MODEL. See GRAPH_FEATURES below.
#
# Network features, the way an AML team would build them - but deliberately
# label-free.
#
# A graph feature that measures distance to a *known-bad* address is a blocklist
# wearing a different hat: it inherits every weakness of the list it reads from
# and would quietly invalidate the whole sender-side argument. These use
# structure only, so they work on infrastructure nobody has reported.
#
# The hypothesis was: scam collection addresses are not independent, they sweep
# into shared consolidation wallets, so the address your money is forwarded to
# is usually also collecting from many other fresh addresses. A mule network's
# signature, visible two hops out with no labels.
#
# MEASURED 2026-10-07, and it is backwards. scripts/m13_payout_fanin.py fetched
# the consolidation wallets themselves for 106 held-out destinations rather than
# reading them out of the warehouse, and counted the wallets that had paid each
# one before the scoring moment:
#
#     scam destinations   median fanin  13      controls  median fanin  75
#     univariate AUC 0.324, where 0.5 is no signal
#
# High fan-in at the payout address indicates the OPPOSITE of a mule network.
# An ordinary wallet forwards to an exchange, and an exchange hot wallet is fed
# by thousands of addresses; a scam collection point forwards to a mule wallet
# fed by a handful. The signal is real and worth about 0.68 inverted, but it is
# not the signal this comment claimed, and the quantity in the warehouse is not
# even that: the warehouse holds histories only for addresses we deliberately
# fetched, so a consolidation wallet's inbound edges are only the ones that
# happen to pass through the sample. On the held-out set that artefact makes
# CONTROLS look higher-fanin (p95 437 against 14), which is a measurement of our
# sampling, not of the chain.
#
# So these three columns are computed and deliberately excluded from
# FEATURE_COLUMNS. Using them would train on a confound. Earning them back means
# ingesting the consolidation wallets themselves, and the one arm that was
# tested, adding the warehouse versions to the destination model, moved ROC-AUC
# 0.9234 to 0.9309 and recall at the shipped 1% operating point 23.2% to 23.0%.
_GRAPH_SQL = f"""
WITH payee AS (
  -- The destination's main payout address, as of the event.
  SELECT e.event_id, e.event_time, l.peer AS payout,
         ROW_NUMBER() OVER (PARTITION BY e.event_id
                            ORDER BY SUM(l.amount_usd) DESC) AS rn
  FROM events e
  JOIN legs l
    ON l.address = e.destination AND l.side = 'out'
   AND l.block_time < e.event_time
  GROUP BY e.event_id, e.event_time, l.peer
),
top_payee AS (SELECT event_id, event_time, payout FROM payee WHERE rn = 1)
SELECT
  t.event_id,
  -- How many other addresses feed the same consolidation wallet.
  COUNT(DISTINCT l2.peer)                                        AS payout_fanin,
  COUNT(DISTINCT l2.peer) FILTER (
      WHERE l2.block_time >= t.event_time - {30*DAY_MS})          AS payout_fanin_30d,
  COALESCE(SUM(l2.amount_usd), 0)                                AS payout_inbound_usd
FROM top_payee t
JOIN legs l2
  ON l2.address = t.payout AND l2.side = 'in'
 AND l2.block_time < t.event_time
GROUP BY t.event_id, t.event_time
"""

# Does the sender's world hold together as a graph? A real payment history has
# counterparties that also deal with each other; a wallet that only ever pays
# out in a star has no such structure.
_CLUSTER_SQL = f"""
WITH ranked AS (
  -- Bounded to the sender's most recent {PEER_CAP} counterparties. The pair
  -- join below is quadratic in this number, and an unbounded version does not
  -- finish: a wallet with 500 counterparties would generate 125k pairs for a
  -- single event. Most recent is deterministic and causal, so the cap costs
  -- resolution on very busy wallets rather than correctness.
  SELECT e.event_id, e.event_time, l.peer,
         ROW_NUMBER() OVER (PARTITION BY e.event_id
                            ORDER BY MAX(l.block_time) DESC) AS rn
  FROM events e JOIN legs l
    ON l.address = e.sender AND l.block_time < e.event_time
  GROUP BY e.event_id, e.event_time, l.peer
),
sp AS (SELECT event_id, event_time, peer FROM ranked WHERE rn <= {PEER_CAP}),
linked AS (
  SELECT a.event_id, a.peer AS p1, b.peer AS p2
  FROM sp a
  JOIN sp b ON b.event_id = a.event_id AND b.peer > a.peer
  JOIN legs l
    ON l.address = a.peer AND l.peer = b.peer
   AND l.block_time < a.event_time
)
SELECT sp.event_id,
       COUNT(DISTINCT sp.peer)                    AS sender_peer_count,
       COUNT(DISTINCT l2.p1 || '>' || l2.p2)      AS sender_peer_links
FROM sp LEFT JOIN linked l2 ON l2.event_id = sp.event_id
GROUP BY sp.event_id
"""

# -------------------------------------------------------------------- sender
_SENDER_SQL = f"""
SELECT
  e.event_id,
  (e.event_time - MIN(l.block_time)) / {DAY_MS}.0                        AS sender_age_days,
  COUNT(*) FILTER (WHERE l.side = 'out')                                 AS sender_outbound_count,
  COUNT(DISTINCT l.peer) FILTER (WHERE l.side = 'out')                   AS sender_distinct_dests,
  COUNT(DISTINCT l.peer)                                                 AS sender_distinct_counterparties,
  MEDIAN(l.amount_usd) FILTER (WHERE l.side = 'out')                     AS sender_median_out_usd,
  MAX(l.amount_usd) FILTER (WHERE l.side = 'out')                        AS sender_max_out_usd,
  COALESCE(SUM(l.amount_usd) FILTER (WHERE l.side = 'out'), 0)           AS sender_lifetime_out_usd,
  COALESCE(SUM(l.amount_usd) FILTER (WHERE l.side = 'in'), 0)
    - COALESCE(SUM(l.amount_usd) FILTER (WHERE l.side = 'out'), 0)       AS sender_balance_proxy_usd,
  COUNT(*) FILTER (WHERE l.side = 'out'
      AND l.block_time >= e.event_time - {HOUR_MS})                      AS sender_burst_1h,
  COUNT(*) FILTER (WHERE l.side = 'out'
      AND l.block_time >= e.event_time - {DAY_MS})                       AS sender_burst_24h,
  COUNT(*) FILTER (WHERE l.side = 'out'
      AND l.block_time >= e.event_time - {7*DAY_MS})                     AS sender_burst_7d,
  COALESCE(MAX(l.amount_usd) FILTER (
      WHERE l.side = 'in' AND l.block_time >= e.event_time - {7*DAY_MS}), 0) AS sender_recent_inflow_max_usd,
  COALESCE(SUM(l.amount_usd) FILTER (
      WHERE l.side = 'in' AND l.block_time >= e.event_time - {7*DAY_MS}), 0) AS sender_recent_inflow_usd_7d,
  -- Hour-of-day by arithmetic, NOT EXTRACT(hour FROM to_timestamp(...)).
  -- EXTRACT resolves in the MACHINE's timezone: on this box it returned
  -- America/New_York, so the same transfer produced a different feature value
  -- on a UTC server - latent train/serve skew triggered by where you deploy.
  -- Worse, `event_hour` has always been computed arithmetically in UTC, so
  -- `hour_deviation` was subtracting a New-York hour from a UTC one and the
  -- feature was incoherent even within Python. Caught by the two-sided browser
  -- parity check, which computes it in UTC and disagreed on 119 of 120 sends.
  AVG((l.block_time // 3600000) % 24)
      FILTER (WHERE l.side = 'out')                                      AS sender_mean_out_hour,
  -- Dormancy: time since this sender last sent anything at all. A quiet
  -- account that suddenly moves money is a standard mule / account-takeover
  -- signal in card and ACH fraud modelling.
  (e.event_time - MAX(l.block_time) FILTER (WHERE l.side = 'out')) / 1000.0
                                                                         AS sender_secs_since_last_out,
  -- Own-baseline dispersion, for a z-score of this transfer against the
  -- sender's own history rather than against a population threshold.
  AVG(l.amount_usd) FILTER (WHERE l.side = 'out')                        AS sender_out_mean_usd,
  STDDEV_SAMP(l.amount_usd) FILTER (WHERE l.side = 'out')                AS sender_out_std_usd,
  COALESCE(SUM(l.amount_usd) FILTER (
      WHERE l.side = 'out' AND l.block_time >= e.event_time - 604800000), 0) AS sender_outflow_usd_7d,
  COALESCE(SUM(l.amount_usd) FILTER (
      WHERE l.side = 'out' AND l.block_time >= e.event_time - 2592000000), 0) AS sender_outflow_usd_30d,
  COUNT(DISTINCT l.peer) FILTER (
      WHERE l.side = 'out' AND l.block_time >= e.event_time - 604800000)  AS sender_distinct_dests_7d,
  COUNT(DISTINCT l.peer) FILTER (
      WHERE l.side = 'out' AND l.block_time >= e.event_time - 2592000000) AS sender_distinct_dests_30d,
  -- Median gap between the sender's own transfers, so dormancy can be judged
  -- relative to that sender's normal rhythm instead of an absolute cutoff.
  MEDIAN(l.amount_usd) FILTER (WHERE l.side = 'in')                      AS sender_median_in_usd
FROM events e
JOIN legs l
  ON l.address = e.sender
 AND l.block_time < e.event_time
GROUP BY e.event_id, e.event_time
"""

# ---------------------------------------------------------------- pair / rel
_PAIR_SQL = f"""
SELECT
  e.event_id,
  COUNT(*)                                             AS sender_prior_sends_to_dest,
  MAX(l.amount_usd)                                    AS pair_max_prior_usd,
  MIN(l.amount_usd)                                    AS pair_min_prior_usd,
  ARG_MIN(l.amount_usd, l.block_time)                  AS pair_first_usd,
  SUM(l.amount_usd)                                    AS pair_prior_usd,
  (e.event_time - MAX(l.block_time)) / 1000.0          AS pair_secs_since_last,
  (MAX(l.block_time) - MIN(l.block_time)) / {DAY_MS}.0 AS pair_span_days
FROM events e
JOIN legs l
  ON l.address = e.sender
 AND l.peer = e.destination
 AND l.side = 'out'
 AND l.block_time < e.event_time
GROUP BY e.event_id, e.event_time
"""

# Relationship-graph clustering: has this destination ever transacted with any
# counterparty the sender already deals with? Real relationships cluster;
# a fresh scam collection address does not.
_CONCENTRATION_SQL = f"""
SELECT
  e.event_id,
  COALESCE(SUM(l.amount_usd) FILTER (
      WHERE l.peer = e.destination
        AND l.block_time >= e.event_time - {7*DAY_MS}), 0)  AS pair_outflow_usd_7d,
  COALESCE(SUM(l.amount_usd) FILTER (
      WHERE l.peer = e.destination), 0)                      AS pair_outflow_usd_all
FROM events e
JOIN legs l
  ON l.address = e.sender
 AND l.side = 'out'
 AND l.block_time < e.event_time
GROUP BY e.event_id, e.event_time, e.destination
"""

_SHARED_SQL = """
WITH sender_peers AS (
  SELECT DISTINCT e.event_id, l.peer
  FROM events e JOIN legs l
    ON l.address = e.sender AND l.block_time < e.event_time
),
dest_peers AS (
  SELECT DISTINCT e.event_id, l.peer
  FROM events e JOIN legs l
    ON l.address = e.destination AND l.block_time < e.event_time
)
SELECT sp.event_id, COUNT(*) AS shared_counterparties
FROM sender_peers sp
JOIN dest_peers dp
  ON dp.event_id = sp.event_id AND dp.peer = sp.peer
GROUP BY sp.event_id
"""


def compute_features(events: pl.DataFrame, transfers: pl.DataFrame) -> pl.DataFrame:
    """Compute the full as-of-time feature matrix for `events`.

    `events` needs: event_id, chain, sender, destination, amount_usd, event_time.
    """
    con = duckdb.connect()
    con.execute("PRAGMA threads=4")
    _register(con, transfers)
    con.register("events", events.to_arrow())

    parts = {
        "dest": _DEST_SQL,
        "cons": _CONSOLIDATION_SQL,
        "sweep": _SWEEP_SQL,
        "sender": _SENDER_SQL,
        "pair": _PAIR_SQL,
        "shared": _SHARED_SQL,
        "funder": _FUNDER_SQL,
        "conc": _CONCENTRATION_SQL,
        "graph": _GRAPH_SQL,
        "cluster": _CLUSTER_SQL,
    }
    out = events
    for name, sql in parts.items():
        df = con.execute(sql).pl()
        log.info("  feature block %-7s -> %d rows, %d cols", name, df.height, df.width)
        out = out.join(df, on="event_id", how="left")
    con.close()
    return _derive(out)


def _derive(df: pl.DataFrame) -> pl.DataFrame:
    """Ratios and flags built from the raw as-of aggregates."""
    zero_fill = [
        "sender_prior_sends_to_dest", "dest_inbound_count", "dest_outbound_count",
        "dest_senders_1d", "dest_senders_7d", "dest_senders_30d", "dest_senders_all",
        "dest_inbound_count_7d", "dest_distinct_out", "sender_burst_1h",
        "sender_burst_24h", "sender_burst_7d", "sender_outbound_count",
        "sender_distinct_dests", "sender_distinct_counterparties",
        "shared_counterparties", "dest_funder_fanout", "pair_prior_usd", "dest_inbound_usd_7d",
        "dest_inbound_usd_30d", "sender_lifetime_out_usd",
        "sender_recent_inflow_usd_7d", "sender_recent_inflow_max_usd",
    ]
    df = df.with_columns([pl.col(c).fill_null(0) for c in zero_fill if c in df.columns])

    eps = 1e-9
    return df.with_columns(
        # The two features the brief expects to dominate.
        (pl.col("amount_usd") / (pl.col("pair_max_prior_usd") + eps))
            .fill_null(-1.0).alias("escalation_ratio"),
        # Escalation measured against the *first* transfer to this destination,
        # not the running maximum. The pattern is "a small test transfer, then
        # much larger ones": against the running max that ratio drops below 1
        # as soon as the peak is passed, so over a whole sequence its median is
        # ~0.48 and it carries almost no signal (univariate AUC 0.47). Against
        # the opening transfer it stays elevated for the rest of the sequence,
        # which is what the hypothesis actually claims.
        (pl.col("amount_usd") / (pl.col("pair_first_usd") + eps))
            .fill_null(-1.0).alias("escalation_vs_first"),
        (pl.col("pair_max_prior_usd") / (pl.col("pair_first_usd") + eps))
            .fill_null(-1.0).alias("pair_peak_vs_first"),
        (pl.col("amount_usd") / (pl.col("sender_median_out_usd") + eps))
            .fill_null(-1.0).alias("amount_vs_sender_median"),
        (pl.col("amount_usd") / (pl.col("sender_max_out_usd") + eps))
            .fill_null(-1.0).alias("amount_vs_sender_max"),
        (pl.col("amount_usd")
            / (pl.col("sender_balance_proxy_usd").clip(lower_bound=0) + eps))
            .fill_null(-1.0).alias("amount_vs_balance"),
        (pl.col("amount_usd") / (pl.col("sender_lifetime_out_usd") + eps))
            .alias("amount_share_of_lifetime_out"),
        (pl.col("dest_outbound_usd_all") / (pl.col("dest_inbound_usd_all") + eps))
            .alias("dest_forward_ratio"),
        (pl.col("dest_inbound_count") / (pl.col("dest_outbound_count") + eps))
            .alias("dest_in_out_ratio"),
        (pl.col("dest_inbound_usd_all") / (pl.col("dest_senders_all") + eps))
            .alias("dest_usd_per_sender"),
        (pl.col("sender_prior_sends_to_dest") == 0).cast(pl.Int8)
            .alias("is_first_send_to_dest"),
        ((pl.col("sender_prior_sends_to_dest") == 0)
            & (pl.col("dest_age_days").fill_null(0) < 30)).cast(pl.Int8)
            .alias("first_send_to_young_dest"),
        (pl.col("shared_counterparties") == 0).cast(pl.Int8)
            .alias("no_shared_counterparties"),
        # Mule-network shape: many fresh addresses feeding one payout wallet.
        pl.col("payout_fanin").fill_null(0),
        pl.col("payout_fanin_30d").fill_null(0),
        pl.col("payout_inbound_usd").fill_null(0.0),
        # Clustering coefficient of the sender's own counterparty graph.
        (pl.col("sender_peer_links").fill_null(0)
            / ((pl.col("sender_peer_count").fill_null(0)
                * (pl.col("sender_peer_count").fill_null(0) - 1) / 2) + 1e-9))
            .clip(0.0, 1.0).alias("sender_graph_clustering"),
        pl.col("sender_peer_links").fill_null(0),
        pl.col("dest_age_days").fill_null(0.0),
        pl.col("sender_age_days").fill_null(0.0),
        pl.col("dest_consolidation_ratio").fill_null(0.0),
        pl.col("dest_median_hold_secs").fill_null(-1.0),
        pl.col("dest_p25_hold_secs").fill_null(-1.0),
        pl.col("pair_secs_since_last").fill_null(-1.0),
        pl.col("pair_span_days").fill_null(-1.0),
        pl.col("sender_median_out_usd").fill_null(0.0),
        pl.col("sender_max_out_usd").fill_null(0.0),
        pl.col("sender_balance_proxy_usd").fill_null(0.0),
        pl.col("sender_mean_out_hour").fill_null(-1.0),
    ).with_columns(
        # ---- behavioural-fraud-modelling derivations (all sender-side) ----
        # Z-score of this transfer against the sender's OWN history. Card-fraud
        # practice scores deviation from the individual's baseline rather than
        # against a population threshold, because "large" only means anything
        # relative to what this person normally does.
        ((pl.col("amount_usd") - pl.col("sender_out_mean_usd"))
            / (pl.col("sender_out_std_usd") + 1.0))
            .fill_null(0.0).alias("amount_zscore_own"),
        # Dormancy relative to the sender's own rhythm: a wallet that normally
        # transacts daily going quiet for a month and then moving money is a
        # different event from one that is always sporadic.
        (pl.col("sender_secs_since_last_out") / 86400.0)
            .fill_null(-1.0).alias("sender_dormancy_days"),
        # Share of the last 7 days of outflow going to this single destination.
        # "All recent funds to one counterparty" is the classic mule/APP shape.
        (pl.col("pair_outflow_usd_7d") / (pl.col("sender_outflow_usd_7d") + 1e-9))
            .fill_null(0.0).clip(0.0, 1.0).alias("pair_share_of_outflow_7d"),
        (pl.col("pair_outflow_usd_all") / (pl.col("sender_lifetime_out_usd") + 1e-9))
            .fill_null(0.0).clip(0.0, 1.0).alias("pair_share_of_outflow_all"),
        # Drawdown: how much of what the wallet holds is leaving.
        (pl.col("sender_outflow_usd_7d")
            / (pl.col("sender_balance_proxy_usd").clip(lower_bound=0) + 1e-9))
            .fill_null(0.0).alias("sender_drawdown_7d"),
        # Counterparty churn: are recent transfers spread across the usual
        # payees, or converging on very few?
        (pl.col("sender_distinct_dests_7d")
            / (pl.col("sender_burst_7d") + 1e-9))
            .fill_null(0.0).alias("sender_dest_churn_7d"),
        (pl.col("sender_distinct_dests_30d")
            / (pl.col("sender_distinct_counterparties") + 1e-9))
            .fill_null(0.0).alias("sender_recent_dest_share"),
        # Round-number amounts. Coached transfers are dictated in round figures
        # ("send 5,000"); organic commerce is not.
        (pl.col("amount_usd") % 1000 == 0).cast(pl.Int8).alias("amount_round_1k"),
        (pl.col("amount_usd") % 100 == 0).cast(pl.Int8).alias("amount_round_100"),
        # On-ramp then out: money arrives and leaves, rather than being held.
        (pl.col("sender_outflow_usd_7d")
            / (pl.col("sender_recent_inflow_usd_7d") + 1e-9))
            .fill_null(0.0).clip(0.0, 50.0).alias("sender_passthrough_7d"),
        # ---- trajectory / acceleration (pure sender behaviour) ----
        # Fraud teams look at whether someone's activity is *accelerating*, not
        # just whether today is large. A wallet under live coaching moves from
        # its normal cadence to a much faster one over days, which shows up as
        # a recent-rate-over-baseline-rate ratio well above 1 even when no
        # single transfer is remarkable.
        ((pl.col("sender_outflow_usd_7d") / 7.0)
            / ((pl.col("sender_outflow_usd_30d") / 30.0) + 1e-9))
            .fill_null(0.0).clip(0.0, 100.0).alias("sender_outflow_accel"),
        (pl.col("sender_burst_24h").cast(pl.Float64)
            / ((pl.col("sender_burst_7d").cast(pl.Float64) / 7.0) + 1e-9))
            .fill_null(0.0).clip(0.0, 100.0).alias("sender_burst_accel"),
        # Lifetime cadence, and how far the last week departs from it.
        (pl.col("sender_outbound_count").cast(pl.Float64)
            / (pl.col("sender_age_days") + 1.0))
            .fill_null(0.0).alias("sender_lifetime_rate"),
        ((pl.col("sender_burst_7d").cast(pl.Float64) / 7.0)
            / ((pl.col("sender_outbound_count").cast(pl.Float64)
                / (pl.col("sender_age_days") + 1.0)) + 1e-9))
            .fill_null(0.0).clip(0.0, 200.0).alias("sender_rate_vs_lifetime"),
        # Value cadence: is the money moving faster than the count suggests?
        ((pl.col("sender_outflow_usd_7d") / 7.0)
            / ((pl.col("sender_lifetime_out_usd")
                / (pl.col("sender_age_days") + 1.0)) + 1e-9))
            .fill_null(0.0).clip(0.0, 500.0).alias("sender_value_rate_vs_lifetime"),
        # Inflow acceleration: victims top up to keep sending.
        (pl.col("sender_recent_inflow_usd_7d")
            / (pl.col("sender_outflow_usd_7d") + 1e-9))
            .fill_null(0.0).clip(0.0, 50.0).alias("sender_inflow_cover_7d"),
        pl.col("sender_out_mean_usd").fill_null(0.0),
        pl.col("sender_out_std_usd").fill_null(0.0),
        pl.col("sender_median_in_usd").fill_null(0.0),
        pl.col("sender_outflow_usd_7d").fill_null(0.0),
        pl.col("sender_outflow_usd_30d").fill_null(0.0),
        pl.col("sender_distinct_dests_7d").fill_null(0),
        pl.col("sender_distinct_dests_30d").fill_null(0),
        pl.col("pair_outflow_usd_7d").fill_null(0.0),
        pl.col("pair_outflow_usd_all").fill_null(0.0),
    ).with_columns(
        (pl.col("event_time").cast(pl.Int64) // 3_600_000 % 24)
            .cast(pl.Float64).alias("event_hour"),
    ).with_columns(
        (pl.col("event_hour") - pl.col("sender_mean_out_hour")).abs()
            .alias("hour_deviation"),
    )


# Columns the model is allowed to see. Anything not listed here (ids, raw
# addresses, timestamps, labels) is excluded by construction.
FEATURE_COLUMNS = [
    "amount_usd",
    "dest_age_days", "dest_inbound_count", "dest_outbound_count",
    "dest_senders_all", "dest_senders_1d", "dest_senders_7d", "dest_senders_30d",
    "dest_inbound_usd_7d", "dest_inbound_usd_30d", "dest_inbound_count_7d",
    "dest_distinct_out", "dest_consolidation_ratio", "dest_median_hold_secs",
    "dest_p25_hold_secs", "dest_forward_ratio", "dest_in_out_ratio",
    "dest_usd_per_sender", "dest_funder_fanout",
    "sender_age_days", "sender_outbound_count", "sender_distinct_dests",
    "sender_distinct_counterparties", "sender_median_out_usd",
    "sender_max_out_usd", "sender_lifetime_out_usd", "sender_balance_proxy_usd",
    "sender_burst_1h", "sender_burst_24h", "sender_burst_7d",
    "sender_recent_inflow_max_usd", "sender_recent_inflow_usd_7d",
    "sender_prior_sends_to_dest", "pair_prior_usd", "pair_secs_since_last",
    "pair_span_days",
    "escalation_ratio", "escalation_vs_first", "pair_peak_vs_first",
    "amount_vs_sender_median", "amount_vs_sender_max",
    "amount_vs_balance", "amount_share_of_lifetime_out",
    "is_first_send_to_dest", "first_send_to_young_dest",
    "shared_counterparties", "no_shared_counterparties",
    "event_hour", "hour_deviation",
    # behavioural fraud-modelling block
    "amount_zscore_own", "sender_dormancy_days", "sender_secs_since_last_out",
    "sender_drawdown_7d", "sender_dest_churn_7d", "sender_recent_dest_share",
    "sender_passthrough_7d", "sender_out_mean_usd", "sender_out_std_usd",
    "sender_median_in_usd", "sender_outflow_usd_7d", "sender_outflow_usd_30d",
    "sender_distinct_dests_7d", "sender_distinct_dests_30d",
    "pair_share_of_outflow_7d", "pair_share_of_outflow_all",
    "pair_outflow_usd_7d", "pair_outflow_usd_all",
    "amount_round_1k", "amount_round_100",
    # trajectory block
    "sender_outflow_accel", "sender_burst_accel", "sender_lifetime_rate",
    "sender_rate_vs_lifetime", "sender_value_rate_vs_lifetime",
    "sender_inflow_cover_7d",
]

# Which family each feature belongs to. Stated explicitly rather than inferred
# from name prefixes, because the brief's central failure check depends on it:
# if sender-behaviour and relationship features carry no weight, we have
# rebuilt a blocklist with extra steps. "amount vs the sender's own baseline"
# is sender behaviour, not a property of the destination, and classifying it by
# prefix would have filed it under the wrong family.
_DESTINATION = [
    "dest_age_days", "dest_inbound_count", "dest_outbound_count",
    "dest_senders_all", "dest_senders_1d", "dest_senders_7d", "dest_senders_30d",
    "dest_inbound_usd_7d", "dest_inbound_usd_30d", "dest_inbound_count_7d",
    "dest_distinct_out", "dest_consolidation_ratio", "dest_median_hold_secs",
    "dest_p25_hold_secs", "dest_forward_ratio", "dest_in_out_ratio",
    "dest_usd_per_sender", "dest_funder_fanout",
]
_SENDER = [
    "sender_outflow_accel", "sender_burst_accel", "sender_lifetime_rate",
    "sender_rate_vs_lifetime", "sender_value_rate_vs_lifetime",
    "sender_inflow_cover_7d",
    "amount_zscore_own", "sender_dormancy_days", "sender_drawdown_7d",
    "sender_dest_churn_7d", "sender_recent_dest_share", "sender_passthrough_7d",
    "sender_out_mean_usd", "sender_out_std_usd", "sender_median_in_usd",
    "sender_outflow_usd_7d", "sender_outflow_usd_30d",
    "sender_distinct_dests_7d", "sender_distinct_dests_30d",
    "sender_secs_since_last_out",
    "sender_age_days", "sender_outbound_count", "sender_distinct_dests",
    "sender_distinct_counterparties", "sender_median_out_usd",
    "sender_max_out_usd", "sender_lifetime_out_usd", "sender_balance_proxy_usd",
    "sender_burst_1h", "sender_burst_24h", "sender_burst_7d",
    "sender_recent_inflow_max_usd", "sender_recent_inflow_usd_7d",
    # deviation from the sender's own historical baseline
    "amount_vs_sender_median", "amount_vs_sender_max", "amount_vs_balance",
    "amount_share_of_lifetime_out",
]
_RELATIONSHIP = [
    "pair_share_of_outflow_7d", "pair_share_of_outflow_all",
    "pair_outflow_usd_7d", "pair_outflow_usd_all",
    "sender_prior_sends_to_dest", "pair_prior_usd", "pair_secs_since_last",
    "pair_span_days", "escalation_ratio", "escalation_vs_first",
    "pair_peak_vs_first", "is_first_send_to_dest",
    "first_send_to_young_dest", "shared_counterparties",
    "no_shared_counterparties",
]
_CONTEXT = ["amount_usd", "event_hour", "hour_deviation",
            "amount_round_1k", "amount_round_100"]

# Computed, measured, and kept out of the model on purpose.
#
# These are the standard AML mule-network features and the idea is sound: scam
# collection addresses sweep into shared consolidation wallets, so the payout
# wallet is usually collecting from many other fresh addresses too. On this
# dataset they cannot measure that, because the second hop is not in the
# warehouse - only 34% of payout wallets have their own history fetched, so
# `payout_fanin` largely counts how much of that wallet we happened to collect.
#
# Worse, the artefact aligns with the label: the control population was built
# as an interconnected two-hop subgraph, so its counterparties really are more
# linked to each other (median clustering 0.0100 against 0.0033 for victims) -
# a property of our sampling, not of the wallets. A model given these would
# partly learn how an address entered our dataset.
#
# They become valid the moment the two-hop neighbourhood is fetched, which is a
# data-collection job rather than a modelling one. Until then they stay out.
GRAPH_FEATURES = [
    "payout_fanin", "payout_fanin_30d", "payout_inbound_usd",
    "sender_graph_clustering", "sender_peer_links",
]

FEATURE_FAMILY: dict[str, str] = {
    **{c: "destination" for c in _DESTINATION},
    **{c: "sender" for c in _SENDER},
    **{c: "relationship" for c in _RELATIONSHIP},
    **{c: "context" for c in _CONTEXT},
}

assert set(FEATURE_FAMILY) == set(FEATURE_COLUMNS), (
    "every feature must be assigned a family: "
    f"missing={set(FEATURE_COLUMNS) - set(FEATURE_FAMILY)}, "
    f"unknown={set(FEATURE_FAMILY) - set(FEATURE_COLUMNS)}"
)


class FeatureEngine:
    """Warm feature engine: builds the causal tables once, scores many times.

    Batch training and the live API share this exact code path. That matters
    more than raw speed - a serving path that computes features differently
    from the training path is the other classic way this product silently
    breaks.
    """

    def __init__(self, transfers: pl.DataFrame) -> None:
        self.con = duckdb.connect()
        self.con.execute("PRAGMA threads=4")
        _register(self.con, transfers)
        self.n_transfers = transfers.height

    def compute(self, events: pl.DataFrame) -> pl.DataFrame:
        self.con.register("events", events.to_arrow())
        out = events
        for sql in (_DEST_SQL, _CONSOLIDATION_SQL, _SWEEP_SQL,
                    _SENDER_SQL, _PAIR_SQL, _SHARED_SQL, _FUNDER_SQL,
                    _CONCENTRATION_SQL, _GRAPH_SQL, _CLUSTER_SQL):
            out = out.join(self.con.execute(sql).pl(), on="event_id", how="left")
        self.con.unregister("events")
        return _derive(out)

    def close(self) -> None:
        self.con.close()

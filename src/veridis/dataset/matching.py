"""M2 - stratified control matching.

Without matching, a model trained on this data learns "large transfer to a new
address = scam" and fires on every legitimate big payment. Controls are matched
to positives on sender account-age, activity level and typical transfer size,
all measured as-of the event, so the model has to find behaviour rather than
size. Performance is then reported at the matched ratio, never rebalanced.
"""
from __future__ import annotations

import logging

import polars as pl

from veridis.config import CONTROL_RATIO, SEED

log = logging.getLogger(__name__)

AGE_BINS = [7.0, 30.0, 90.0, 365.0]
COUNT_BINS = [1.0, 3.0, 10.0, 30.0, 100.0]
SIZE_BINS = [50.0, 200.0, 1_000.0, 5_000.0]


def add_strata(df: pl.DataFrame) -> pl.DataFrame:
    """Bucket each event on the sender's as-of-time profile."""
    return df.with_columns(
        pl.col("sender_age_days").fill_null(0).cut(AGE_BINS, labels=[f"a{i}" for i in range(len(AGE_BINS) + 1)]).cast(pl.Utf8).alias("s_age"),
        pl.col("sender_outbound_count").fill_null(0).cast(pl.Float64).cut(COUNT_BINS, labels=[f"c{i}" for i in range(len(COUNT_BINS) + 1)]).cast(pl.Utf8).alias("s_count"),
        pl.col("sender_median_out_usd").fill_null(0).cut(SIZE_BINS, labels=[f"s{i}" for i in range(len(SIZE_BINS) + 1)]).cast(pl.Utf8).alias("s_size"),
    ).with_columns(
        (pl.col("chain") + "|" + pl.col("s_age") + "|" + pl.col("s_count") + "|" + pl.col("s_size")).alias("stratum")
    )


def match_controls(
    events: pl.DataFrame, ratio: int = CONTROL_RATIO, seed: int = SEED
) -> pl.DataFrame:
    """Keep all positives; sample negatives to `ratio`:1, backing off as needed.

    Exact matching on (chain, age, activity, size) can starve: if victims and
    ordinary senders occupy different corners of the covariate space, some
    strata have no controls at all, and a stratum with 0 controls contributes
    positives the model can separate for free. Rather than silently accepting
    that, we back off to progressively coarser strata and record the level each
    control was matched at, so the cost is visible in the report.
    """
    events = add_strata(events)
    pos = events.filter(pl.col("label") == 1)
    neg = events.filter(pl.col("label") == 0)
    if pos.height == 0 or neg.height == 0:
        return events

    levels = [
        ("stratum_l1", ["chain", "s_age", "s_count", "s_size"]),
        ("stratum_l2", ["chain", "s_age", "s_count"]),
        ("stratum_l3", ["chain", "s_age"]),
        ("stratum_l4", ["chain"]),
    ]
    for name, cols in levels:
        expr = pl.col(cols[0])
        for c in cols[1:]:
            expr = expr + "|" + pl.col(c)
        pos = pos.with_columns(expr.alias(name))
        neg = neg.with_columns(expr.alias(name))

    # Internal row id so matching does not depend on an event_id column.
    neg = neg.with_row_index("_nid")
    taken: list[pl.DataFrame] = []
    used: set[int] = set()
    # How many controls each positive stratum still needs, level by level.
    need = (
        pos.group_by("stratum").len().rename({"len": "n_pos"})
        .with_columns((pl.col("n_pos") * ratio).alias("n_want"))
    )
    remaining = need
    rng_seed = seed
    for name, _cols in levels:
        if remaining.height == 0:
            break
        # Map each positive stratum to its coarser key at this level.
        keymap = pos.select(["stratum", name]).unique()
        want = remaining.join(keymap, on="stratum", how="left")
        want_by_key = want.group_by(name).agg(pl.col("n_want").sum().alias("n_want"))

        pool = neg.filter(~pl.col("_nid").is_in(used)) if used else neg
        if pool.height == 0:
            break
        rng_seed += 1
        picked = (
            pool.join(want_by_key, on=name, how="inner")
            .with_columns(
                pl.int_range(pl.len()).shuffle(seed=rng_seed).over(name).alias("_r")
            )
            .filter(pl.col("_r") < pl.col("n_want"))
            .drop("_r", "n_want")
            .with_columns(pl.lit(name).alias("match_level"))
        )
        if picked.height:
            used |= set(picked["_nid"].to_list())
            taken.append(picked)
            got = (
                picked.group_by(name).len().rename({"len": "got"})
            )
            remaining = (
                want.join(got, on=name, how="left")
                .with_columns(pl.col("got").fill_null(0))
                .with_columns((pl.col("n_want") - pl.col("got")).alias("n_want"))
                .filter(pl.col("n_want") > 0)
                .select(["stratum", "n_pos", "n_want"])
            )
        else:
            remaining = want.select(["stratum", "n_pos", "n_want"])

    drop_cols = [n for n, _ in levels] + ["_nid"]
    sampled = (
        pl.concat(taken, how="vertical_relaxed") if taken else neg.head(0)
    )
    pos = pos.with_columns(pl.lit("positive").alias("match_level"))
    out = pl.concat(
        [pos.drop(drop_cols, strict=False), sampled.drop(drop_cols, strict=False)],
        how="vertical_relaxed",
    )
    _report(pos, sampled)
    if sampled.height:
        log.info("control match levels: %s",
                 sampled.group_by("match_level").len().sort("len", descending=True).to_dicts())
    return out


def _report(pos: pl.DataFrame, neg: pl.DataFrame) -> None:
    ratio = neg.height / max(pos.height, 1)
    log.info(
        "matched controls: %d positives, %d negatives (%.1f:1)",
        pos.height, neg.height, ratio,
    )
    cov = (
        pos.group_by("stratum").len().rename({"len": "pos"})
        .join(neg.group_by("stratum").len().rename({"len": "neg"}),
              on="stratum", how="left")
        .with_columns(pl.col("neg").fill_null(0))
        .with_columns((pl.col("neg") / pl.col("pos")).alias("achieved"))
        .sort("pos", descending=True)
    )
    starved = cov.filter(pl.col("achieved") < 1.0)
    if starved.height:
        log.warning(
            "%d strata have fewer than 1:1 controls - matching is thin there",
            starved.height,
        )


def balance_report(events: pl.DataFrame) -> pl.DataFrame:
    """Covariate balance between positives and matched negatives.

    If matching worked, the medians of the matching covariates should be close;
    a large gap in `amount_usd` is expected and is the signal, not a flaw.
    """
    cols = ["sender_age_days", "sender_outbound_count", "sender_median_out_usd", "amount_usd"]
    return (
        events.group_by("label")
        .agg([pl.col(c).median().alias(c) for c in cols] + [pl.len().alias("n")])
        .sort("label")
    )


# --------------------------------------------------------------- NN matching
# Coarse bins cannot fix a control pool drawn from a different part of the
# covariate space. Sampling the firehose samples *transfers*, which is
# size-biased: a random transfer is far more likely to come from a high-volume
# sender, while victims are ordinary retail wallets. Nearest-neighbour matching
# on the standardised covariates picks, for each positive, the controls that
# actually resemble it - optimising balance directly instead of hoping the bins
# line up.
MATCH_COVARIATES = [
    "sender_age_days",
    "sender_outbound_count",
    "sender_median_out_usd",
]


def _covariate_matrix(df: pl.DataFrame) -> "np.ndarray":
    import numpy as np

    cols = []
    for c in MATCH_COVARIATES:
        v = df[c].fill_null(0).to_numpy().astype(float)
        cols.append(np.log1p(np.clip(v, 0, None)))  # heavy-tailed -> log scale
    return np.column_stack(cols)


# Match exactly on this, never approximately.
#
# Control events are drawn from a two-hop subgraph of addresses that transact
# with each other, which selects hard for *repeat* payment pairs: unmatched,
# controls carried a median of 19 prior sends to the same destination against
# 1 for victims. "Have you sent here before" then becomes the dominant feature
# and the model is partly reading control construction rather than behaviour -
# and real-world FPR would be understated, because a legitimate first-time
# send to a new address is exactly what would get caught.
EXACT_MATCH_ON = "is_first_send_to_dest"


def match_controls_nn(
    events: pl.DataFrame, ratio: int = CONTROL_RATIO, seed: int = SEED
) -> pl.DataFrame:
    """Match `ratio` nearest controls per positive, without replacement.

    Matching is exact on `EXACT_MATCH_ON` and nearest-neighbour on the
    continuous covariates within each exact stratum.
    """
    import numpy as np

    events = add_strata(events)
    if EXACT_MATCH_ON in events.columns:
        parts = []
        for (key,), grp in events.group_by([EXACT_MATCH_ON], maintain_order=True):
            picked = _nn_within(grp, ratio, seed + int(key or 0))
            if picked is not None:
                parts.append(picked)
        if parts:
            out = pl.concat(parts, how="vertical_relaxed")
            p_, n_ = out.filter(pl.col("label") == 1), out.filter(pl.col("label") == 0)
            _report(p_, n_)
            log.info(
                "exact-matched on %s: %s",
                EXACT_MATCH_ON,
                out.group_by([EXACT_MATCH_ON, "label"]).len()
                   .sort([EXACT_MATCH_ON, "label"]).to_dicts(),
            )
            return out
    return _nn_within(events, ratio, seed) or events


def _nn_within(
    events: pl.DataFrame, ratio: int, seed: int
) -> pl.DataFrame | None:
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    pos = events.filter(pl.col("label") == 1)
    neg = events.filter(pl.col("label") == 0)
    if pos.height == 0 or neg.height == 0:
        return None

    P, N = _covariate_matrix(pos), _covariate_matrix(neg)
    mu, sigma = N.mean(axis=0), N.std(axis=0)
    sigma[sigma == 0] = 1.0
    P, N = (P - mu) / sigma, (N - mu) / sigma

    # Ask for many more neighbours than needed. Matching is without
    # replacement, so early positives claim the obvious candidates and later
    # ones are left searching; too small a k makes them give up while usable
    # controls are still unused, which costs ratio (and therefore FPR
    # resolution) for no gain in balance.
    k = int(min(neg.height, max(ratio * 20, 200)))
    nn = NearestNeighbors(n_neighbors=k).fit(N)
    _, idx = nn.kneighbors(P)

    used: set[int] = set()
    chosen: list[int] = []
    order = np.random.default_rng(seed).permutation(pos.height)
    for i in order:
        taken = 0
        for j in idx[i]:
            j = int(j)
            if j in used:
                continue
            used.add(j)
            chosen.append(j)
            taken += 1
            if taken >= ratio:
                break

    sampled = neg[chosen].with_columns(pl.lit("nearest_neighbour").alias("match_level"))
    pos = pos.with_columns(pl.lit("positive").alias("match_level"))
    drop = ["stratum_l1", "stratum_l2", "stratum_l3", "stratum_l4"]
    return pl.concat(
        [pos.drop(drop, strict=False), sampled.drop(drop, strict=False)],
        how="vertical_relaxed",
    )


def standardised_mean_difference(events: pl.DataFrame) -> pl.DataFrame:
    """Balance diagnostic. |SMD| < 0.1 is the usual 'balanced' threshold."""
    import numpy as np

    pos = events.filter(pl.col("label") == 1)
    neg = events.filter(pl.col("label") == 0)
    rows = []
    for c in MATCH_COVARIATES + ["amount_usd"]:
        a = np.log1p(np.clip(pos[c].fill_null(0).to_numpy().astype(float), 0, None))
        b = np.log1p(np.clip(neg[c].fill_null(0).to_numpy().astype(float), 0, None))
        sd = np.sqrt((a.var() + b.var()) / 2) or 1.0
        rows.append({
            "covariate": c,
            "pos_median": float(pos[c].median() or 0),
            "neg_median": float(neg[c].median() or 0),
            "smd": float((a.mean() - b.mean()) / sd),
        })
    return pl.DataFrame(rows).with_columns(pl.col("smd").abs().alias("abs_smd"))

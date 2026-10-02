# Which chains this works on, and why not the others

Short version: Tron and Ethereum. Not Arbitrum, not Polygon, not Bitcoin — and
the reason is the same each time. The model is graded against an on-chain freeze
list. No freeze list that reflects *that chain's* behaviour means nothing to
train on and, worse, no way to check our own answers.

Every figure below is measured, and the queries are reproducible with the
Etherscan V2 key the ingest already uses.

## Tron and Ethereum

Tether freezes addresses on both when law enforcement asks, and publishes it
on-chain as `AddedBlackList(address)`. That is the closest public proxy for
where stablecoin fraud actually lands.

| chain | model | held-out ROC-AUC |
|---|---|---|
| Tron | destination | 0.955 |
| Tron | two-sided | 0.975 |
| Ethereum | destination | 0.932 |
| Ethereum | two-sided | 0.872 |

87% of the addresses Tether froze for fraud in 2026 are on Tron, up from 39% in
2023. The money is where the model is.

## Arbitrum and Polygon: labels exist, but they are not about those chains

Tether does not blacklist on either — `AddedBlackList` returns **0 events** on
both. Circle does, on native USDC, and at first glance it looks workable:

| chain · issuer | blacklisted addresses | first |
|---|---|---|
| USDC Ethereum | 846 | 2020-06 |
| USDC Arbitrum | 582 | 2023-06 |
| USDC Polygon | 576 | 2023-07 |
| USDT Arbitrum | 0 | — |
| USDT Polygon | 0 | — |

The problem is what those 582 are. Circle blacklists an address across every
chain it issues on, wherever the address actually offended:

    of Arbitrum's 582 blacklisted addresses
      573 (98%) are also blacklisted on Ethereum
      571 (98%) are also blacklisted on Polygon
        9       are Arbitrum-only

So the list is replicated, not earned. Checking whether those addresses ever
touched Arbitrum settles it — 60 sampled at random:

      2 of 60 (3%) have ANY Arbitrum stablecoin history
     76 transfers across all 60, of which 40 inbound
     -> about 19 usable addresses across the whole 582

Nineteen. That is not a thin training set, it is not a training set. It is not
a validation set either: the model is scored on a destination's inbound
behaviour, and these addresses have essentially no inbound behaviour on
Arbitrum to score.

Note this is a *stronger* reason to stay off those chains than "no labels". A
blocklist that is replicated across chains tells you an address is bad
somewhere; it tells you nothing about what it did on the chain you are asked
about, which is exactly the question a pre-send check has to answer.

## Why not simply reuse the Ethereum model there

Because models do not transfer across chains, and we measured it rather than
assuming. `data/processed/crosschain_report.json`: the Tron model applied to
Ethereum unchanged — no retraining, no recalibration, the Tron operating point
as-is — scored

    ROC-AUC 0.682   against Tron's own 0.947
    recall  10.2%   at an 11.7% false-alarm rate

The signal did not carry. Ethereum needed a model fitted on Ethereum's own
labels, which is what it has. Arbitrum would need the same, and has nothing to
fit one on.

## Native ETH: measured, and it does not help

80% of frozen Ethereum addresses also receive native ether, which we never
read. That looked like an obvious gap. It was measured rather than assumed, on
one cohort with one set of probes and two warehouses - stablecoins alone, and
the same addresses read fully:

    warehouse                   ROC-AUC  PR-AUC  TPR@5%FPR
    stablecoins only (ships)     0.9341  0.8106     76.2%
    stablecoins + native ETH     0.9333  0.8187     72.8%
    stablecoins only, >=$1       0.9367  0.8222     77.8%
    stablecoins + ETH,   >=$1    0.9333  0.8256     75.5%

Reading ether moves ROC-AUC -0.0008 and costs 3.3 points of recall at a 5%
false-alarm budget. Removing sub-$1 dust from both assets in both arms widens
the gap rather than closing it, so this is not a spam artifact.

The reason is visible in the counterparties. Among frozen addresses there are
14,173 distinct ether payers against 7,635 stablecoin payers, and they overlap
by **7%**. The ether coming into a collection point is almost entirely a
different population from the victims - gas funding, operational wallets,
dusting - so adding it nearly triples `dest_senders_all` with traffic that is
not people being defrauded. That feature carries more signal than any other,
and the dilution is asymmetric: frozen addresses gain 2.7x more payers, ordinary
ones 1.3x, which compresses exactly the gap the model reads.

"The address receives ether" and "someone paid this address" are different
events. Counting the first as the second makes the model worse.

Worth being clear about what this does NOT mean. Checking an address before
sending it ETH already works, and needs nothing from this: the verdict is about
the destination's behaviour, and a collection point is a collection point
whatever you are about to send it. What was tested here is narrower - whether
ether belongs in the FEATURES - and the answer is no.

The price layer built for this is kept regardless (`veridis.chain.prices`): any
non-dollar asset needs point-in-time USD, and the rule it encodes - value a
transfer at the last candle that had already CLOSED, never the close of the day
it happened in - is the kind of lookahead that makes a backtest unreproducible.

## Bitcoin

No issuer, so nobody can freeze an output. There is no label to learn from and
no way to grade ourselves, which is a different problem from not having got to
it yet.

## What would change this

An Arbitrum-native label source: a freeze list reflecting what an address did
*on Arbitrum*. Until one exists, the honest position is that Arbitrum is
unsupported — not "coming soon".

## The control arm is older than the world the product lives in

Found while adding the Ethereum arm to the nightly ledger, and it bears on every
Ethereum figure in this repo.

A trial pass flagged **19 of 189** live USDT recipients (10%) at the high band.
The held-out evaluation puts the false-positive rate at **2%**. Eleven of the
nineteen had fewer than five payers and under $1,000 received - a 0.1-day-old
address holding $15 is not a collection point.

The first suspicion was that the model is overconfident on thin histories. It is
not. On the held-out set it is at its BEST there:

    payers   frozen  ordinary   HIGH|frozen  HIGH|ordinary  precision
    1-2         465     1,458           62%             2%        89%
    3-4         202       648           43%             0%        99%
    5-19        263     1,124           31%             0%        95%
    20+          26       682           12%             0%       100%

The asymmetry is age, and it comes from how control probes are built. A control
needs `mark - 180d > first_seen`, so every ordinary address in the evaluation
predates its pseudo-freeze by six months. Frozen addresses carry no such
requirement, because collectors are short-lived and get frozen soon after they
appear. The result:

    address age at scoring time    frozen     ordinary
      p25                           34 d        370 d
      median                       138 d        999 d
      p75                          497 d      1,455 d
      under 30 days                  23%           5%

The median ordinary address in the evaluation is **two and three quarter years
old**. The median live USDT recipient is nothing like that. So the model had
"old means ordinary" available as a shortcut - true in the evaluation, false in
production, where a young address is usually a new wallet or a fresh deposit
address.

This does not make the held-out ROC-AUC wrong. It makes it an answer to a
narrower question than the product asks: *can you separate a frozen address from
a wallet that has existed for three years?* rather than *can you separate it from
the addresses people actually pay?*

What it costs today: the live site returns "High risk destination" for an
address with one payer, $15 received and two hours of history. Verified on the
deployed site, not locally.

The fix is in the control arm, not the model or the thresholds. Controls should
be sampled the way the product encounters addresses - from the recent transfer
stream, with the age distribution that implies - rather than from historical
windows filtered to 180-plus days. Until then the Ethereum ledger arm is built
but not scheduled: a ledger entry is a public claim, and these would be claims
we have no reason to believe.

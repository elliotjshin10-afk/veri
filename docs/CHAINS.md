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

## Bitcoin

No issuer, so nobody can freeze an output. There is no label to learn from and
no way to grade ourselves, which is a different problem from not having got to
it yet.

## What would change this

An Arbitrum-native label source: a freeze list reflecting what an address did
*on Arbitrum*. Until one exists, the honest position is that Arbitrum is
unsupported — not "coming soon".

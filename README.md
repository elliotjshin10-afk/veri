# Veridis — pre-send scam detection for irreversible payments

A risk API a wallet or exchange calls *before* a stablecoin transfer executes.
It returns a calibrated probability, a band, and reasons the user can verify
themselves on a block explorer.

```
POST /score
{ "chain": "tron", "sender": "T...", "destination": "T...",
  "amount_usd": 3000, "asset": "USDT" }

→ { "score": 0.87, "band": "high_risk",
    "reasons": [
      "Destination address first seen 4 days ago",
      "47 unrelated wallets have sent to it in the past 7 days",
      "You have never sent to this address before",
      "This transfer is 15x larger than your previous one to it"
    ],
    "model_version": "latest", "computed_at": "..." }
```

Stablecoins are becoming normal money for normal people, and they have none of
the fraud protection normal money has. This is the layer that stops the send.

## The thesis

Blocklists fail by construction. A scam destination address is days old and
clean: no sanctions hit, no illicit history, nothing for a screening vendor to
flag. By the time it is listed, the money is gone.

What *is* detectable is the victim-side behavioural signature — a small test
transfer, escalating amounts to a brand-new counterparty, a sharp break from the
sender's own baseline, bursts of transfers under real-time coaching. That
signature exists weeks before the large loss.

The replay demo (`make demo`) is the direct test of this claim: one real
victim's real history, stepped transfer by transfer, blocklist verdict on the
left, our score on the right.

## Quickstart

```bash
make setup          # venv + deps   (macOS also needs: brew install libomp)
make labels         # M0  public scam labels with provenance
make scam-history   # M1a scam-address transfer history
make victims        # M1b victim extraction + victim histories
make controls       # M2  matched control population
make features       # M3  as-of-time feature layer
make train          # M4  temporal+grouped evaluation
make demo           # M6  replay harness -> demo/index.html
make api            # M5  serve on :8000
make test           # leakage gate
```

## Data sources (all public, all free)

| Need | Source | Notes |
|---|---|---|
| Tron scam labels | **Tether TRC-20 `AddedBlackList` events** | 8,552 addresses, 2020-06 → today, each with an on-chain freeze timestamp |
| Sanctions labels | OFAC SDN `sdn.csv` | 114 Tron, 91 Ethereum |
| Community labels | CryptoScamDB, ScamSniffer, MEW darklist | Ethereum-heavy; 20 valid Tron |
| Tron history | TronGrid free tier | TRC-20 transfers, no key needed |
| Ethereum history | Etherscan (needs key) | see *Cross-chain* below |

**Why Tether freezes are the label backbone.** Public Tron scam lists are thin:
CryptoScamDB yields 20 valid Tron addresses and ScamSniffer yields zero. The
Tether freeze list is on-chain, timestamped, law-enforcement-driven, and two
orders of magnitude larger. The freeze timestamp doubles as `first_reported_at`,
which is exactly what the temporal split needs.

**Label caveat, tracked in the data.** A Tether freeze is not a
pig-butchering conviction. The list mixes retail scam collection addresses with
laundering infrastructure, hacks and sanctions cases. Two guards:

1. **Provenance on every row** — `sources`, `source_count`, `label_basis`, so
   sensitivity to label quality is measurable rather than assumed.
2. **A retail-collection filter** (`dataset/events.py`) — an address is only
   used to source victims if it looks like a retail collection point: ≥5
   distinct inbound senders, ≤$250k average per sender, inbound concentrated
   and forwarded on. Without it, the "victims" of a $99M address are other
   criminal wallets, which would corrupt the behavioural signature.

## How this differs from recipient-side fraud prevention

The established lane is **recipient-side**: identify the scammer's
infrastructure, then block payments into it. Alterya is the reference point —
YC and Battery backed, **acquired by Chainalysis in January 2025 for $150M**,
serving Binance, Coinbase and Block. Their method is AI monitoring of *"web,
social, and chat channels"* plus KYC signals, and they position explicitly on
*"recipient-side risk ... rather than just sender behaviour."*

We ask the other question — *is this person being scammed?* — and it is
answerable with **no information about the recipient at all**.

That is a testable claim, not a slogan, so `make train` tests it. The same
model is retrained four times, each arm allowed a different view of the
transfer (`src/veridis/model/ablation.py`):

| arm | what it can see |
|---|---|
| `destination_only` | the ceiling for a blocklist or recipient-side intelligence |
| `sender_behaviour_only` | the victim's own pattern, nothing about the counterparty |
| `sender_side_only` | sender behaviour + the relationship, **zero destination features** |
| `full` | both halves |

Plus an **unseen-destination test**: scoring restricted to transfers whose
destination had no prior inbound activity at all — an address effectively
invisible to anything that works by knowing something about the recipient.

Tests assert the sender-side arms genuinely cannot see destination features
(`tests/test_fraud_features.py`), so the claim cannot quietly rot as features
are added. Full write-up in [`docs/positioning.md`](docs/positioning.md).

## Researching any wallet

`/score` answers *should this person send?*. `/research` answers *what is this
address?* — with no sender and no transfer in mind, for any Tron wallet,
whether or not we have seen it before.

```bash
make research ADDR=TQyzUvjrrENU6poCGWoxaTnxXBZtDE5hQM
```

```
fetched from chain  (2.8s, 780,711 transfers in warehouse)

  ON A PUBLIC LIST  source=tether_freeze  first reported 2026-03-25
  verdict: some collection-like behaviour
    - 97 unrelated wallets have paid it
    - funds leave within about 56 minutes of arriving
```

Or over HTTP: `GET /research?address=T...`

**Warm and cold are reported separately.** An address already indexed is pure
compute (single-digit milliseconds). An address we have never seen needs its
TRC-20 history pulled from TronGrid first — seconds, because the free tier is
rate-limited. The response says which path it took (`"source": "indexed"` vs
`"cold_fetch"`) so the two are never confused. In production the warehouse is
an indexer and the cold path disappears.

**The verdict is rule-based, not the model.** Every observation cites a number
the caller can check on a block explorer. Describing an address should not
depend on whose transfer prompted the question, and a model score is not a
description.

**Scale is checked before shape.** An exchange hot wallet has the *same shape*
as a scam collector — many payers in, few payees out, everything forwarded
immediately. A shape-only rule set called a real Binance hot wallet a
collection address (it tripped four of six rules). Retail collection points
move thousands per victim; that wallet moved **$20.4bn** at $617M per paying
wallet. Scale now gates shape, and a years-old address needs more than shape
to be called a collector.

**Indexed is not the same as present.** Most addresses in the warehouse are
counterparties seen through someone else's history, so we hold one or two of
their transfers. Profiling those as complete produces confident nonsense — one
address read as "100% of outflow to a single address" from a single observed
transfer, and turned out on a real fetch to be $30bn of exchange
infrastructure. Only addresses whose own history was fetched count as indexed;
everything else is fetched before it is described.

**The published demo cannot do this.** An Artifact page is sandboxed and cannot
reach TronGrid, so the shared demo scores a fixed set of real held-out
transfers and says plainly that it will not invent a result for an address it
has no history for. Live lookup is the API and the CLI.

## Behavioural fraud-modelling features

The sender half is built from techniques that card, ACH and APP-fraud teams
already rely on, translated to on-chain data:

| technique | feature | why it matters here |
|---|---|---|
| Deviation from own baseline | `amount_zscore_own` | $5,000 is unremarkable for a $5k/week sender and alarming for a $50/week one; a population threshold cannot tell them apart |
| Dormancy then activity | `sender_dormancy_days` | a quiet wallet that suddenly moves money is the classic mule / takeover shape |
| Counterparty concentration | `pair_share_of_outflow_7d` | "all recent funds to one new payee" is the APP-fraud signature |
| Balance drawdown | `sender_drawdown_7d` | the terminal sweep |
| Pass-through | `sender_passthrough_7d` | money arriving and leaving rather than being held |
| Counterparty churn | `sender_dest_churn_7d` | a behavioural break from the usual set of payees |
| Round-number amounts | `amount_round_1k` | coached transfers are dictated in round figures; organic commerce is not |
| Velocity | `sender_burst_1h/24h/7d` | real-time coaching produces bursts |

69 features total: 31 sender, 18 destination, 15 relationship, 5 context.

## Methodology

This is the part that decides whether the model is real. The easiest way to
fail here is to build something that looks excellent and is quietly cheating.

### Point-in-time correctness

Every feature for an event at time `t` reads only transfers with
`block_time < t`. The classic failure is counting *all* wallets that ever sent
to a destination — which counts the later victims and leaks the label.

`features/asof.py` has no whole-history aggregate anywhere in it. Expensive
quantities that would otherwise need a per-event correlated subquery (sweep
gap, running balance) are precomputed **causally** — each row looks only at
rows strictly before it — so they can be aggregated under the same strict time
filter.

**The gate is behavioural, not a code review** (`tests/test_leakage.py`):
compute the feature matrix, then append transfers dated at or after each event
— 40 new senders, a $99,999 inbound, a same-millisecond tie — recompute, and
require the matrix to be bit-identical. If any feature peeks, it moves.

```
make test
```

### Temporal + grouped splitting

Split on `first_reported_at`, never randomly: a random split scatters one scam
ring's addresses across both sides. Because every event for a given scam
address shares that address's freeze timestamp, splitting on it *also* enforces
group integrity — a scam operation lands entirely on one side. This is asserted
at train time, not assumed (`assert_group_integrity`).

### Matched controls

Negatives must resemble positives on everything except the behaviour we want
the model to find. Without that, the model learns "large transfer = scam" and
fires on every legitimate big payment.

Controls are drawn as a **connected two-hop subgraph** rather than isolated
addresses, so a control event's sender *and* destination both have fetched
history — matching the positives, where victim and scam address are both fully
fetched. If control destinations had thinner history than scam destinations,
the model could separate the classes on data completeness alone. That would
look like signal and generalise to nothing.

**Coarse bins were not enough.** Sampling the transfer firehose samples
*transfers*, which is size-biased: a random transfer is far more likely to come
from a high-volume sender, while victims are ordinary retail wallets. Bin
matching left 53% of controls matched only on chain — i.e. not matched at all —
and the classes still differed sharply on sender activity. So matching is
**nearest-neighbour on the standardised, log-scaled covariates** (account age,
outbound count, typical transfer size), without replacement, which optimises
balance directly instead of hoping the bins line up.

Balance is then reported as a standardised mean difference, not asserted:

| covariate | SMD after NN matching |
|---|---|
| `sender_median_out_usd` | 0.047 (balanced) |
| `sender_outbound_count` | 0.158 |
| `sender_age_days` | 0.293 (**still imbalanced**) |

Victims are older accounts than firehose-sampled controls, and that gap
survives matching. It is a real limitation of this control pool and is stated
rather than buried — a better pool would be sampled uniformly over *addresses*
rather than over transfers.

Metrics are reported at the achieved ratio (5.8:1), never rebalanced.

### Metrics

Primary is **early-detection rate**: of victim relationships, what share are
flagged at or before their k-th transfer, at FPR ≤ 1% on matched controls
(k = 1, 2, 3, 5). Catching transfer 7 is worthless — the money is gone.

Secondary is **dollars protected**: share of victim losses that moved at or
after our first flag. PR-AUC is reported rather than ROC-AUC (severe class
imbalance makes ROC-AUC flattering), and FPR is always stated at the operating
threshold.

## Engineering notes

**TronGrid throughput is the binding constraint.** Measured, not guessed:

| config | throughput | retries |
|---|---|---|
| rate 3.0/s, concurrency 3 | 0.23 addr/s | 64 retries, 3 hard failures |
| rate 1.6/s, concurrency 1 | 0.22 addr/s | 27 retries |
| **rate 0.85/s, concurrency 1** | **0.63 addr/s** | **0 retries, 0 errors** |

The limiter is burst-sensitive, so concurrency is actively counterproductive —
sequential at ~0.85 req/s is roughly 3x faster end-to-end than three parallel
workers. The 496-address scam-history pass ran at that setting with 9 retries
and zero errors.

**There is also a depleting quota, not just a rate limit.** After roughly
1,500 requests the same endpoint began returning 429 to *every* call, and
backing off to 0.5 req/s still only got 4/12 through. This is why the ingest
is staged and resumable, why every response is cached to disk keyed on the
full request, and why `TRON_RATE` is an environment variable: a run that hits
the quota wall can be resumed later at a lower rate without refetching
anything it already has. Retried 429s consume quota too, so backoff starts at
8s rather than 1s.

A TronGrid API key raises these limits substantially and is the first thing to
add for a production-scale dataset; none was used here.

## Results

Trained on real Tron data, temporal + grouped split, control pool matched
5.7:1. Full report: `reports/evaluation.md`.

| metric | value |
|---|---|
| PR-AUC | **0.780** vs 0.149 base rate (**5.2x lift**) |
| ROC-AUC | 0.942 |
| FPR at operating point | **0.98%** (budget 1%) |
| Precision / recall | 0.886 / 0.437 |
| Early detection, k=1 | **35.3%** [28.4%, 41.7%] |
| Early detection, k=3 | 53.7% [46.8%, 60.1%] |
| Dollars after first flag | **59.5%** of $9,658,364 |
| Calibration (ECE) | 0.0279 |
| API p95 latency | **34.1 ms** over 1,280,979 transfers |

Test set: 218 held-out victim relationships across
52 scam addresses, 5,727 events.

### These numbers went *down*, on purpose

An earlier run reported k=1 detection of 47.7% and PR-AUC 0.898. Those were
inflated by a control pool too thin to hold a 1% false-positive budget
honestly - the pipeline's own base-rate guard said so at the time. Doubling the
control population to 957k transfers moved the base rate from 21.5% to
14.9% and the numbers fell:

| | thin controls (3.6:1) | realistic controls (5.7:1) |
|---|---|---|
| Early detection k=1 | 47.7% | **35.3%** |
| Early detection k=3 | 69.3% | **53.7%** |
| Dollars after first flag | 72.5% | **59.5%** |
| PR-AUC | 0.898 | 0.780 |
| PR-AUC lift over base | 4.2x | **5.2x** |

Lift went *up* while the headline rates went down, which is the tell: the model
did not get worse, the yardstick got honest. Real-world prevalence is lower
still, so treat even these as optimistic.

### The differentiation result

Same model, retrained with different views of the transfer:

| arm | features | PR-AUC lift | caught at k=1 | caught by k=3 |
|---|---|---|---|---|
| Everything | 75 | **5.72x** | **41.7%** | **59.6%** |
| Sender side only (no destination features) | 57 | 4.87x | 24.3% | 36.7% |
| Destination only (the recipient-side ceiling) | 23 | 4.75x | 27.1% | 40.8% |
| Sender behaviour alone (no relationship) | 42 | 3.86x | 16.5% | 25.2% |

**The claim is independence, not superiority.** The two halves are close in
strength (4.87x against 4.75x). What matters is
that they are largely independent: together they catch 42% on the first
transfer against 27% and 24% alone - roughly
15% points more than the better half.
A recipient-side system is leaving that on the table and cannot recover it by
getting better at recipients.

**Where destination data does not exist** - transfers whose destination had no
prior inbound activity - the sender-side model still reaches PR-AUC
**0.645** against 0.127 base (5.1x, n=236).

Signal by family: relationship 34.7%, sender 31.2%,
destination 26.1% - **66% sender-side**.

### Risk analytics

Four things a risk function expects that a bare classifier does not provide.

**Cost, not accuracy.** Nobody buys recall; a wallet trades money kept against
customers interrupted. No cost per interruption is assumed &mdash; that price is
the customer's &mdash; so the curve reports dollars protected *per interruption*:

| budget | interruptions | protected | per interruption |
|---|---|---|---|
| 0.10% | 3 | $2,701,365 | $900,455 |
| 0.50% | 15 | $6,257,587 | $417,172 |
| 1.00% | 31 | $7,004,399 | $225,948 |
| 2.00% | 62 | $7,593,619 | $122,478 |
| 5.00% | 155 | $8,511,668 | $54,914 |
| 10.00% | 310 | $9,192,916 | $29,655 |

**Uncertainty.** Bootstrapped over victim *relationships*, not events &mdash;
events inside one relationship are not independent, and resampling them would
give intervals far too tight.

| k | detection | 95% CI |
|---|---|---|
| 1 | 47.7% | 41.3% &ndash; 54.1% |
| 2 | 65.1% | 58.7% &ndash; 72.0% |
| 3 | 69.3% | 63.3% &ndash; 75.2% |
| 5 | 72.0% | 66.1% &ndash; 78.4% |

ROC-AUC 0.965 [0.958, 0.970] over 218 relationships.

**Calibration.** ECE **0.0154**, worst bin gap 0.076.
This was 0.098 until a real bug surfaced: Platt scaling was being fitted on
LightGBM's output, which is already a probability, so a sigmoid was being asked
to correct a sigmoid. Fitting on the score's *logit* &mdash; which is what Platt
scaling is defined on &mdash; cut the error more than sixfold.

**Stability.** Population stability index between the train and test periods:

| family | median PSI | material shifts |
|---|---|---|
| destination | 0.131 | 6/18 |
| sender | 0.083 | 2/37 |
| relationship | 0.063 | 0/12 |
| context | 0.054 | 0/3 |

**Destination features drift most** (0.131 against
0.083 for sender features). Scam infrastructure is rebuilt
constantly; how a person behaves when they are being talked into a transfer is
not. That is an argument for the sender side being the more durable half,
separate from how well either scores today.

### A measured negative: sequential evidence does not help

A victim relationship is a sequence, so accumulating log-likelihood ratios along
it (Wald's sequential test) should in principle beat scoring each transfer
alone. It does not:

| decay | k=1 | k=2 | k=3 | k=5 |
|---|---|---|---|---|
| 1.0 | 0.0% | 7.8% | 17.4% | 24.3% |
| 0.9 | 0.0% | 19.7% | 30.7% | 38.5% |
| 0.7 | 14.2% | 39.9% | 47.7% | 51.8% |
| 0.5 | 27.1% | 51.4% | 56.4% | 60.6% |
| 0.3 | 37.6% | 58.7% | 63.3% | 66.5% |
| 0.15 | 42.7% | 64.2% | 68.8% | 71.6% |
| **point score** | **47.7%** | **65.1%** | **69.3%** | **72.0%** |

The statistic converges to the point score as decay approaches 0 and never
exceeds it. That it degenerates to the right limit is evidence the
implementation is correct; the reason it adds nothing is that the as-of-time
features (`sender_prior_sends_to_dest`, `pair_share_of_outflow_7d`,
`escalation_ratio`) already encode the relationship history, so accumulating the
*output* on top double-counts it and imports noise from long control
relationships.

Getting there took two wrong turns worth recording. `logit(score) - logit(prior)`
is not a likelihood ratio: on a miscalibrated score an ordinary transfer
contributes weakly *positive* evidence, so long legitimate relationships drift
toward the alarm and detection at k=1 collapsed to zero. Binning the ratio
empirically fixed the sign, but 20 bins is far too coarse for a 1% threshold
&mdash; the same discretisation trap that made isotonic unusable for
thresholding. Taking the LLR analytically from the Platt coefficients
(`a*z + b - logit(prior)`) is exact, strictly monotone, and makes k=1 rank
identically to the point score, as it must.

### Address-level risk

`/research` originally scored addresses with hand-written rules &mdash; many
payers, few payees, funds forwarded fast. Measured against our own labels the
rule set was **anti-correlated**: it called 32.2% of ordinary control wallets
"behaves like a collection address" against 23.5% of addresses actually on a
public scam list. The reason is structural. An exchange deposit wallet, a
consolidation address and a scam collector have the same *shape*; shape alone
cannot separate them.

It is now the destination-only arm of the model, evaluated at address level:
**ROC-AUC 0.908** over 493 listed and 1,198
ordinary addresses, median score 0.66 against
0.0015. The high band fires on 54.2% of listed
addresses and 5.0% of controls. The checkable facts remain &mdash; they are what
a user verifies &mdash; but as observations, not a verdict.

## Robustness slices

| slice | result |
|---|---|
| Hard negatives (destination already had ≥5 senders, n=2,747) | PR-AUC 0.902 vs 0.237 base; k=1 55.0% |
| First-send only (218 victims vs 226 controls) | ROC-AUC **0.886**, FPR 0.88% |

The first-send slice is the case a blocking interstitial actually fires on, and
`is_first_send_to_dest` is constant there, so the model cannot lean on "have
you paid this address before".

## Scale achieved, and what limits it

The pipeline is built to the brief's targets and parameterised by scale. What
actually ran is bounded by the free-tier quota described above, not by the
code:

| milestone | brief target | achieved |
|---|---|---|
| M0 corroborated scam labels | ≥ 500 | **8,552** Tron (+472 Ethereum) |
| M1 victims identified | ≥ 5,000 | **9,207** distinct victims, 10,233 relationships |
| M1 victim histories fetched | — | 54 victims, 22,953 transfers (quota-bound) |
| M2 matched controls | 10:1 | 5.8:1 achieved (nearest-neighbour matched) |
| M3 leakage gate | green | **green** |
| M4 early detection at k=1 | ≤ 1% FPR | 28.6% at 0.97% FPR |
| M5 API p95 | < 300 ms | **17.5 ms** |
| M6 replay demo | real data | real held-out victim |
| M7 cross-chain | train Tron → test ETH | blocked on credentials |

Label collection and victim *identification* hit the brief's targets comfortably,
because they are cheap: one pass over the freeze list, then one history fetch
per scam address. The expensive step is fetching a full transaction history for
each victim and each control address — one address at a time, against a quota
that depletes. That, not the modelling, is what caps dataset size here.

Everything is cached and resumable, so raising the scale is a matter of running
the same commands for longer with an API key:

```bash
TRON_RATE=3 make victims N=5000 PAGES=6
TRON_RATE=3 make controls
```

`scripts/harvest_cache.py` rebuilds the dataset from whatever has already been
cached, without making any requests, so a partial ingest is always usable.

## Cross-chain (M7) — blocked on credentials

Training on Tron and testing on Ethereum is implemented
(`scripts/m7_crosschain.py`) but **has not been run**. It needs full pre-loss
history for labelled Ethereum addresses, which is archive data. Free public RPC
endpoints refuse archive-range `eth_getLogs` (`"Archive requests require a
personal token"`), and Etherscan, BigQuery and Dune all require an account.
Set `ETHERSCAN_API_KEY` and `make crosschain` to run it. No weaker experiment
is substituted and labelled cross-chain.

## Non-goals

No wallet integration, no browser extension, no on-chain writes, no custody —
we return a number and never touch funds, so no money transmitter license. No
licensed blocklist data: the comparison baseline is a public list, and the
point is that it fails. No PII, no KYC, no off-chain identity. We score
transactions, not people, and make no attempt to name operators.

## Layout

```
src/veridis/
  chain/      address encoding, cached HTTP, TronGrid + EVM clients
  labels/     M0  public sources, corroboration, provenance
  dataset/    M1/M2  victims, events, control matching
  features/   M3  the as-of-time feature layer  <- the important file
  model/      M4/M5  training, metrics, SHAP reason codes
  api/        M5  FastAPI /score
tests/        the leakage gate
scripts/      one per milestone, all resumable
```

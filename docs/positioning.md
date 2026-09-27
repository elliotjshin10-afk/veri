# Veridis — positioning

## One line

Stablecoins are becoming normal money for normal people, and they have none of
the fraud protection normal money has. We're the layer that stops the send.

## The problem

Irreversible push payments have no chargeback, no Regulation E, and no
incumbent fraud layer. When a victim is coached into sending USDT to a scammer,
there is no recall, no dispute, no issuer to eat the loss. The send *is* the
loss event, so the only place to intervene is before it executes.

## Market context

| | |
|---|---|
| Stablecoin real-economy payment volume, 2025 | ≈ **$400B**, ~60% B2B → ~**$160B** consumer/P2P |
| Total stablecoin supply | ~**$316B**, growing ~50% YoY |
| Active stablecoin addresses | 19.6M → 30M (Feb 2024 → Feb 2025), **+53%** |
| Illicit addresses received, 2025 | ≥**$154B**, +162%, **84% in stablecoins** |
| Scams and fraud specifically | ≈ **$17B**, ~$14B visible on-chain |
| FBI IC3 crypto complaints, 2025 | **181,565**, >**$11.36B** reported losses |
| GENIUS Act full regime effective | **January 18, 2027** |

A note on the denominator: we deliberately do **not** cite the ~$7.2T monthly
raw on-chain transfer figure. That number is bots, MEV and exchange churn.
Real-economy payment volume is roughly $400B a year, and the consumer slice of
that — where scam losses actually land — is around $160B. Using the gross
number would inflate the TAM by four orders of magnitude and tell a fintech
investor we do not understand our own market.

## Why now

The GENIUS Act regime takes full effect **January 18, 2027**. Regulated
stablecoin issuers and the wallets distributing them will be held to consumer
protection expectations that the rails do not currently support. Today there is
no pre-send fraud layer to buy. That is the window.

## The competitive map

The crypto-fraud lane most people mean is **recipient-side**: identify the
scammer's infrastructure, then stop payments into it.

**Alterya** is the reference point. Founded 2022, Y Combinator and Battery
backed, ~$9.8M seed, **acquired by Chainalysis in January 2025 for $150M**,
serving Binance, Coinbase and Block. Their method, in their own framing, is
AI-driven monitoring of *"web, social, and chat channels"* to *"detect scams at
inception"*, combined with KYC signals, and they position explicitly on
*"recipient-side risk ... rather than just sender behaviour."*

That is a good business and it is now consolidated inside the incumbent. It is
also a different question from ours.

| | Recipient-side (Alterya / Chainalysis, blocklists) | Sender-side (Veridis) |
|---|---|---|
| Question | Is this address known-bad? | Is this person being scammed? |
| Evidence | Off-chain web/social/chat monitoring, scam reports, KYC | The sender's own on-chain behaviour |
| Requires the scam to have been observed somewhere | Yes | **No** |
| Requires data partnerships / KYC access | Yes | **No** |
| Touches PII | Yes | **None — addresses only** |
| Works on an address with zero footprint | No | **Yes** |

**Why the distinction is not rhetorical.** In our replay, a real victim sent
five transfers to a destination that no public source listed until **nearly
three years later**. Every recipient-side system, by construction, had nothing
to say on the days the money actually moved. The sender's behaviour, meanwhile,
was already anomalous at transfer one.

**We are a complement, not a competitor.** A wallet wants both checks: the
recipient check catches known infrastructure cheaply, and the sender check
covers the window before anything is known — which is the window in which
retail victims are actually robbed. That also makes Chainalysis a plausible
distribution partner rather than only a rival, and it means we do not need to
win a data-coverage race we would lose.

**The measurable version of the claim.** We retrain the same model with the
destination features removed entirely and report it (`make train` prints the
ablation). A sender-only model that still separates victims from matched
controls is detecting the victim, not the recipient — and that signal is
structurally unavailable to everyone in the recipient-side lane.

## Why this is defensible

Blocklists fail by construction, and that is not a criticism of the vendors —
it is arithmetic. A scam destination address is days old and clean. Nothing has
been reported against it, because the victims have not realised yet. By the
time it is listed, the money has been swept.

Our signal is on the **victim's** side of the transaction, which is visible
weeks earlier: a small test transfer that succeeds, escalating amounts to a
counterparty with no prior relationship, a sharp break from the sender's own
baseline, bursts of transfers under live coaching, ending in a balance sweep.
That signature does not depend on anyone having reported the scam yet.

Data advantage compounds: every scam address that is eventually frozen or
reported labels the *entire* victim cohort that preceded it, retroactively,
with full transaction history attached and no customer relationship required.

## Why the wedge generalises

This is pre-send fraud detection for irreversible push payments. Stablecoins
are the wedge because there is no chargeback, no Regulation E and no incumbent.
The identical behavioural signal applies to FedNow, RTP, Pix and UPI, which
move orders of magnitude more money and are seeing the same authorised-push-
payment fraud. The model is behavioural, not chain-specific — which is exactly
what the cross-chain test (M7) is designed to demonstrate.

## What we sell

A `/score` call and a reason string. The consuming wallet decides whether to
warn, add friction, or route to review. We never touch funds, hold no custody,
write nothing on-chain, and need no money transmitter license.

The threshold is a customer-tunable FPR budget, not a fixed score: a wallet
will not ship a model that interrupts 5% of legitimate sends, so the operating
point is chosen on the false-positive rate the customer will tolerate and the
recall follows from it.

# The prediction ledger

Every performance number in this repo is retrospective. A backtest asks the
reader to trust that the protocol was fixed before the result was seen, and
that trust is exactly what a sceptical reader will not extend — correctly, since
this project has had to walk back two headline numbers for that reason:

- `53% recall @ 5% FPR` became `22% @ 1% FPR` once recall and the false-alarm
  rate were measured in the same experiment instead of two different ones.
- A retrained model's `63% TPR` became `22.7%` once the control population it
  was tested against was one it had never trained on.

Both corrections were found by testing harder, not by the numbers being
dishonest. But a reader cannot see that from outside, and shouldn't have to.

The ledger removes the need to trust. Each night we score addresses that are on
no list anywhere and write down the ones that look like collection points. When
Tether later freezes one, the gap between our entry and their freeze is a lead
time with both ends public and on-chain.

## Running it

```
make ledger            # predict, resolve, report
make ledger N=800      # more candidates in the prediction pass
make ledger-verify     # recompute both hash chains
```

Nightly it runs in CI, not on a laptop — see the timestamping section below for
why that matters. To run it on a machine instead:

```
0 3 * * *  cd /path/to/veridis && make ledger >> /var/log/veridis-ledger.log 2>&1
```

Order matters. `predict` runs before `resolve` so the newest entries are checked
on the same run rather than sitting unjudged for a day.

## What the files are

| file | what it holds |
|---|---|
| `data/ledger/predictions.jsonl` | one line per flagged address: score, threshold, the features behind it, and the hash of the model that produced it |
| `data/ledger/resolutions.jsonl` | one line per prediction that later got frozen, with the lead time |
| `site/ledger.json` | the summary the page renders |

Resolutions live in their own file rather than mutating predictions. An
append-only log that gets edited in place is not append-only.

## What the hash chain proves, and what it does not

Each entry stores the SHA-256 of `previous_hash || canonical(payload)`. Editing,
reordering or deleting any entry changes every hash after it, and
`make ledger-verify` recomputes the whole chain. `tests/test_ledger.py` asserts
each of those four failure modes.

**A chain proves order and integrity, not time.** On its own it says nothing
about *when* an entry was written — whoever holds the file could regenerate the
whole thing, chain and all.

The timestamp comes from somewhere we do not control. The nightly job runs as a
GitHub Action (`.github/workflows/ledger.yml`) and commits that night's entries,
so each batch carries a commit timestamp set by GitHub rather than by us. A
reader checking whether we really called an address before Tether froze it can
compare two dates neither of us issued: the commit, and the freeze transaction.

That is the whole reason the ledger is tracked in git while the multi-gigabyte
warehouse is not.

Stronger still would be anchoring each head hash on a public chain, where not
even the host is trusted. Worth doing eventually; the current setup is already
enough that the claim does not rest on our word.

## Why the report is split into cohorts

A prediction made last week has not failed — it has not been given time to come
true. Counting it as a miss understates the model; quietly dropping it
overstates it. Both errors come from ignoring that the data is right-censored.

So outcomes are reported by cohort age, and only cohorts older than the
backtest's median lead time (90 days) carry a resolution rate. Younger entries
are shown with their count and marked "too early to judge", so nothing is hidden
and nothing is counted before its time.

## Honest limits

- **The freeze list is not ground truth about fraud.** It is the best public,
  timestamped signal available, but Tether freezes for reasons beyond scams, and
  many scam addresses are never frozen at all. An unresolved prediction is not
  proof we were wrong.
- **Candidates are sampled, not exhaustive.** The pool is addresses receiving
  USDT in the sampled windows, so the ledger measures the model, not coverage of
  Tron.
- **Predictions are made at a 1% false-alarm budget.** At a looser threshold the
  ledger would fill with wallets that were never going to be frozen, and the
  resolution rate would say more about the threshold than the model.

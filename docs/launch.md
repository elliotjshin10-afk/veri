# Launch copy

Written to be posted, not read internally. The angle is the measurement
honesty, because the tool itself is a card with three fields and that is not
what earns attention from a technical audience. Everything below is a figure
the site already shows and `make` can reproduce.

---

## Show HN

**Title**

    Show HN: Veridis – check a stablecoin address before you send

**Body**

Stablecoin transfers cannot be reversed. If someone talks you into sending
USDT to a scam address, there is no chargeback, no dispute, and no issuer to
eat the loss. The send is the loss event.

Most crypto risk tools answer "is this address on a list". That works when the
address is known, and the whole problem is that it isn't: the address
collecting a pig-butchering scam is typically weeks old and has never been
reported, because the victims haven't realised yet. By the time it is listed,
it has been emptied. In our held-out cases the typical gap between the payment
and the freeze was 195 days.

So this does the other thing. It reads the destination's entire transfer
history off a public block explorer, plus the history of the wallet that
address forwards money to, turns that into 19 behavioural numbers, and runs 400
gradient-boosted trees over them. All of it happens in your browser. The
address you type goes to the explorer, not to us.

Three things that might be interesting:

**The second hop runs backwards from the textbook.** The standard AML intuition
is that scam addresses sweep into shared mule wallets, so a high fan-in at the
payout address is suspicious. We measured it and it is the opposite: scam
destinations forward to wallets with a median of 4 other payers, ordinary ones
to wallets with 71, because an ordinary wallet forwards to an exchange and an
exchange hot wallet is fed by thousands. Our own code asserted the wrong
direction in a comment for a long time. Fixing it was worth +2.5pp of recall at
the operating point.

**We publish the number that makes us look bad.** 96% of transfers into
addresses that were later frozen met a warning. That figure is held out, and
quoted alone it misleads, because recall doesn't depend on how common fraud is
and precision depends on nothing else. We sampled 3,540 addresses receiving
USDT over a week and none were on Tether's freeze list, which puts live
prevalence under 0.085%. At that rate most warnings on a random payment would
be false. The figure that survives any base rate is the likelihood ratio: a
stop multiplies the odds you're paying a collection point by 39. That's what's
on the front page, and the precision table is on the how-it-works page.

**Three of our own published numbers got worse under scrutiny, not better.** An
age confound between the two arms, a graph feature that was measuring which
addresses we'd chosen to fetch rather than the chain, and a lead-time claim
that looked flat because each horizon was a different and older population.
Each is written up with the before and after.

Limits, stated on the site rather than buried: it can't stop or reverse
anything, it doesn't know who owns an address, USDT on Tron and Ethereum only,
and the labels are addresses Tether froze, which selects for large reported
fraud. A scam too small to be reported never enters the training set or the
evaluation.

Live: https://elliotjshin10-afk.github.io/veri/

Try the middle sample address. It's on no public list and the page still calls
it a collection point, which is the entire argument in one click.

---

## X / short form

> Stablecoin transfers can't be reversed, and the address collecting a scam is
> usually weeks old and on no list. By the time it's listed the money is gone.
>
> So we scored the behaviour instead. Runs in your browser, nothing sent to us.
>
> https://elliotjshin10-afk.github.io/veri/

> The part I'd push back on if I read this: we catch 96% of payments into
> addresses later frozen, and on a random payment most of our warnings would
> still be wrong.
>
> Both are true. Live prevalence is under 0.085%, so precision is brutal.
>
> The honest number is the likelihood ratio: a stop makes it 39x more likely
> you're paying a collection point, whoever you are.

> The thing I didn't expect: the standard mule-network signal runs backwards.
>
> High fan-in at the wallet your money forwards to means exchange, not mule.
> Scam destinations: median 4 other payers. Ordinary: 71.
>
> Our own code had the wrong direction written in a comment for a year.

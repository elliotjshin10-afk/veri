# Getting a waitlist

## Why emails beat letters of intent here

An LOI from a wallet takes months and a warm intro. An email takes ten seconds
and no relationship. But the thing worth optimising is not the size of the
list &mdash; it is **what the list is made of and what people did before joining
it**.

For YC specifically, these are not equivalent:

> "500 people joined our waitlist."

> "3,100 addresses checked in three weeks. 412 people left an email. 27 of
> those used a work address at a wallet, exchange or payments company, and
> 9 replied when we wrote back."

The second is traction. The first is a form. Both take the same afternoon to
build; only one of them requires the product to be useful first.

So the funnel is deliberately **tool first, email second**. Somebody who just
watched us explain what an address does has a reason to care what we build
next. Somebody staring at a signup box does not.

## Ship it (about 20 minutes)

1. **Get a form endpoint.** [Formspree](https://formspree.io) or
   [Basin](https://usebasin.com); both free at this volume. Create a form,
   copy the POST URL.
2. **Paste it** into `FORM_ENDPOINT` at the top of `site/index.html`. Left
   blank, the form refuses to submit and says so rather than dropping leads
   silently.
3. **Deploy `site/`.** Any of these, free, no build step:
   - Netlify: drag the folder onto app.netlify.com/drop
   - Vercel: `npx vercel --prod` inside `site/`
   - Cloudflare Pages or GitHub Pages: point at the folder
4. **Buy a domain.** A real one matters more than it should for credibility.
5. **Add analytics** &mdash; Plausible or Cloudflare Web Analytics, both
   cookie-free. You need *checks performed*, not just pageviews. That number
   is the traction metric.

`site/` ships `address_index.json` alongside the page, so the checker works on
2,282 real addresses immediately with no backend.

## What to instrument

| metric | why it matters |
|---|---|
| Addresses checked | the actual usage number; this is what you report |
| Checks per visitor | >1 means it is useful, not a curiosity |
| Email conversion after a check | tells you whether the result lands |
| **Work-email share** | the only number that predicts revenue |
| Replies to your follow-up | separates interest from politeness |

Segment the list by email domain the day it starts filling. A gmail signup is
a vote; `risk@somewallet.com` is a sales lead, and it should get a personal
reply within a day, not a launch announcement in six weeks.

## Where the first thousand come from

Ranked by leverage, not by size.

**1. Show HN.** The technical story is the distribution here, and it is
unusually good: *"I trained a model to spot pig-butchering victims before they
send &mdash; then found my own metrics were inflated three times and fixed
them."* HN rewards exactly that. Lead with the honest negatives (precision is
0.04 at real prevalence; the escalation ratio did not work; the graph features
were reading our own data collection), because on HN they are credibility, not
weakness. Expect wallet and exchange engineers in the comments &mdash; those
are the leads.

**2. The checker is the share mechanism.** "Check this address before you
send" is a thing people send each other, which a signup page is not. Make the
result page shareable: a link that reopens that specific address.

**3. Scam-adjacent communities, carefully.** r/CryptoCurrency, r/Tronix,
r/scams, and the larger recovery-support groups. Go as someone with a free
tool, never as someone selling. These communities are heavily preyed on by
fake "recovery services", so anything that smells like one gets removed
instantly &mdash; and deserves to.

**4. Crypto safety accounts on X.** ZachXBT, ScamSniffer, Chainabuse and the
smaller investigators. Do not pitch. Send a genuinely useful address analysis
on something they are already looking at.

**5. Direct outreach to wallet risk teams.** 20 personal emails beat any
broadcast. Lead with a specific address from their own chain and what it did,
not with a deck.

## The one thing to avoid

Do not buy the list, do not scrape, and do not let the number become the goal.
A waitlist of people who will not open your launch email is worse than no
waitlist, because it teaches you the wrong thing about demand &mdash; and at
this stage, learning the wrong thing is the expensive mistake.

"""Rebuild site/index.html's inlined data, and the artifact preview copy.

Two things the page must never depend on a network round trip for: the three
examples and the two comparison cards. They are a few KB, they are the first
thing anyone sees, and a fetch for them fails exactly where it hurts most - a
sandboxed preview blocks outside requests, so the page renders with an empty
example list and looks broken.

The address index is split. Inlining all of it made lookup work with no network
at all, which is what a sandboxed preview needs - but at 10,785 addresses that
became a 1MB page, and a landing page that costs half a megabyte before it says
anything has traded one problem for a worse one.

So: a working subset ships inside the page, and the full index is a lazy fetch
of the same compact encoding. The deployed site resolves everything; the preview
resolves the subset and says honestly what it cannot reach. The subset is chosen
by what somebody is actually likely to paste - addresses on a public list, worst
first - plus a spread across the other bands so the type-ahead is not all red.

The rows carry the profile only. The human-readable notes are regenerated in
the browser by one function that mirrors collection_verdict, so the wording has
a single implementation rather than one here and one in the live path.
"""
import json, pathlib, re, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SITE = ROOT / "site"
PROCESSED = ROOT / "data" / "processed"


def build_examples() -> dict:
    idx = json.loads((SITE / "address_index.json").read_text())
    entries = idx["entries"]

    def first(fn):
        return next((e for e in entries if fn(e)), None)

    # Labels name what the wallet IS. They used to describe the model's own
    # cleverness - "Reported nowhere, we flag it anyway" - which reads as a
    # boast to anyone who does not already know what a blocklist is.
    picks = [
        (first(lambda e: e["b"] == "high" and e["l"]),
         "A scam collection wallet"),
        (first(lambda e: e["b"] == "high" and not e["l"]),
         "A collection wallet nobody has reported yet"),
        (first(lambda e: e["b"] == "ordinary" and not e["l"]
               and (e["p"].get("senders_all") or 0) > 3),
         "An ordinary personal wallet"),
    ]
    out = []
    for e, label in picks:
        if not e:
            continue
        row = {"label": label}
        row.update({k: e[k] for k in ("a", "s", "b", "l", "p") if k in e})
        if e.get("l"):
            row.update({"ls": e.get("ls"), "lw": e.get("lw")})
        out.append(row)
    # A slightly wider set shown the moment the box is clicked, before anyone
    # has typed. Spans the bands on purpose: a dropdown of eight high-risk
    # addresses would suggest everything is high risk.
    def take(fn, n):
        return [e for e in entries if fn(e)][:n]

    starter = (
        take(lambda e: e["b"] == "high" and e["l"], 2)
        + take(lambda e: e["b"] == "high" and not e["l"], 2)
        + take(lambda e: e["b"] == "elevated", 2)
        + take(lambda e: e["b"] == "ordinary" and (e["p"].get("senders_all") or 0) > 3, 2)
    )
    seen, starter_rows = set(), []
    for e in starter:
        if e["a"] in seen:
            continue
        seen.add(e["a"])
        starter_rows.append({k: e[k] for k in ("a", "s", "b", "l", "p") if k in e})

    return {"examples": out, "starter": starter_rows,
            "model": idx.get("model"), "indexed": idx["count"]}


BANDS = ["ordinary", "elevated", "high"]

# How many rows travel inside the page. ~92 bytes each, so 900 is ~83KB - small
# enough to be invisible next to the fonts, large enough that the demo works
# offline and the type-ahead always has something to show.
INLINE_ROWS = 900

# Column order must match decodeIdx() in site/index.html.
IDX_COLS = ("age_days", "senders_all", "inbound_count", "outbound_count",
            "inbound_usd", "usd_per_sender", "forward_ratio",
            "consolidation_ratio", "median_hold_secs")
# Every field a note threshold reads (0.85 forwarded, 0.6 consolidated, 60d,
# 3600s) is stored exactly. Rounding them was a false economy: at 2dp it shifted
# percentages by a point and pushed a 0.595 consolidation over the 0.6 line,
# inventing a note that was not there, and at full precision these fields cost
# no more bytes because the source values are already short.
# Only the large money figures round, and only to the dollar.
IDX_EXACT = {"age_days", "forward_ratio", "consolidation_ratio"}


def build_index_rows(entries: list) -> list:
    """Array-encode the index: ~90 bytes a row instead of ~520 as objects.

    Notes are dropped - the browser regenerates them - and floats are rounded
    to the precision the page actually displays. Scores keep 3dp, which is far
    inside the gap between band thresholds.
    """
    rows = []
    for e in entries:
        p = e.get("p", {})
        row = [e["a"], round(e["s"], 3), BANDS.index(e["b"]), 1 if e.get("l") else 0]
        for col in IDX_COLS:
            v = p.get(col)
            row.append(v if col in IDX_EXACT else None if v is None else round(v))
        rows.append(row)
    return rows


def leadtime_facts() -> dict | None:
    """Headline figures for the readout.

    The page quotes recall in PAYMENTS, not addresses. Address recall answers
    "what share of scam collectors do we catch", which is not what the product
    does - it sits in a send flow and warns on one payment at a time. The two
    numbers differ in our favour, because the collectors we catch are the busy
    ones: at a 1% false-alarm budget we catch 24% of the addresses but 33% of
    the payments into them.

    Not dollars. The holdings contain a single $212.9bn USDT transfer, larger
    than Tether's whole supply, so any dollar total is set by a few treasury
    movements rather than by victims. Payments are robust to that; the median
    one here is $5,000, which is the scale this is actually about.

    Both halves come from the same two-arm point-in-time run (m9_roc.py) - an
    earlier version paired recall from one protocol with a false-positive rate
    measured a different way, which flattered it.
    """
    roc, base = PROCESSED / "leadtime_roc.json", PROCESSED / "leadtime_report.json"
    pay = PROCESSED / "leadtime_payments.json"
    if not (roc.exists() and base.exists() and pay.exists()):
        return None
    r, b, q = (json.loads(x.read_text()) for x in (roc, base, pay))
    days, key = r["headline_horizon"], "0.01"
    h, op = r["horizons"][str(days)], q["operating_points"][key]
    return {
        "population": b["population"],
        "headline_days": days,
        "payments": q["payments_in_window"],
        "median_payment": q["median_payment_usd"],
        "payment_pct": f"{op['payment_recall']:.0%}",
        "address_pct": f"{op['address_recall']:.0%}",
        "fpr_pct": "1%",
        "n_pos": h["n_pos"], "n_neg": h["n_neg"],
        "roc_auc": round(h["roc_auc"], 3),
        "loose_tpr": f"{h['tpr']:.0%}", "loose_fpr": f"{h['fpr']:.0%}",
        "ops": [{"fpr": f"{float(k):.0%}",
                 "pay": f"{q['operating_points'][k]['payment_recall']:.0%}"}
                for k in ("0.01", "0.02", "0.05", "0.10")],
    }


def ledger_facts() -> dict | None:
    """The published prediction log, trimmed to what the page shows."""
    p = SITE / "ledger.json"
    if not p.exists():
        return None
    r = json.loads(p.read_text())
    return {"predictions": r["predictions"], "head": r["head"][:16],
            "generated_at": r["generated_at"], "cohorts": r["cohorts"],
            "open_recent": [{"a": x["address"], "s": round(x["score"], 3),
                             "d": x["predicted_at"][:10], "age": round(x["age_days"], 1)}
                            for x in r["open_recent"][:6]],
            "confirmed": [{"a": x["address"], "lead": x["lead_days"]}
                          for x in r["confirmed"][:6]]}


def pair_facts() -> dict | None:
    """Headline quality of the two-sided model, beside the one-sided one."""
    pair = SITE / "model_pair.json"
    dest = SITE / "model_dest.json"
    if not (pair.exists() and dest.exists()):
        return None
    pe = json.loads(pair.read_text())["evaluation"]
    de = json.loads(dest.read_text()).get("evaluation", {})
    return {"roc_auc": pe["roc_auc"], "pr_auc": pe["pr_auc"],
            "tpr_at_1pct_fpr": pe["tpr_at_1pct_fpr"],
            "dest_roc_auc": de.get("roc_auc", 0.0)}


def main() -> None:
    ex = build_examples()
    (SITE / "examples.json").write_text(json.dumps(ex, separators=(",", ":")))

    boot = {
        "examples": ex["examples"],
        "starter": ex["starter"],
        "model": ex["model"],
        "indexed": ex["indexed"],
        "leadtime": leadtime_facts(),
        # Measured quality of the two-sided model, quoted where the page invites
        # somebody to supply both sides.
        "pair": pair_facts(),
        # Small enough to inline, and it must be: the ledger is the page's
        # strongest claim and a fetch for it fails in exactly the sandboxed
        # preview where people are shown the page.
        "ledger": ledger_facts(),
        "scenarios": json.loads((SITE / "scenarios.json").read_text()),
        "thresholds": json.loads((SITE / "model_dest.json").read_text())["thresholds"],
    }
    idx = json.loads((SITE / "address_index.json").read_text())
    rows = build_index_rows(idx["entries"])

    # Full set as a separate file, fetched lazily by the page.
    full_blob = json.dumps(rows, separators=(",", ":"))
    (SITE / "index_compact.json").write_text(full_blob)

    # The inlined subset: everything the page names by hand, then listed
    # addresses worst-first, then a spread of the rest so every band is
    # represented in the dropdown.
    pinned = {e["a"] for e in boot["examples"]} | {e["a"] for e in boot["starter"]}
    by_addr = {r[0]: r for r in rows}
    chosen = [by_addr[a] for a in pinned if a in by_addr]
    seen = set(pinned)
    listed = [r for r in rows if r[3] and r[0] not in seen]
    listed.sort(key=lambda r: -r[1])
    for r in listed[:INLINE_ROWS - len(chosen)]:
        chosen.append(r); seen.add(r[0])
    if len(chosen) < INLINE_ROWS:
        rest = [r for r in rows if r[0] not in seen]
        step = max(1, len(rest) // max(1, INLINE_ROWS - len(chosen)))
        for r in rest[::step][:INLINE_ROWS - len(chosen)]:
            chosen.append(r)
    idx_blob = json.dumps(chosen, separators=(",", ":"))

    html = (SITE / "index.html").read_text(encoding="utf-8")
    for name, new in (("BOOT", "var BOOT = " + json.dumps(boot, separators=(",", ":")) + ";"),
                      ("IDX", "var IDX = " + idx_blob + ";")):
        html, n = re.subn(r"var %s = [\[{].*?[\]}];" % name, lambda _: new,
                          html, count=1, flags=re.S)
        if not n:
            sys.exit(f"{name} block not found in site/index.html")
    (SITE / "index.html").write_text(html, encoding="utf-8")

    # Artifact preview copy: the publish wrapper supplies doctype/html/head.
    title = re.search(r"<title>.*?</title>", html, re.S).group(0)
    fonts = "\n".join(re.findall(
        r'<link rel="stylesheet" href="https://fonts\.googleapis[^>]*>', html))
    style = re.search(r"<style>.*?</style>", html, re.S).group(0)
    body = re.search(r"<body>(.*?)</body>", html, re.S).group(1)
    (SITE / "_artifact.html").write_text(
        f"{title}\n{fonts}\n{style}\n{body.strip()}\n", encoding="utf-8")

    print(f"inlined {len(boot['examples'])} examples, {len(boot['scenarios'])} scenarios "
          f"({len(json.dumps(boot)) / 1024:.1f} KB); "
          f"{len(chosen):,} addresses inlined ({len(idx_blob) / 1024:.0f} KB), "
          f"{len(rows):,} in index_compact.json ({len(full_blob) / 1024:.0f} KB, fetched lazily)")
    print("wrote site/examples.json, site/index.html, site/_artifact.html")


if __name__ == "__main__":
    main()

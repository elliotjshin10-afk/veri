"""Assemble the deployable site: only the files the page actually requests.

site/ holds build inputs as well as outputs. address_index.json is 6.6MB and is
the SOURCE build_site.py compresses into index_compact.json - shipping it would
quadruple the deploy for a file no browser ever asks for. This copies exactly
what index.html fetches at runtime, so what goes on the host is what the page
needs and nothing else.
"""
from __future__ import annotations

import pathlib, re, shutil, sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SITE, DIST = ROOT / "site", ROOT / "dist"

# index.html plus every asset it fetches or imports.
RUNTIME = ["index.html", "scorer.js", "index_compact.json",
           "model_dest.json", "model_pair.json",
           # Fetched at run time so a nightly ledger commit updates the live
           # page without rebuilding the site.
           "ledger.json",
           # The Ethereum model, now that it has been judged against ordinary
           # wallets rather than a scam collector's own neighbours.
           "model_eth.json",
           # The Ethereum two-sided model. Optional (see OPTIONAL below).
           "model_pair_eth.json",
           # The replay: one victim, scored at each transfer as it happened.
           # A lookup page cannot show the minutes the product exists for,
           # because there is no "before" on a page you visit afterwards.
           "replay.html", "replay.json",
           # The send sheet: connect, send, warning. The whole product in three
           # beats, run live against the chain rather than against fixtures.
           "send.html",
           # The decided-cases log: held-out transfers, the verdict each got
           # before it settled, and what the chain did afterwards. A score is a
           # claim about the future and this is the only page that shows one
           # beside the future it claimed.
           "history.html", "history.json",
           # Social preview card. A link with no card is a link nobody clicks.
           "og.png"]


# Files the page ASKS for but can run without, because it has a defined answer
# when they are absent. model_pair_eth.json is the two-sided Ethereum model: with
# it a connected wallet gets a relationship verdict, without it the page falls
# back to the destination-only answer, which is weaker but not wrong. Everything
# else in RUNTIME is load-bearing and a missing file stops the build - a page
# deployed without its index or its destination model is broken, not degraded.
OPTIONAL = {"model_pair_eth.json"}


def main() -> None:
    html = (SITE / "index.html").read_text(encoding="utf-8")
    asked = set(re.findall(r'grab\("([^"]+)"\)', html)) \
        | {m.lstrip("./") for m in re.findall(r'import\("([^"]+)"\)', html)}
    missing = asked - set(RUNTIME)
    if missing:
        sys.exit(f"index.html requests files not in RUNTIME: {sorted(missing)}")

    if DIST.exists():
        shutil.rmtree(DIST)
    DIST.mkdir()
    total = 0
    for name in RUNTIME:
        src = SITE / name
        if not src.exists():
            if name in OPTIONAL:
                print(f"  {name:22s} ABSENT - the page degrades without it")
                continue
            sys.exit(f"missing {src} - run `make site` first")
        shutil.copy2(src, DIST / name)
        total += src.stat().st_size
        print(f"  {name:22s} {src.stat().st_size/1024:7.0f} KB")

    # A deploy is a public claim about the ledger's contents, so publish the
    # CHAIN head - the value the page displays and the one a reader can
    # recompute from predictions.jsonl. The first version wrote sha256 of the
    # file instead, which is a real fingerprint of nothing anyone can check
    # against, and which disagreed with the page by construction.
    led = ROOT / "data" / "ledger" / "predictions.jsonl"
    if led.exists():
        lines = [l for l in led.read_text().splitlines() if l.strip()]
        if lines:
            import json
            head = json.loads(lines[-1])["hash"]
            (DIST / "ledger-head.txt").write_text(
                f"{head}\n{len(lines)} entries\n")
            print(f"  ledger-head.txt        {head[:16]} ({len(lines)} entries)")

    print(f"\ndist/ ready - {total/1024/1024:.1f} MB, {len(RUNTIME)} files")
    print("deploy the CONTENTS of dist/ as a static site")


if __name__ == "__main__":
    main()

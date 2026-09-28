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
           # Social preview card. A link with no card is a link nobody clicks.
           "og.png"]


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

"""Verify the site's reasons() reproduces collection_verdict exactly.

The lookup page inlines its address index and regenerates the notes in
the browser, so the wording people read is JavaScript's, not Python's. That is
one implementation rather than two only for as long as they agree - this check
is what keeps them honest.

It extracts the shipped reasons() and decoder straight out of site/lookup.html
rather than a copy of them, runs them over every indexed address, and compares
against collection_verdict applied to the same profile. No tolerance: the two
must produce identical strings, ties included.

The rows come from site/index_compact.json - the full index the deployed page
fetches - not from the subset inlined in the HTML, so the check covers every
address the site can answer for rather than the few hundred it ships with.

(The `n` field stored in address_index.json is NOT the reference. It was
written before the profile was rounded into the index, so on a handful of
addresses it disagrees with its own profile.)
"""
import json, pathlib, subprocess, sys, tempfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from veridis.api.live import collection_verdict  # noqa: E402


def extract(html: str, start: str, end: str) -> str:
    i = html.index(start)
    return html[i:html.index(end, i)]


def main() -> None:
    html = (ROOT / "site" / "lookup.html").read_text(encoding="utf-8")
    index = json.loads((ROOT / "site" / "address_index.json").read_text())

    compact = (ROOT / "site" / "index_compact.json").read_text()
    shipped = (
        "var IDX = " + compact + ";\nvar INDEX = null;\n"
        + extract(html, "  var BANDS = ", "\n  function loadModel") + "\n"
        + extract(html, "  var INSTITUTIONAL_PER_SENDER", "\n  /* Each fetch")
    )
    # The two sides used to disagree on one punctuation mark, the page carrying
    # an HTML entity where Python carried a hyphen, and this driver quietly
    # normalised it away. Both say the same thing now, so the comparison is on
    # the strings as written and a real divergence cannot hide behind a rewrite.
    driver = """
const m = decodeIdx(), out = [];
for (const [k, e] of m) out.push([e.a, reasons(e.p)]);
console.log(JSON.stringify(out));
"""
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as fh:
        fh.write(shipped + driver)
        path = fh.name
    got = dict(json.loads(subprocess.run(
        ["node", path], capture_output=True, text=True, check=True).stdout))

    profiles = {e["a"]: e["p"] for e in index["entries"]}
    if set(got) != set(profiles):
        sys.exit(f"index mismatch: {len(got)} in index_compact.json vs "
                 f"{len(profiles)} in address_index.json")

    bad = []
    for addr, profile in profiles.items():
        _, expected = collection_verdict(profile)
        if got[addr] != expected:
            bad.append((addr, expected, got[addr]))

    print(f"notes parity: {len(profiles) - len(bad)}/{len(profiles)} addresses identical")
    for addr, expected, actual in bad[:5]:
        print(f"\n  {addr}\n    python: {expected}\n    js    : {actual}")
    if bad:
        sys.exit(f"{len(bad)} addresses differ between collection_verdict and reasons()")


if __name__ == "__main__":
    main()

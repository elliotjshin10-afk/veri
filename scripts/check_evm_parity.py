"""Prove the browser normalises Etherscan exactly as Python does.

computeFeatures is shared between chains and parity-checked separately, so the
only surface Ethereum adds is the fetcher: turning an Etherscan `tokentx` row
into {from, to, usd, t}. A disagreement there is invisible - both sides produce
plausible transfers - and it would feed the model something it was never trained
on, which is the failure this project guards against everywhere else.

This used to drive the browser with Blockscout payloads, which stopped being
what ships when the fetcher moved to Etherscan: the harness was proving a path
no longer in the product. It now uses real cached Etherscan payloads, so the
awkward cases are present - odd decimals, zero-value rows, and the impersonation
tokens Ethereum is full of (Cyrillic and mathematical-bold USDT lookalikes, and
tokens whose "symbol" is an advertisement), every one of which must be dropped
by both sides or the model is fed counterfeit stablecoins.
"""
from __future__ import annotations

import gzip, json, pathlib, subprocess, sys, tempfile
sys.path.insert(0, "src")

from veridis.chain.etherscan import normalise
from veridis.config import ROOT

CACHE = ROOT / "data" / "cache" / "etherscan"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200


def payloads(limit: int):
    """Cached Etherscan token-transfer pages, largest first.

    Largest first on purpose: a page with a thousand rows carries far more of
    the odd cases than a page with one, and the point is to exercise them.
    """
    found = []
    for f in CACHE.rglob("*.json.gz"):
        try:
            with gzip.open(f, "rt") as fh:
                d = json.load(fh)
        except Exception:
            continue
        if not isinstance(d, dict):
            continue
        rows = d.get("result")
        if isinstance(rows, list) and rows and isinstance(rows[0], dict) \
           and "tokenSymbol" in rows[0]:
            found.append(rows)
    found.sort(key=len, reverse=True)
    return found[:limit]


def main() -> None:
    batches = payloads(N)
    if not batches:
        sys.exit("no cached Etherscan token-transfer payloads to check against")
    raw = sum(len(b) for b in batches)
    symbols = {str(r.get("tokenSymbol")) for b in batches for r in b}
    print(f"{len(batches)} cached payloads, {raw:,} raw rows, "
          f"{len(symbols):,} distinct token symbols")

    want = []
    for b in batches:
        want.append([{"from": r["from_address"], "to": r["to_address"],
                      "usd": r["amount_usd"], "t": r["block_time"]}
                     for r in normalise(b) if r["amount_usd"] > 0])
    kept = sum(len(w) for w in want)
    print(f"{kept:,} rows survive the stablecoin filter "
          f"({raw - kept:,} dropped as non-stable or zero-value)")

    scorer = (ROOT / "site" / "scorer.js").resolve().as_uri()
    driver = f"""
import {{fetchTransfersEvm}} from '{scorer}';
import fs from 'fs';
const batches = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const out = [];
for (const result of batches) {{
  // One page, served in Etherscan's own envelope, through the shipped fetcher.
  global.fetch = async () => ({{ok: true, status: 200,
    json: async () => ({{status: "1", message: "OK", result}})}});
  const rows = await fetchTransfersEvm('0x0', {{maxPages: 1}});
  out.push(rows.map(r => ({{from: r.from, to: r.to, usd: r.usd, t: r.t}})));
}}
console.log(JSON.stringify(out));
"""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(batches, fh); data = fh.name
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as fh:
        fh.write(driver); drv = fh.name
    res = subprocess.run(["node", drv, data], capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit("node failed:\n" + res.stderr[:2500])
    got = json.loads(res.stdout)

    bad, compared = [], 0
    for i, (w, g) in enumerate(zip(want, got)):
        if len(w) != len(g):
            bad.append((i, f"row count {len(w)} vs {len(g)}"))
            continue
        for a, b in zip(w, g):
            compared += 1
            if a["from"] != b["from"] or a["to"] != b["to"] or a["t"] != b["t"]:
                bad.append((i, f"ids/time differ: {a} vs {b}")); break
            if abs(a["usd"] - b["usd"]) > max(abs(a["usd"]), 1.0) * 1e-12:
                bad.append((i, f"amount {a['usd']!r} vs {b['usd']!r}")); break

    print(f"compared {compared:,} normalised transfers")
    if bad:
        print(f"\nEVM PARITY FAILED - {len(bad)} payloads differ:")
        for i, why in bad[:8]:
            print(f"   payload {i}: {why}")
        sys.exit(1)
    print("\nEVM PARITY OK - the browser normalises Etherscan identically.")


if __name__ == "__main__":
    main()

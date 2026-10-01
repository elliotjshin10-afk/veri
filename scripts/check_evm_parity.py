"""Prove the browser normalises Blockscout exactly as Python does.

computeFeatures is already shared between chains and already parity-checked, so
the only new surface Ethereum adds is the fetcher: turning a Blockscout payload
into {from, to, usd, t}. A disagreement there is invisible - both sides produce
plausible transfers - and it would feed the model something it was never
trained on, which is the failure this project guards against everywhere else.

Real cached payloads are used rather than hand-written ones, so the awkward
cases are present: non-stable tokens to drop, odd decimals, zero-value rows,
and tokens whose decimals live on the token rather than the total.
"""
from __future__ import annotations

import gzip, json, pathlib, subprocess, sys, tempfile
sys.path.insert(0, "src")

from veridis.chain.evm import _normalise
from veridis.config import ROOT

CACHE = ROOT / "data" / "cache" / "blockscout"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 200


def payloads(limit: int):
    out = []
    for f in CACHE.rglob("*"):
        if not f.is_file():
            continue
        try:
            raw = f.read_bytes()
            if raw[:2] == b"\x1f\x8b":
                raw = gzip.decompress(raw)
            d = json.loads(raw)
        except Exception:
            continue
        if isinstance(d, dict) and d.get("items") and isinstance(d["items"][0], dict) \
           and "token" in d["items"][0]:
            out.append(d["items"])
            if len(out) >= limit:
                break
    return out


def main() -> None:
    batches = payloads(N)
    if not batches:
        sys.exit("no cached Blockscout token-transfer payloads to check against")
    rows = sum(len(b) for b in batches)
    print(f"{len(batches)} cached payloads, {rows:,} raw transfer rows")

    # Python side. _normalise takes the address only to shape the row; the
    # fields compared here do not depend on it.
    want = []
    for b in batches:
        want.append([{"from": r["from_address"], "to": r["to_address"],
                      "usd": r["amount_usd"], "t": r["block_time"]}
                     for r in _normalise(b, "0x0") if r["amount_usd"] > 0])

    scorer = (ROOT / "site" / "scorer.js").resolve().as_uri()
    driver = f"""
import {{fetchTransfersEvm}} from '{scorer}';
import fs from 'fs';
const batches = JSON.parse(fs.readFileSync(process.argv[2] === '--' ? process.argv[3] : process.argv[2], 'utf8'));
// Drive the real fetcher by stubbing one page of results through fetch().
const out = [];
for (const items of batches) {{
  global.fetch = async () => ({{ok: true, status: 200, json: async () => ({{items}})}});
  const rows = await fetchTransfersEvm('0x0', {{maxPages: 1}});
  out.push(rows.map(r => ({{from: r.from, to: r.to, usd: r.usd, t: r.t}})));
}}
console.log(JSON.stringify(out));
"""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(batches, fh); data = fh.name
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as fh:
        fh.write(driver); drv = fh.name
    res = subprocess.run(["node", drv, "--", data], capture_output=True, text=True)
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
    print("\nEVM PARITY OK - the browser normalises Blockscout identically.")


if __name__ == "__main__":
    main()

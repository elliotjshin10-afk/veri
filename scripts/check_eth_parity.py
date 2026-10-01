"""Prove the browser scores the Ethereum model exactly as LightGBM does.

model_eth.json is a hand-rolled export of LightGBM's trees, walked by the
browser's own `score()`. The Tron models have been checked against the library
that produced them since the start; the Ethereum model shipped without that
check, which left every live Ethereum verdict resting on an unverified
re-implementation of someone else's tree format.

The sample is the 400 held-out rows closest to a band edge - where a tiny
disagreement is the difference between "ordinary" and "elevated", and so where a
bug would actually change what a person is told.

Run: python scripts/check_eth_parity.py
"""
from __future__ import annotations

import json, pathlib, subprocess, sys, tempfile
sys.path.insert(0, "src")

from veridis.config import PROCESSED, ROOT, SITE

SAMPLE = PROCESSED / "eth_parity_sample.json"
MODEL = SITE / "model_eth.json"


def main() -> None:
    if not SAMPLE.exists():
        sys.exit(f"no {SAMPLE.name} - run scripts/m7_pit_model.py first")
    if not MODEL.exists():
        sys.exit(f"no {MODEL.name} - the Ethereum model has not been shipped")
    sample = json.loads(SAMPLE.read_text())
    model = json.loads(MODEL.read_text())

    if sample["features"] != model["features"]:
        sys.exit("feature order differs between the sample and the shipped model:\n"
                 f"  sample: {sample['features']}\n  model:  {model['features']}")
    rows = sample["rows"]
    print(f"{len(rows)} held-out rows, {len(model['features'])} features, "
          f"{len(model['trees'])} trees")

    scorer = (ROOT / "site" / "scorer.js").resolve().as_uri()
    driver = f"""
import {{score}} from '{scorer}';
import fs from 'fs';
const model = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
const rows = JSON.parse(fs.readFileSync(process.argv[3], 'utf8'));
const names = model.features;
console.log(JSON.stringify(rows.map(x => {{
  const f = {{}};
  names.forEach((n, i) => {{ f[n] = x[i]; }});
  return score(model, f);
}})));
"""
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump([r["x"] for r in rows], fh); xs = fh.name
    with tempfile.NamedTemporaryFile("w", suffix=".mjs", delete=False) as fh:
        fh.write(driver); drv = fh.name
    res = subprocess.run(["node", drv, str(MODEL), xs], capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit("node failed:\n" + res.stderr[:2500])
    got = json.loads(res.stdout)

    worst, where = 0.0, -1
    for i, (r, g) in enumerate(zip(rows, got)):
        d = abs(r["p"] - g)
        if d > worst:
            worst, where = d, i
    print(f"worst absolute score difference: {worst:.3e}")

    # The bands are the thing a person sees, so check they agree as labels too,
    # not just as numbers near each other.
    thr = model["thresholds"]

    def band(p):
        return "high" if p >= thr["high"] else "elevated" if p >= thr["elevated"] \
            else "ordinary"

    flips = [i for i, (r, g) in enumerate(zip(rows, got))
             if band(r["p"]) != band(g)]
    if flips:
        i = flips[0]
        print(f"\nETH PARITY FAILED - {len(flips)} rows land in different bands, "
              f"e.g. row {i}: python {rows[i]['p']:.9f} ({band(rows[i]['p'])}) "
              f"vs browser {got[i]:.9f} ({band(got[i])})")
        sys.exit(1)
    if worst > 1e-9:
        print(f"\nETH PARITY FAILED - scores differ by {worst:.3e} at row {where}: "
              f"python {rows[where]['p']!r} vs browser {got[where]!r}")
        sys.exit(1)
    print("\nETH PARITY OK - the browser scores the Ethereum model identically.")


if __name__ == "__main__":
    main()

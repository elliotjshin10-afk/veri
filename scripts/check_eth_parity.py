"""Prove the browser scores the Ethereum models exactly as LightGBM does.

model_eth.json and model_pair_eth.json are hand-rolled exports of LightGBM's
trees, walked by the browser's own `score()`. The Tron models have been checked
against the library that produced them since the start; the Ethereum ones
shipped without that check, which left every live Ethereum verdict resting on an
unverified re-implementation of someone else's tree format.

The sample is the 400 held-out rows closest to a band edge - where a tiny
disagreement is the difference between "ordinary" and "elevated", and so where a
bug would actually change what a person is told.

Run: python scripts/check_eth_parity.py
"""
from __future__ import annotations

import json, pathlib, subprocess, sys, tempfile
sys.path.insert(0, "src")

from veridis.config import PROCESSED, ROOT, SITE

# Both Ethereum models, checked by one script rather than two near-identical
# ones. The destination model answers an address lookup; the pair model answers
# a real send with the sender known. They share this code path because they
# share the failure: a tree-walk that disagrees with the library that produced
# the trees.
MODELS = (
    ("destination", PROCESSED / "eth_parity_sample.json", SITE / "model_eth.json",
     "scripts/m7_pit_model.py", True),
    ("two-sided", PROCESSED / "eth_pair_parity_sample.json",
     SITE / "model_pair_eth.json", "scripts/m10_eth_pair_model.py", False),
)


def check_one(label: str, sample_path, model_path, built_by: str,
              required: bool) -> bool:
    """True if checked and identical; False if it failed. A model that is not
    shipped yet is skipped, not failed - the page degrades without the pair
    model by design, so its absence is a defined state."""
    if not model_path.exists() or not sample_path.exists():
        if required:
            sys.exit(f"no {model_path.name}/{sample_path.name} - run {built_by}")
        print(f"\n{label}: not shipped yet (build it with {built_by}) - skipped")
        return True
    print(f"\n{label} model: {model_path.name}")
    sample = json.loads(sample_path.read_text())
    model = json.loads(model_path.read_text())

    if sample["features"] != model["features"]:
        print("  FAILED - feature order differs between sample and shipped model")
        return False
    rows = sample["rows"]
    print(f"  {len(rows)} held-out rows, {len(model['features'])} features, "
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
    res = subprocess.run(["node", drv, str(model_path), xs],
                         capture_output=True, text=True)
    if res.returncode != 0:
        print("  node failed:\n" + res.stderr[:1500])
        return False
    got = json.loads(res.stdout)

    worst, where = 0.0, -1
    for i, (r, g) in enumerate(zip(rows, got)):
        d = abs(r["p"] - g)
        if d > worst:
            worst, where = d, i
    print(f"  worst absolute score difference: {worst:.3e}")

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
        print(f"  FAILED - {len(flips)} rows land in different bands, e.g. row {i}: "
              f"python {rows[i]['p']:.9f} ({band(rows[i]['p'])}) "
              f"vs browser {got[i]:.9f} ({band(got[i])})")
        return False
    if worst > 1e-9:
        print(f"  FAILED - scores differ by {worst:.3e} at row {where}: "
              f"python {rows[where]['p']!r} vs browser {got[where]!r}")
        return False
    print("  identical")
    return True


def main() -> None:
    ok = True
    for args in MODELS:
        ok = check_one(*args) and ok
    if not ok:
        print("\nETH PARITY FAILED")
        sys.exit(1)
    print("\nETH PARITY OK - the browser scores every shipped Ethereum model "
          "identically.")


if __name__ == "__main__":
    main()

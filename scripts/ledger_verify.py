"""Recompute both chains and FAIL if either is broken.

This was a one-liner in the Makefile that printed the result and exited zero,
which meant the workflow step guarding the ledger could not actually stop a
run - a tampered chain would have printed "False" and sailed through to the
commit. A check that cannot fail is decoration.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from veridis.config import ROOT
from veridis.ledger import Ledger, predictions_path, resolutions_path


def main() -> None:
    bad = False
    for name, path in (("predictions", predictions_path(ROOT)),
                       ("resolutions", resolutions_path(ROOT))):
        ok, msg = Ledger(path).verify()
        print(f"  {'ok  ' if ok else 'BAD '} {name:12s} {msg}")
        bad |= not ok
    if bad:
        sys.exit("chain verification failed - refusing to continue")
    print("both chains intact")


if __name__ == "__main__":
    main()

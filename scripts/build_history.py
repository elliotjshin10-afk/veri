"""Build the decided-cases log: what we said, and what happened afterwards.

A risk score is a claim about the future, and the only honest way to show one is
beside the future it claimed. So this takes real transfers from the held-out
half of the data, keeps only those going to a destination the model never saw
in training, scores each with the exact artefact the browser ships
(site/model_pair.json, walked by the same code path), and pairs the verdict with
what the chain did next: the address was frozen N days later, or it never was.

Two rules keep this from becoming a highlight reel.

The cases are drawn at random inside each verdict/outcome stratum with a fixed
seed, never chosen by eye. And both kinds of mistake are in the list on purpose:
scams that read ordinary, and ordinary transfers that drew a warning. A log
showing only the wins would be worth nothing to anyone deciding whether to trust
the thing - and the rates reported beside the log are computed over all 4,337
qualifying held-out transfers, not over the 18 shown, so the sample cannot
flatter them.

Writes site/history.json. Run: python scripts/build_history.py
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys
import time

sys.path.insert(0, "src")

import lightgbm as lgb
import numpy as np
import polars as pl

from veridis.config import INTERIM, PROCESSED, SITE
from veridis.features.asof import FEATURE_COLUMNS, FEATURE_FAMILY
from veridis.model import browser_model
from veridis.model.reasons import ReasonEngine
from veridis.model.train import temporal_split

PAIR = [c for c in FEATURE_COLUMNS
        if FEATURE_FAMILY[c] in ("destination", "context", "relationship")
        and c != "dest_funder_fanout"]

# (label, band, how many) - the shape of the sample, fixed before looking at any
# of it. The two mistake strata are not optional: if the data has them the log
# shows them.
WANT = [
    (1, "high", 6),
    (1, "elevated", 3),
    (1, "ordinary", 2),
    (0, "ordinary", 5),
    (0, "elevated", 1),
    (0, "high", 1),
]
SEED = 17


def main() -> None:
    model = browser_model.load(SITE / "model_pair.json")
    if model["features"] != PAIR:
        sys.exit("site/model_pair.json has a different feature list than this script")

    events = pl.read_parquet(PROCESSED / "events_features.parquet")
    sp = temporal_split(events)
    test = sp.test

    X = test.select(PAIR).to_numpy()
    p = np.array(browser_model.score(model, X.tolist()))
    thr = model["thresholds"]
    bands = np.where(p >= thr["high"], "high",
                     np.where(p >= thr["elevated"], "elevated", "ordinary"))
    test = test.with_columns(pl.Series("p", p), pl.Series("band", bands))

    # The split is temporal, so a destination can appear on both sides of it -
    # 1,408 of the 5,745 held-out transfers go to an address the model has seen
    # before. Those are legitimate test events (nothing after the cutoff is
    # known at training time) but they cannot carry the stronger claim, so the
    # log is drawn only from transfers to destinations training never saw, and
    # the rates printed beside it are computed on that same stricter set.
    trained_on = set(sp.train["destination"].to_list())
    test = test.with_columns(
        pl.col("destination").is_in(list(trained_on)).not_().alias("unseen"))
    unseen = test.filter("unseen")
    if not unseen.height:
        sys.exit("no held-out transfers to an unseen destination")

    def tally(df: pl.DataFrame) -> dict:
        y, b = df["label"].to_numpy(), df["band"].to_numpy()
        out = {"n": int(df.height), "n_scam": int((y == 1).sum()),
               "n_control": int((y == 0).sum())}
        for lab, key in ((1, "scam"), (0, "control")):
            for name in ("high", "elevated", "ordinary"):
                out[f"{key}_{name}"] = int(((b == name) & (y == lab)).sum())
        return out

    summary = tally(unseen)
    summary_all = tally(test)

    booster = lgb.Booster(model_file=str(PROCESSED / "model_pair.txt"))
    engine = ReasonEngine(booster, PAIR)

    # Labels are a snapshot. Say which one, so "never listed" carries its date.
    sa = pl.read_parquet(INTERIM / "scam_addresses.parquet")
    labels_to = int(sa.filter(pl.col("chain") == "tron")["first_reported_at"].max())

    rng = np.random.default_rng(SEED)
    cases, used_dest, used_sender = [], set(), set()
    for lab, b, k in WANT:
        pool = (unseen.filter((pl.col("label") == lab) & (pl.col("band") == b))
                .with_row_index("ix"))
        if not pool.height:
            print(f"  no held-out cases at label={lab} band={b}")
            continue
        order = rng.permutation(pool.height)
        taken = 0
        for ix in order:
            row = pool.row(int(ix), named=True)
            # One row per relationship: six transfers from one victim is one
            # story told six times, and reads as padding.
            if row["destination"] in used_dest or row["sender"] in used_sender:
                continue
            cases.append(row)
            used_dest.add(row["destination"])
            used_sender.add(row["sender"])
            taken += 1
            if taken >= k:
                break
        if taken < k:
            print(f"  only {taken} of {k} available at label={lab} band={b}")

    Xc = np.array([[c[f] for f in PAIR] for c in cases], dtype=float)
    warn = engine.explain(Xc, cases, top_n=3)
    clear = engine.reassure(Xc, cases, top_n=3)

    out = []
    for c, w, cl in zip(cases, warn, clear):
        # The sentence explains the VERDICT, not the outcome: a warning is
        # justified by what raised it, a clearance by what held it down.
        why = (w[0] if w else None) if c["band"] != "ordinary" else (cl[0] if cl else None)
        if why is None:
            why = ("Several weaker signals, none decisive on its own"
                   if c["band"] != "ordinary"
                   else "Nothing in either address's history stood out")
        froze = int(c["frozen_at"]) if c["frozen_at"] is not None else None
        out.append({
            "t": int(c["event_time"]),
            "amt": round(float(c["amount_usd"]), 2),
            "from": c["sender"],
            "to": c["destination"],
            "k": int(c["transfer_k"]),
            "band": c["band"],
            "p": round(float(c["p"]), 4),
            "scam": int(c["label"]),
            "why": why.rstrip("."),
            "froze": froze,
            "lead_days": round((froze - int(c["event_time"])) / 86_400_000, 1)
                         if froze else None,
            "unlisted_days": None if froze
                             else round((labels_to - int(c["event_time"])) / 86_400_000, 1),
        })
    out.sort(key=lambda r: -r["t"])

    doc = {
        "built_at": int(time.time() * 1000),
        "labels_current_to": labels_to,
        "chain": "tron",
        "asset": "USDT",
        "model": {
            "file": "model_pair.json",
            "sha256_16": hashlib.sha256(
                (SITE / "model_pair.json").read_bytes()).hexdigest()[:16],
            "thresholds": thr,
            "evaluation": model.get("evaluation", {}),
        },
        "summary": summary,
        "summary_all_heldout": summary_all,
        "cases": out,
    }
    path = SITE / "history.json"
    path.write_text(json.dumps(doc, separators=(",", ":")))

    print(f"held out: {summary_all['n']:,} transfers; "
          f"{summary['n']:,} of them to a destination training never saw "
          f"({summary['n_scam']:,} scam / {summary['n_control']:,} ordinary)")
    print(f"  scam    stopped {summary['scam_high']:>4}  warned {summary['scam_elevated']:>4}"
          f"  through {summary['scam_ordinary']:>4}")
    print(f"  ordinary stopped {summary['control_high']:>3}  warned {summary['control_elevated']:>4}"
          f"  through {summary['control_ordinary']:>4}")
    print(f"\nwrote {path} - {len(out)} cases")
    for r in out:
        tag = ("frozen +%.0fd" % r["lead_days"]) if r["froze"] else "never listed"
        print(f"  {r['band']:<9}{'scam' if r['scam'] else 'ord ':<5}"
              f"${r['amt']:>12,.0f}  {tag:<14} {r['why'][:52]}")


if __name__ == "__main__":
    main()

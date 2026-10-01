"""Does the method work on Ethereum, or only on Tron?

The earlier cross-chain test took the Tron model, changed nothing, and scored
Ethereum: ROC-AUC 0.682 against 0.947. That measured TRANSFER, not the method -
and the same experiment on Tron would have looked just as bad, because the
model was trained on the wrong task there too.

This retrains instead. Same point-in-time protocol as Tron: frozen addresses
probed at several horizons before the freeze that listed them, ordinary wallets
probed before a pseudo-freeze drawn from the positives' own dates, split
temporally on the mark date so an address sits on one side only.

ONE DIFFERENCE, FORCED BY THE DATA SOURCE. Blockscout pages newest-first, so a
truncated history is missing an address's EARLIEST activity - age, lifetime
counts and every "since first seen" feature are wrong at every point in time,
not merely late. There is no as-of cutoff that repairs that, so only complete
histories are used. It costs more than a third of the addresses and it is not
optional.
"""
from __future__ import annotations

import json, sys
sys.path.insert(0, "src")
import lightgbm as lgb
import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score, roc_auc_score

from veridis.config import INTERIM, PROCESSED, SITE
from veridis.dataset.holdings import (ETH_INDEPENDENT_SET, eth_complete,
                                      eth_transfers)
from veridis.features.asof import FeatureEngine
from veridis.model.address_risk import DEST_FEATURES

DAY = 86_400_000
HORIZONS = [7, 30, 90, 180]
TEST_FRAC = 0.30
SEED = 17
BROWSER = [f for f in DEST_FEATURES if f != "dest_funder_fanout"]

# What we hold lives in veridis.dataset.holdings, not here. This script used to
# keep its own copy of the list, which is how the Tron side ended up with three
# scripts that disagreed about the data and three published numbers to match.
INDEPENDENT = ETH_INDEPENDENT_SET


def slim(t):
    def node(n):
        if "leaf_value" in n:
            return {"v": n["leaf_value"]}
        return {"f": n["split_feature"], "t": n["threshold"],
                "d": 1 if n.get("default_left") else 0,
                "l": node(n["left_child"]), "r": node(n["right_child"])}
    return node(t["tree_structure"])


def main() -> None:
    rng = np.random.default_rng(SEED)
    tf = eth_transfers()
    complete = eth_complete()

    span = (pl.concat([
        tf.select(pl.col("to_address").alias("address"), "block_time"),
        tf.select(pl.col("from_address").alias("address"), "block_time")],
        how="vertical_relaxed").group_by("address")
        .agg(pl.col("block_time").min().alias("first_seen"),
             pl.col("block_time").max().alias("last_seen")))

    sa = pl.read_parquet(INTERIM / "scam_addresses.parquet")
    scam = set(sa.filter(pl.col("accepted"))["address"].to_list())
    pos = (sa.filter((pl.col("chain") == "ethereum") & pl.col("accepted")
                     & pl.col("first_reported_at").is_not_null())
           .select("address", pl.col("first_reported_at").alias("mark"))
           .filter(pl.col("address").is_in(list(complete))))
    ctrl: set[str] = set()
    for n in ("eth_control_seeds", "eth_control_peers", "eth_control2_truncated",
              INDEPENDENT):
        p = INTERIM / f"{n}.parquet"
        if p.exists():
            ctrl |= set(pl.read_parquet(p)["address"].to_list())
    ctrl = (ctrl & complete) - scam
    ip = INTERIM / f"{INDEPENDENT}.parquet"
    independent = (set(pl.read_parquet(ip)["address"].to_list()) & ctrl) if ip.exists() else set()
    print(f"complete histories: {len(pos):,} frozen, {len(ctrl):,} ordinary "
          f"({tf.height:,} transfers)")
    # A pseudo-freeze must sit inside the control's own life. Drawing one blind
    # from the positives put it before the address existed, every probe was
    # dropped by the coverage guard, and the negative arm came back empty - a
    # silent failure that reads as a LightGBM error about scale_pos_weight.
    marks = np.sort(pos["mark"].to_numpy())
    first = dict(zip(span["address"].to_list(), span["first_seen"].to_list()))
    rowsn, skipped = [], 0
    for a in sorted(ctrl):
        f = first.get(a)
        if f is None:
            skipped += 1
            continue
        usable = marks[marks - max(HORIZONS) * DAY > f]
        if not len(usable):
            skipped += 1
            continue
        rowsn.append({"address": a, "mark": int(rng.choice(usable))})
    neg = pl.DataFrame(rowsn)
    print(f"ordinary wallets usable: {neg.height:,} "
          f"({skipped:,} too young for any pseudo-freeze date)")
    if neg.height < 150:
        sys.exit("too few usable ordinary wallets - run scripts/m7_controls_more.py")

    rows = []
    for df, label in ((pos, 1), (neg, 0)):
        d = df.join(span, on="address", how="left").filter(pl.col("first_seen").is_not_null())
        for r in d.iter_rows(named=True):
            for h in HORIZONS:
                t = r["mark"] - h * DAY
                if t <= r["first_seen"]:
                    continue
                rows.append({"address": r["address"], "label": label, "event_time": t,
                             "mark": r["mark"]})
    pr = pl.DataFrame(rows)
    cut = float(np.quantile(pr["mark"].to_numpy(), 1 - TEST_FRAC))
    is_test = pr["mark"].to_numpy() >= cut
    y = pr["label"].to_numpy()
    print(f"probes {pr.height:,} over {pr['address'].n_unique():,} addresses; "
          f"test {int(is_test.sum()):,} "
          f"({int((y[is_test]==1).sum()):,} frozen / {int((y[is_test]==0).sum()):,} ordinary)")

    ev = (pr.select(pl.col("address").alias("destination"), "event_time")
          .with_columns(pl.lit("ethereum").alias("chain"), pl.lit("__probe__").alias("sender"),
                        pl.lit(1000.0).alias("amount_usd")).with_row_index("event_id"))
    engine = FeatureEngine(tf)
    feats = engine.compute(ev)
    engine.close()

    out = {}
    print(f"\nindependent ordinary wallets available: {len(independent):,}")
    print("\n  arm        ROC-AUC  PR-AUC   TPR@5%FPR   (pos/neg)")
    for name, cols in (("full   ", DEST_FEATURES), ("browser", BROWSER)):
        X = feats.select(cols).to_numpy()
        ytr = y[~is_test]
        b = lgb.train({
            "objective": "binary", "learning_rate": 0.05, "num_leaves": 31,
            "min_data_in_leaf": 30, "feature_fraction": 0.8, "bagging_fraction": 0.8,
            "bagging_freq": 1, "lambda_l2": 1.0,
            "scale_pos_weight": (ytr == 0).sum() / max((ytr == 1).sum(), 1),
            "verbose": -1, "seed": SEED, "num_threads": 4,
        }, lgb.Dataset(X[~is_test], label=ytr, feature_name=cols), num_boost_round=400)
        s, yt = b.predict(X[is_test]), y[is_test]
        addr_t = np.array(pr["address"].to_list())[is_test]
        res = {}
        for tag, mask in (("all", np.ones_like(yt, bool)),
                          ("independent",
                           np.array([a in independent for a in addr_t]) | (yt == 1))):
            if len(set(yt[mask])) < 2:
                continue
            sm, ym = s[mask], yt[mask]
            thr = float(np.quantile(np.sort(sm[ym == 0]), 0.95))
            res[tag] = {"roc_auc": float(roc_auc_score(ym, sm)),
                        "pr_auc": float(average_precision_score(ym, sm)),
                        "tpr_at_5pct_fpr": float((sm[ym == 1] >= thr).mean()),
                        "n_pos": int((ym == 1).sum()), "n_neg": int((ym == 0).sum())}
        out[name.strip()] = res
        for tag, r in res.items():
            lab = name if tag == "all" else "   \u21b3 indep"
            print(f"  {lab}   {r['roc_auc']:.4f}  {r['pr_auc']:.4f}   "
                  f"{r['tpr_at_5pct_fpr']:.1%}   ({r['n_pos']:,}/{r['n_neg']:,})")
        if name.strip() == "full":
            b.save_model(str(PROCESSED / "model_eth_pit.txt"))
        else:
            # Bands from the INDEPENDENT controls only: "high" must mean "in
            # the top 1% of ordinary wallets", and the scam-adjacent pool is
            # not that population.
            ind = np.array([a in independent for a in addr_t]) & (yt == 0)
            pool = s[ind] if ind.sum() >= 150 else s[yt == 0]
            thr = {"elevated": float(np.quantile(np.sort(pool), 0.90)),
                   "high": float(np.quantile(np.sort(pool), 0.99))}
            browser_export = (b, thr, res.get("independent") or res.get("all"),
                              int(ind.sum()))

    # Only ship a browser model once it has been judged against ordinary
    # wallets. Shipping one scored against a scam collector's own neighbours
    # would be the cross-chain mistake again, wearing a better number.
    b, thr, ev, n_ind = browser_export
    if ev and n_ind >= 150:
        SITE.joinpath("model_eth.json").write_text(json.dumps(
            {"features": BROWSER, "trees": [slim(t) for t in b.dump_model()["tree_info"]],
             "thresholds": thr, "evaluation": {**ev, "chain": "ethereum",
                                               "controls": "independent"}},
            separators=(",", ":")))
        print(f"wrote site/model_eth.json (judged on {n_ind:,} independent wallets)")
        # A sample of held-out rows with LightGBM's own score, so the browser's
        # tree-walk can be checked against the library that produced it. The
        # Tron models have had this check since the start; the Ethereum model
        # shipped without one, which left the live Ethereum verdict resting on
        # an unverified re-implementation.
        Xb = feats.select(BROWSER).to_numpy()[is_test]
        # Nearest an actual band edge, not nearest 0.5: the bands sit at the 90th
        # and 99th percentile of the control scores, so 0.5 is nowhere near a
        # decision and rows picked around it would prove the least interesting
        # part of the range. These are the rows where a disagreement in the last
        # decimal place changes what a person is told.
        edge = np.minimum(np.abs(s - thr["elevated"]), np.abs(s - thr["high"]))
        take = np.argsort(edge)[:400]
        PROCESSED.joinpath("eth_parity_sample.json").write_text(json.dumps(
            {"features": BROWSER,
             "rows": [{"x": [float(v) for v in Xb[i]], "p": float(s[i])}
                      for i in take]}, separators=(",", ":")))
        print(f"wrote eth_parity_sample.json ({len(take)} rows for browser parity)")

        # What the SHIPPED bands actually do. tpr_at_5pct_fpr is a comparable
        # research number, but nobody sees a 5% FPR - they see "high" and
        # "elevated", set at the 99th and 90th percentile of ordinary wallets.
        # Those two rows are the ones the product can be held to.
        ind_mask = np.array([a in independent for a in addr_t])
        bands = {}
        for label, lo in (("high", thr["high"]), ("elevated", thr["elevated"])):
            caught = float((s[yt == 1] >= lo).mean())
            fp = float((s[(yt == 0) & ind_mask] >= lo).mean()) if ind_mask.any() else None
            bands[label] = {"threshold": lo, "frozen_caught": caught,
                            "ordinary_flagged": fp}
            print(f"  {label:>8} band: catches {caught:6.1%} of frozen addresses, "
                  f"flags {fp:5.1%} of ordinary ones")
        out["bands"] = bands
    else:
        print(f"NOT writing site/model_eth.json - only {n_ind} independent "
              f"ordinary wallets in the test, too few to set a band on")

    (PROCESSED / "eth_pit_report.json").write_text(json.dumps(
        {"results": out, "horizons": HORIZONS, "complete_only": True,
         "n_pos_addresses": len(pos), "n_ctrl_addresses": len(ctrl)}, indent=2))
    # The held-out addresses, written down. Without this list there is no way to
    # demo the model honestly: every Ethereum address we have is either one it
    # trained on or one nobody has checked, and picking a frozen address at
    # random to show off is as likely to pick a training example as not.
    addr_all = np.array(pr["address"].to_list())
    holdout = {
        "cut_freeze_date_ms": int(cut),
        "frozen": sorted(set(addr_all[is_test & (y == 1)].tolist())),
        "ordinary": sorted(set(addr_all[is_test & (y == 0)].tolist())),
    }
    PROCESSED.joinpath("eth_holdout.json").write_text(json.dumps(holdout, indent=1))
    print(f"wrote eth_holdout.json ({len(holdout['frozen']):,} frozen, "
          f"{len(holdout['ordinary']):,} ordinary - none seen in training)")

    print(f"\nwrote model_eth_pit.txt and eth_pit_report.json")


if __name__ == "__main__":
    main()

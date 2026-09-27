"""End-to-end smoke test on synthetic chain data.

Builds a small world containing the behavioural signature we claim to detect
(escalating transfers into fresh, fast-forwarding collection addresses) plus
ordinary payment traffic, then runs the real pipeline over it: events ->
as-of-time features -> matched controls -> temporal split -> model -> metrics
-> reason codes. This catches integration breakage without waiting on ingest.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from veridis.dataset.events import build_control_events, build_positive_events, extract_victims  # noqa: E402
from veridis.dataset.matching import match_controls  # noqa: E402
from veridis.features.asof import FEATURE_COLUMNS, FeatureEngine  # noqa: E402
from veridis.model.evaluate import early_detection  # noqa: E402
from veridis.model.reasons import ReasonEngine  # noqa: E402
from veridis.model.train import calibrate, evaluate_all, temporal_split, train_model  # noqa: E402

DAY = 86_400_000
T0 = 1_700_000_000_000


def _world(seed: int = 3) -> tuple[pl.DataFrame, pl.DataFrame]:
    rng = random.Random(seed)
    tx: list[dict] = []
    n = 0

    def add(f, t, a, bt):
        nonlocal n
        tx.append({"chain": "tron", "tx_hash": f"h{n}", "from_address": f,
                   "to_address": t, "amount_usd": float(a), "asset": "USDT",
                   "block_time": int(bt)})
        n += 1

    scam_rows = []
    # 24 scam operations, each fresh, each collecting from several victims and
    # sweeping onward to one address.
    for s in range(24):
        scam = f"scam{s}"
        born = T0 + rng.randint(0, 300) * DAY
        frozen = born + rng.randint(60, 120) * DAY
        scam_rows.append({"address": scam, "chain": "tron",
                          "first_reported_at": frozen})
        for v in range(rng.randint(4, 7)):
            victim = f"v{s}_{v}"
            start = born + rng.randint(1, 20) * DAY
            # ordinary prior life
            for j in range(rng.randint(4, 10)):
                add(victim, f"normal{rng.randint(0, 40)}",
                    rng.uniform(20, 200), start - rng.randint(30, 400) * DAY)
            add(f"exchange{rng.randint(0,3)}", victim, rng.uniform(3000, 40000),
                start - rng.randint(1, 20) * DAY)
            # the signature: small test transfer, then escalation
            amt = rng.uniform(80, 250)
            t = start
            for k in range(rng.randint(3, 6)):
                add(victim, scam, amt, t)
                add(scam, f"collector{s % 3}", amt * 0.98, t + rng.randint(60, 900) * 1000)
                amt *= rng.uniform(2.5, 5.0)
                t += rng.randint(1, 6) * DAY

    # ordinary payment traffic between long-lived addresses
    for i in range(260):
        a = f"norm{i}"
        for j in range(rng.randint(4, 14)):
            b = f"norm{rng.randint(0, 259)}"
            if a == b:
                continue
            # spans the same window as the scam activity, as real controls do
            add(a, b, rng.uniform(30, 300), T0 + rng.randint(0, 400) * DAY)
        # some ordinary senders also receive a large on-ramp, like victims do
        if i % 3 == 0:
            add(f"exchange{rng.randint(0,3)}", a, rng.uniform(3000, 40000),
                T0 + rng.randint(0, 200) * DAY)

    schema = {"chain": pl.Utf8, "tx_hash": pl.Utf8, "from_address": pl.Utf8,
              "to_address": pl.Utf8, "amount_usd": pl.Float64, "asset": pl.Utf8,
              "block_time": pl.Int64}
    return pl.DataFrame(tx, schema=schema), pl.DataFrame(scam_rows)


def test_pipeline_learns_the_signature_end_to_end():
    transfers, scam = _world()
    victims = extract_victims(transfers, scam)
    assert victims.height > 40, "synthetic world should produce victims"

    pos = build_positive_events(transfers, victims)
    neg = build_control_events(
        transfers,
        exclude_senders=victims["victim_address"],
        scam_addresses=scam["address"],
        n_events=pos.height * 30,
    )
    events = pl.concat([pos, neg], how="vertical_relaxed").with_row_index("event_id")

    engine = FeatureEngine(transfers)
    feats = engine.compute(events)
    engine.close()
    assert feats.height == events.height
    assert not feats.select(FEATURE_COLUMNS).to_numpy().dtype == object

    matched = match_controls(feats, ratio=10)
    assert int((matched["label"] == 1).sum()) == pos.height

    split = temporal_split(matched, quantile=0.6)
    assert split.train.height and split.test.height

    booster = train_model(split.train)
    iso = calibrate(booster, split.train)
    report, scored = evaluate_all(booster, iso, split, target_fpr=0.01)

    # Group integrity is asserted inside evaluate_all; surface it explicitly.
    assert report["groups"]["scam_address_overlap"] == 0
    # The signature is deliberately strong here, so a working pipeline must
    # beat the base rate by a wide margin. This is a smoke test, not a claim
    # about real-world performance.
    assert report["metrics"]["pr_auc"] > 3 * report["metrics"]["base_rate"]

    ed = early_detection(scored, report["metrics"]["threshold"])
    assert ed.height == 4

    reasons = ReasonEngine(booster, FEATURE_COLUMNS).explain(
        scored.select(FEATURE_COLUMNS).to_numpy()[:5], scored.head(5).to_dicts()
    )
    assert len(reasons) == 5
    assert all(isinstance(r, list) for r in reasons)

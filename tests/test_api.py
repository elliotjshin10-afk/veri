"""The /score contract.

Exercises the real endpoint against a small in-memory model and warehouse, so
the request/response shape, banding and reason rendering are covered without
depending on a trained artefact on disk.
"""
from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fastapi.testclient import TestClient  # noqa: E402

from veridis.api import main as api  # noqa: E402
from veridis.api.live import LiveWarehouse  # noqa: E402
from veridis.features.asof import FEATURE_COLUMNS  # noqa: E402
from veridis.model.reasons import ReasonEngine  # noqa: E402

DAY = 86_400_000
T0 = 1_700_000_000_000


@pytest.fixture(scope="module")
def client():
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from test_endtoend import _world  # reuse the synthetic chain world

    from veridis.dataset.events import build_control_events, build_positive_events, extract_victims
    from veridis.dataset.matching import match_controls
    from veridis.model.train import calibrate, temporal_split, train_model

    transfers, scam = _world()
    victims = extract_victims(transfers, scam)
    pos = build_positive_events(transfers, victims)
    neg = build_control_events(transfers, victims["victim_address"],
                              scam["address"], n_events=pos.height * 30)
    events = pl.concat([pos, neg], how="vertical_relaxed").with_row_index("event_id")
    warehouse = LiveWarehouse(transfers)
    feats = warehouse.engine.compute(events)
    matched = match_controls(feats, ratio=10)
    split = temporal_split(matched, quantile=0.6)
    booster = train_model(split.train)

    state = {
        "booster": booster,
        "iso": calibrate(booster, split.train),
        "warehouse": warehouse,
        "labels": set(),
        "reasons": ReasonEngine(booster, FEATURE_COLUMNS),
        "thresholds": {"caution": 0.30, "high_risk": 0.70},
        "version": "test",
        "n_transfers": transfers.height,
    }
    # Enter the app's lifespan FIRST, then install our state. Startup loads
    # whatever model happens to be on disk, which would otherwise clobber the
    # fixture and make this test depend on the last training run's feature
    # count rather than on the code under test.
    with TestClient(api.app) as c:
        api.STATE.clear()
        api.STATE.update(state)
        yield c
    warehouse.engine.close()


def test_health_reports_loaded_model(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_score_returns_the_documented_shape(client):
    r = client.post("/score", json={
        "chain": "tron", "sender": "v0_0", "destination": "scam0",
        "amount_usd": 5000, "asset": "USDT",
    })
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {
        "score", "band", "reasons", "model_version", "computed_at",
        "latency_ms", "features_available",
        # "indexed" vs "cold_fetch": a caller must be able to tell a warm
        # answer from one that waited on a chain fetch.
        "source", "fetch_ms",
    }
    assert body["source"] in {"indexed", "cold_fetch"}
    assert 0.0 <= body["score"] <= 1.0
    assert body["band"] in {"safe", "caution", "high_risk"}
    assert isinstance(body["reasons"], list) and len(body["reasons"]) <= 5
    assert all(isinstance(x, str) and x for x in body["reasons"])


def test_unknown_addresses_say_so_rather_than_inventing_reasons(client):
    r = client.post("/score", json={
        "chain": "tron", "sender": "never_seen_sender",
        "destination": "never_seen_destination", "amount_usd": 1000,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["features_available"] is False
    assert "No prior on-chain history found" in body["reasons"][0]


def test_amount_must_be_positive(client):
    r = client.post("/score", json={
        "chain": "tron", "sender": "a", "destination": "b", "amount_usd": 0,
    })
    assert r.status_code == 422

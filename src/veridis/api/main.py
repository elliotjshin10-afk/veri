"""M5 - the scoring API.

One endpoint. A wallet calls it before a transfer executes and gets a number,
a band, and reasons the user can check themselves.

Feature computation runs against a local warehouse of indexed transfers - the
stand-in for the node/indexer a production deployment would already have. The
same FeatureEngine serves training and serving, so the features cannot drift
apart between the two.
"""
from __future__ import annotations

import logging
import pickle
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import lightgbm as lgb
import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from veridis.config import INTERIM, PROCESSED, TARGET_FPR
from veridis.api.live import LiveWarehouse, collection_verdict
from veridis.features.asof import FEATURE_COLUMNS
from veridis.model.reasons import ReasonEngine
from veridis.model.scoring import band as band_of, predict as predict_scores

log = logging.getLogger(__name__)

STATE: dict = {}


class ScoreRequest(BaseModel):
    chain: str = Field(default="tron")
    sender: str
    destination: str
    amount_usd: float = Field(gt=0)
    asset: str = Field(default="USDT")


class ScoreResponse(BaseModel):
    score: float
    band: str
    reasons: list[str]
    model_version: str
    computed_at: str
    latency_ms: float
    features_available: bool
    # "indexed" = both addresses already in the warehouse (the production path).
    # "cold_fetch" = we pulled history from chain first, which is seconds, not
    # milliseconds. Reported so the two are never confused.
    source: str = "indexed"
    fetch_ms: float | None = None


def load_state(version: str = "latest") -> dict:
    model_path = PROCESSED / f"model_{version}.txt"
    if not model_path.exists():
        raise FileNotFoundError(f"no model at {model_path}; run `make train` first")
    booster = lgb.Booster(model_file=str(model_path))
    calib_path = PROCESSED / f"calibrator_{version}.pkl"
    iso = pickle.loads(calib_path.read_bytes()) if calib_path.exists() else None

    transfers = pl.read_parquet(PROCESSED / "warehouse_transfers.parquet")
    # Only addresses whose own history the ingest fetched count as indexed.
    indexed: set[str] = set()
    for name in ("scam_truncated.parquet", "victim_truncated.parquet",
                 "control_truncated.parquet"):
        path = INTERIM / name
        if path.exists():
            indexed |= set(pl.read_parquet(path)["address"].to_list())
    log.info("warehouse: %d transfers, %d fully indexed addresses",
             transfers.height, len(indexed))
    warehouse = LiveWarehouse(transfers, indexed=indexed)

    thresholds_path = PROCESSED / f"thresholds_{version}.json"
    if thresholds_path.exists():
        import json
        thresholds = json.loads(thresholds_path.read_text())
    else:
        thresholds = {"caution": 0.30, "high_risk": 0.70}

    labels = pl.read_parquet(INTERIM / "scam_addresses.parquet").filter(
        pl.col("accepted"))
    return {
        "booster": booster,
        "iso": iso,
        "warehouse": warehouse,
        "labels": set(labels["address"].to_list()),
        "reasons": ReasonEngine(booster, FEATURE_COLUMNS),
        "thresholds": thresholds,
        "version": version,
        "n_transfers": transfers.height,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        STATE.update(load_state())
        log.info("loaded model %s over %d transfers",
                 STATE["version"], STATE["n_transfers"])
    except FileNotFoundError as exc:
        log.warning("starting without a model: %s", exc)
    yield
    if "warehouse" in STATE:
        STATE["warehouse"].engine.close()


app = FastAPI(title="Veridis pre-send risk API", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if "booster" in STATE else "no_model",
        "model_version": STATE.get("version"),
        "warehouse_transfers": (
            STATE["warehouse"].n_transfers if "warehouse" in STATE else 0),
        "target_fpr": TARGET_FPR,
    }


@app.post("/score", response_model=ScoreResponse)
async def score(req: ScoreRequest) -> ScoreResponse:
    if "booster" not in STATE:
        raise HTTPException(503, "model not loaded; run `make train`")
    t0 = time.perf_counter()

    # Fetch anything we have never indexed. Warm requests skip this entirely.
    fetch = await STATE["warehouse"].ensure([req.sender, req.destination])

    now_ms = int(time.time() * 1000)
    events = pl.DataFrame([{
        "event_id": 0,
        "chain": req.chain,
        "sender": req.sender,
        "destination": req.destination,
        "amount_usd": float(req.amount_usd),
        "event_time": now_ms,
    }], schema={
        "event_id": pl.Int64, "chain": pl.Utf8, "sender": pl.Utf8,
        "destination": pl.Utf8, "amount_usd": pl.Float64, "event_time": pl.Int64,
    })

    feats = STATE["warehouse"].engine.compute(events)
    X = feats.select(FEATURE_COLUMNS).to_numpy()
    raw, prob_arr = predict_scores(STATE["booster"], STATE["iso"], X)
    prob = float(prob_arr[0])

    row = feats.row(0, named=True)
    reasons = STATE["reasons"].explain(X, [row], top_n=5)[0]
    # Nothing known about either address is itself worth saying.
    known = bool(row.get("dest_inbound_count", 0) or row.get("sender_outbound_count", 0))
    if not known:
        reasons = ["No prior on-chain history found for this destination"] + reasons

    return ScoreResponse(
        score=round(prob, 4),
        band=band_of(float(raw[0]), STATE["thresholds"]),
        reasons=reasons[:5],
        model_version=STATE["version"],
        computed_at=datetime.now(timezone.utc).isoformat(),
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        features_available=known,
        source="cold_fetch" if fetch["cold"] else "indexed",
        fetch_ms=fetch["fetch_ms"] or None,
    )


class ResearchResponse(BaseModel):
    address: str
    chain: str
    known_to_us: bool
    on_scam_list: bool
    verdict: str
    observations: list[str]
    profile: dict
    source: str
    fetch_ms: float | None = None
    latency_ms: float


@app.get("/research", response_model=ResearchResponse)
async def research(address: str, chain: str = "tron") -> ResearchResponse:
    """Look up any wallet, with no sender and no transfer in mind.

    `/score` answers "should this person send?"; this answers "what is this
    address?". Everything returned is a fact about on-chain behaviour that the
    caller can check on a block explorer - deliberately rule-based rather than
    a model score, because a description of an address should not depend on
    whose transfer prompted it.
    """
    if "warehouse" not in STATE:
        raise HTTPException(503, "warehouse not loaded; run `make features`")
    t0 = time.perf_counter()
    wh = STATE["warehouse"]
    fetch = await wh.ensure([address])
    now_ms = int(time.time() * 1000)
    prof = wh.profile(address, now_ms)

    if prof is None:
        return ResearchResponse(
            address=address, chain=chain, known_to_us=False,
            on_scam_list=address in STATE["labels"],
            verdict="no stablecoin activity found",
            observations=["We found no TRC-20 stablecoin transfers for this address."],
            profile={}, source="cold_fetch" if fetch["cold"] else "indexed",
            fetch_ms=fetch["fetch_ms"] or None,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
        )

    verdict, notes = collection_verdict(prof)
    listed = address in STATE["labels"]
    if listed:
        notes.insert(0, "This address appears on a public scam or sanctions list.")
    return ResearchResponse(
        address=address, chain=chain, known_to_us=True, on_scam_list=listed,
        verdict=verdict, observations=notes,
        profile={k: v for k, v in prof.items() if v is not None},
        source="cold_fetch" if fetch["cold"] else "indexed",
        fetch_ms=fetch["fetch_ms"] or None,
        latency_ms=round((time.perf_counter() - t0) * 1000, 2),
    )

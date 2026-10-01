"""Disk-cached, rate-limited async HTTP with retry/backoff.

Free chain-API tiers throttle aggressively and the ingest is re-run many times
during development, so every response is cached to disk and never re-fetched.
The cache is keyed on the full request (url + sorted params), gzipped JSON.
"""
from __future__ import annotations

import asyncio
import gzip
import hashlib
import json
import logging
import time
from pathlib import Path
from collections.abc import Callable
from typing import Any

import httpx

from veridis.config import CACHE

log = logging.getLogger(__name__)


def _cache_key(url: str, params: dict[str, Any] | None) -> str:
    blob = url + "?" + json.dumps(params or {}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class TokenBucket:
    """Simple async token bucket; smooths bursts against per-IP rate limits."""

    def __init__(self, rate_per_sec: float, burst: int | None = None) -> None:
        self.rate = rate_per_sec
        self.capacity = burst if burst is not None else max(1.0, rate_per_sec)
        self._tokens = float(self.capacity)
        self._updated = time.monotonic()
        self._lock = asyncio.Lock()

    async def take(self) -> None:
        async with self._lock:
            while True:
                now = time.monotonic()
                self._tokens = min(
                    self.capacity, self._tokens + (now - self._updated) * self.rate
                )
                self._updated = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                await asyncio.sleep((1.0 - self._tokens) / self.rate)


class CachedClient:
    """Async JSON client with disk cache, token-bucket pacing and backoff."""

    def __init__(
        self,
        namespace: str,
        rate_per_sec: float = 8.0,
        concurrency: int = 6,
        timeout: float = 40.0,
        max_attempts: int = 8,
        backoff_start: float = 1.0,
        headers: dict[str, str] | None = None,
        offline: bool = False,
        is_refusal: Callable[[Any], bool] | None = None,
        burst: int | None = None,
    ) -> None:
        self.dir = CACHE / namespace
        self.dir.mkdir(parents=True, exist_ok=True)
        # burst=1 for an API with a hard per-second cap. The default capacity
        # equals the rate, so a client idle for a moment fires its whole second
        # of allowance at once - three requests inside 120ms, which a 3/sec
        # limit refuses even though the average rate is well under it.
        self.bucket = TokenBucket(rate_per_sec, burst=burst)
        self.sem = asyncio.Semaphore(concurrency)
        self.max_attempts = max_attempts
        self.backoff_start = backoff_start
        # Offline: serve only what is already cached and never touch the
        # network. Lets the pipeline be rebuilt from a partial ingest while a
        # quota window is exhausted, instead of blocking on it.
        self.offline = offline
        # Some APIs refuse INSIDE a 200: Etherscan answers an exceeded rate
        # limit with HTTP 200 and {status: "0", result: "Max calls per sec..."}.
        # Status-code retrying never sees it, so without this hook the refusal
        # is cached as though it were data - and a caller that reads the body
        # as "end of list" records a cut-short history as a complete one.
        self.is_refusal = is_refusal
        self._client = httpx.AsyncClient(
            timeout=timeout, headers=headers or {}, follow_redirects=True
        )
        self.stats = {"hit": 0, "miss": 0, "retry": 0, "error": 0, "absent": 0,
                      "refused": 0}

    async def __aenter__(self) -> "CachedClient":
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self._client.aclose()

    def _path(self, key: str) -> Path:
        return self.dir / key[:2] / f"{key}.json.gz"

    def peek(self, url: str, params: dict[str, Any] | None = None) -> Any | None:
        p = self._path(_cache_key(url, params))
        if p.exists():
            with gzip.open(p, "rt") as fh:
                return json.load(fh)
        return None

    def _store(self, key: str, payload: Any) -> None:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".tmp")
        with gzip.open(tmp, "wt") as fh:
            json.dump(payload, fh)
        tmp.replace(p)

    async def get_json(
        self, url: str, params: dict[str, Any] | None = None
    ) -> Any | None:
        key = _cache_key(url, params)
        path = self._path(key)
        if path.exists():
            self.stats["hit"] += 1
            with gzip.open(path, "rt") as fh:
                return json.load(fh)

        if self.offline:
            self.stats["absent"] += 1
            return None

        async with self.sem:
            delay = self.backoff_start
            for attempt in range(1, self.max_attempts + 1):
                await self.bucket.take()
                try:
                    resp = await self._client.get(url, params=params)
                except (httpx.TimeoutException, httpx.TransportError) as exc:
                    log.debug("transport error %s (attempt %d): %s", url, attempt, exc)
                else:
                    if resp.status_code == 200:
                        try:
                            payload = resp.json()
                        except ValueError:
                            log.warning("non-JSON 200 from %s", url)
                            self.stats["error"] += 1
                            return None
                        if self.is_refusal and self.is_refusal(payload):
                            # Never cached: a refusal is not an answer, and a
                            # cached one would be served forever.
                            self.stats["refused"] += 1
                            self.stats["retry"] += 1
                            if attempt == self.max_attempts:
                                break
                            await asyncio.sleep(delay)
                            delay = min(delay * 2, 120.0)
                            continue
                        self.stats["miss"] += 1
                        self._store(key, payload)
                        return payload
                    if resp.status_code in (404, 400):
                        # A genuinely absent resource: cache the negative so we
                        # do not re-ask on every run.
                        self.stats["miss"] += 1
                        self._store(key, None)
                        return None
                    if resp.status_code not in (429, 500, 502, 503, 504):
                        log.warning("http %s from %s", resp.status_code, url)
                        self.stats["error"] += 1
                        return None
                    retry_after = resp.headers.get("Retry-After")
                    if retry_after and retry_after.isdigit():
                        delay = max(delay, float(retry_after))

                self.stats["retry"] += 1
                if attempt == self.max_attempts:
                    break
                await asyncio.sleep(delay)
                delay = min(delay * 2, 120.0)

            self.stats["error"] += 1
            log.warning("giving up on %s after %d attempts", url, self.max_attempts)
            return None

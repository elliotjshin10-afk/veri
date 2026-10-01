"""A refusal must never be mistaken for data.

Etherscan does not use HTTP 429 for rate limiting. It answers HTTP 200 with
{"status": "0", "result": "Max calls per sec rate limit reached (3/sec)"}. Code
that only inspects the status code sees a success, and code that reads the body
as a list sees "end of history".

That combination silently recorded cut-short histories as complete ones: 932 of
2,277 cached responses were refusals, every one of them feeding the model a
wallet whose age and lifetime totals were a fraction of the truth. These tests
pin the three places that have to tell the difference.
"""
from __future__ import annotations

import asyncio

import pytest

from veridis.chain import etherscan as es

REFUSAL = {"status": "0", "message": "NOTOK",
           "result": "Max calls per sec rate limit reached (3/sec)"}
EMPTY = {"status": "0", "message": "No transactions found", "result": []}
EMPTY_STR = {"status": "0", "message": "No transactions found",
             "result": "No transactions found"}


def test_refusal_is_recognised():
    assert es.is_refusal(REFUSAL)


@pytest.mark.parametrize("payload", [
    {"status": "1", "message": "OK", "result": [{"hash": "0x1"}]},
    EMPTY, EMPTY_STR, None, [],
])
def test_ordinary_answers_are_not_refusals(payload):
    assert not es.is_refusal(payload)


def test_empty_history_is_distinguished_from_a_refusal():
    assert es._is_empty(EMPTY_STR)
    assert not es._is_empty(REFUSAL)


class _Stub:
    """Serves queued payloads, recording what was asked for."""

    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = 0

    async def get_json(self, url, params=None):
        self.calls += 1
        return self.payloads.pop(0) if self.payloads else {"status": "1", "result": []}


def _rows(n):
    return [{"tokenSymbol": "USDT", "tokenDecimal": "6", "value": "1000000",
             "timeStamp": str(1600000000 + i), "hash": f"0x{i:064x}",
             "from": f"0x{i:040x}", "to": f"0x{9:040x}"} for i in range(n)]


def test_a_refusal_mid_history_raises_rather_than_truncating():
    """The bug: page 2 refused, so the caller kept page 1 and reported it
    complete. A 7,936-transfer wallet became a 1,000-transfer wallet whose age
    was four years too young, and nothing in the data said so."""
    c = _Stub([{"status": "1", "result": _rows(es.PAGE)}, REFUSAL])
    with pytest.raises(RuntimeError, match="refused"):
        asyncio.run(es.token_transfers(c, "0xabc", "key", max_pages=4))


def test_an_empty_page_ends_the_history_as_complete():
    c = _Stub([{"status": "1", "result": _rows(es.PAGE)}, EMPTY_STR])
    rows, truncated = asyncio.run(
        es.token_transfers(c, "0xabc", "key", max_pages=4))
    assert len(rows) == es.PAGE
    assert truncated is False


def test_exhausting_the_page_budget_reports_truncated():
    c = _Stub([{"status": "1", "result": _rows(es.PAGE)} for _ in range(2)])
    rows, truncated = asyncio.run(
        es.token_transfers(c, "0xabc", "key", max_pages=2))
    assert len(rows) == 2 * es.PAGE
    assert truncated is True


def test_short_page_ends_the_history_as_complete():
    c = _Stub([{"status": "1", "result": _rows(es.PAGE - 1)}])
    rows, truncated = asyncio.run(
        es.token_transfers(c, "0xabc", "key", max_pages=4))
    assert truncated is False


def test_the_client_is_configured_for_the_real_limit():
    """The documented 5/sec is wrong; the refusal body states 3/sec. A client
    paced above that is refused constantly, and burst must be 1 - a full bucket
    fires a whole second of allowance at once."""
    c = es.client()
    assert c.bucket.rate <= 3.0
    assert c.bucket.capacity == 1
    assert c.is_refusal is not None


def test_a_refusal_is_retried_and_never_cached(tmp_path, monkeypatch):
    """The cache is the lasting damage: a stored refusal is served as data for
    every later run, so retraining alone would not have fixed it."""
    from veridis.chain import http as vh

    class Resp:
        status_code = 200
        headers: dict[str, str] = {}

        def __init__(self, payload):
            self._p = payload

        def json(self):
            return self._p

    served = [REFUSAL, REFUSAL, {"status": "1", "result": [{"ok": 1}]}]
    seen = []

    class FakeHttp:
        async def get(self, url, params=None):
            seen.append(url)
            return Resp(served[min(len(seen) - 1, len(served) - 1)])

        async def aclose(self):
            pass

    monkeypatch.setattr(vh, "CACHE", tmp_path)
    c = vh.CachedClient(namespace="t", rate_per_sec=1000, is_refusal=es.is_refusal,
                        backoff_start=0.001)
    c._client = FakeHttp()
    got = asyncio.run(c.get_json("https://example.test/x"))
    assert got == {"status": "1", "result": [{"ok": 1}]}
    assert c.stats["refused"] == 2
    # Only the real answer was written to disk.
    stored = list(tmp_path.rglob("*.json.gz"))
    assert len(stored) == 1

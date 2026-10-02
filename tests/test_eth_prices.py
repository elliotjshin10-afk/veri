"""A price is a time series, which is the shape of thing that leaks.

Stablecoins needed no price at all - USDT is a dollar. Native ETH needs one for
every dollar-denominated feature, and the obvious implementation (look up the
close of the day the transfer happened on) reaches hours into the future: that
close is set at the END of the day. A model trained on it would score well in
backtest and could not be reproduced live, which is the exact failure this
project guards against everywhere else.

These pin the rule: value a transfer at the last candle that had already CLOSED.
"""
from __future__ import annotations

import polars as pl
import pytest

from veridis.chain.prices import DAY, EthUsd

# Three days, deliberately far apart in value so a wrong pick is unmistakable.
D0 = 1_700_000_000 // DAY * DAY          # some day boundary
SERIES = pl.DataFrame({
    "day": [D0, D0 + DAY, D0 + 2 * DAY],
    "close": [100.0, 200.0, 400.0],
})


@pytest.fixture
def p():
    return EthUsd(SERIES)


def ms(sec: float) -> int:
    return int(sec * 1000)


def test_a_transfer_uses_the_last_candle_that_had_closed(p):
    # One second into day 1: day 0 has closed, day 1 has not.
    assert p.at(ms(D0 + DAY + 1)) == 100.0


def test_it_never_uses_the_close_of_the_day_it_is_in(p):
    # Late in day 1, day 1's close (200) is still hours away.
    assert p.at(ms(D0 + 2 * DAY - 1)) == 100.0


def test_the_moment_a_candle_closes_it_becomes_usable(p):
    assert p.at(ms(D0 + 2 * DAY)) == 200.0


def test_later_days_use_later_closes(p):
    assert p.at(ms(D0 + 3 * DAY)) == 400.0


def test_before_the_series_there_is_no_price(p):
    assert p.at(ms(D0)) is None
    assert p.at(ms(D0 - 10 * DAY)) is None


def test_no_price_is_ever_from_the_future(p):
    """The general statement of the rule, checked across the whole series."""
    days = SERIES["day"].to_list()
    closes = SERIES["close"].to_list()
    for t in range(D0 - DAY, D0 + 4 * DAY, 3600):
        got = p.at(ms(t))
        if got is None:
            continue
        # the candle we used must have closed at or before t
        i = closes.index(got)
        assert days[i] + DAY <= t, f"price at {t} came from a candle closing later"


def test_a_real_series_is_ordered_and_positive():
    """Guards the stored file rather than the lookup: an unsorted or zero-priced
    series would make bisect silently wrong."""
    from veridis.chain.prices import STORE
    if not STORE.exists():
        pytest.skip("price series not fetched in this environment")
    df = pl.read_parquet(STORE)
    days = df["day"].to_list()
    assert days == sorted(days)
    assert len(set(days)) == len(days)
    assert df["close"].min() > 0

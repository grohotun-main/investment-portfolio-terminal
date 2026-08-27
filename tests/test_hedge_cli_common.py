"""Tests for parsers/_hedge_cli_common.py.

Focus: the Polygon snapshot fetch must survive a *transient* read-timeout
instead of dying on the first slow response. The Option IV refresh
(`fetch_option_position_iv.py`) calls `fetch_expiry_chain` here; a bare
`requests.get(..., timeout=30)` with no retry let one slow snapshot response
abort the whole step (and leave option_position_snapshot.csv stale, which is
what ATM IV reads for sleeve discovery). The fix mirrors the proven
`fetch_atm_iv_history._get_json` retry layer.

The retry helper takes injectable `get`/`sleep` so the backoff logic is
unit-testable without real HTTP or wall-clock delay.
"""
import sys
import unittest
from datetime import date
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from _hedge_cli_common import _get_json, fetch_expiry_chain  # noqa: E402


class _Resp:
    """Minimal stand-in for a requests.Response."""

    def __init__(self, status, payload=None, headers=None):
        self.status_code = status
        self._payload = payload if payload is not None else {}
        self.headers = headers or {}

    def json(self):
        return self._payload


class _ScriptedGet:
    """Callable returning queued responses in order; a queued Exception is
    raised instead. Records how many times it was called."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def __call__(self, url, params=None, timeout=None):
        item = self.script[self.calls]
        self.calls += 1
        if isinstance(item, Exception):
            raise item
        return item


class _SleepRec:
    """Records requested sleep delays without actually sleeping."""

    def __init__(self):
        self.delays = []

    def __call__(self, d):
        self.delays.append(d)


class TestGetJsonRetry(unittest.TestCase):
    """_get_json: retry transient failures (timeout / 429 / 5xx) with backoff,
    fail loud on persistent or non-transient ones."""

    def test_success_returns_json_without_sleeping(self):
        get = _ScriptedGet([_Resp(200, {"results": [{"x": 1}]})])
        sleep = _SleepRec()
        out = _get_json("u", {}, get=get, sleep=sleep)
        self.assertEqual(out["results"][0]["x"], 1)
        self.assertEqual(sleep.delays, [])
        self.assertEqual(get.calls, 1)

    def test_retries_on_read_timeout_then_succeeds(self):
        # The reported failure: api.polygon.io read timed out. With retry the
        # next attempt succeeds instead of killing the whole Option IV step.
        get = _ScriptedGet([
            requests.exceptions.ReadTimeout("read timed out"),
            _Resp(200, {"ok": 1}),
        ])
        sleep = _SleepRec()
        out = _get_json("u", {}, get=get, sleep=sleep, base_delay=0.5)
        self.assertEqual(out["ok"], 1)
        self.assertEqual(get.calls, 2)
        self.assertEqual(len(sleep.delays), 1)

    def test_retries_on_429_then_succeeds(self):
        get = _ScriptedGet([_Resp(429), _Resp(429), _Resp(200, {"ok": 1})])
        sleep = _SleepRec()
        out = _get_json("u", {}, get=get, sleep=sleep, base_delay=0.5)
        self.assertEqual(out["ok"], 1)
        self.assertEqual(get.calls, 3)
        self.assertEqual(len(sleep.delays), 2)

    def test_honors_retry_after_header(self):
        get = _ScriptedGet([_Resp(429, headers={"Retry-After": "7"}),
                            _Resp(200, {})])
        sleep = _SleepRec()
        _get_json("u", {}, get=get, sleep=sleep)
        self.assertEqual(sleep.delays[0], 7.0)

    def test_raises_after_exhausting_timeout_retries(self):
        get = _ScriptedGet([requests.exceptions.ReadTimeout("boom")] * 10)
        sleep = _SleepRec()
        with self.assertRaises(requests.exceptions.ReadTimeout):
            _get_json("u", {}, get=get, sleep=sleep, max_retries=3)
        self.assertEqual(get.calls, 4)  # 1 initial + 3 retries

    def test_raises_immediately_on_non_retryable_404(self):
        get = _ScriptedGet([_Resp(404)])
        sleep = _SleepRec()
        with self.assertRaises(requests.exceptions.HTTPError):
            _get_json("u", {}, get=get, sleep=sleep, max_retries=3)
        self.assertEqual(get.calls, 1)  # no retry — retrying can't fix a 404
        self.assertEqual(sleep.delays, [])


class TestFetchExpiryChainResilience(unittest.TestCase):
    """fetch_expiry_chain routes through the retry layer, so a single transient
    timeout no longer aborts the snapshot pull the Option IV refresh depends on."""

    def _contract_payload(self):
        return {"results": [{
            "details": {"ticker": "O:SPY260717P00500000",
                        "contract_type": "put", "strike_price": 500.0,
                        "expiration_date": "2026-07-17"},
            "greeks": {"delta": -0.5, "vega": 0.1},
            "day": {"close": 12.3},
            "implied_volatility": 0.2,
            "open_interest": 100,
            "last_quote": {"bid": 12.0, "ask": 12.6},
            "underlying_asset": {"price": 501.2},
        }]}

    def test_survives_transient_timeout(self):
        get = _ScriptedGet([
            requests.exceptions.ReadTimeout("read timed out"),
            _Resp(200, self._contract_payload()),
        ])
        sleep = _SleepRec()
        chain, spot = fetch_expiry_chain(
            "SPY", date(2026, 7, 17), "KEY", "https://base",
            get=get, sleep=sleep,
        )
        self.assertEqual(len(chain), 1)
        self.assertEqual(float(chain.iloc[0]["strike"]), 500.0)
        self.assertEqual(chain.iloc[0]["contract_type"], "put")
        self.assertEqual(spot, 501.2)
        self.assertEqual(get.calls, 2)  # timed-out attempt + successful retry


if __name__ == "__main__":
    unittest.main()

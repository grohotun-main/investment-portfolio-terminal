"""Tests for parsers/fetch_risk_free_rate.py — the FRED DGS3MO fetcher.

Focus: the retry budget must be bounded and fail FAST. A blocked/slow FRED
endpoint previously hung the "Refresh all data" button for ~5 minutes
(MAX_ATTEMPTS=5 x a 60s read timeout) before failing, which both dominated
the refresh time and sank the whole market-data step to a failure.
"""
import inspect
import os
import sys
import unittest
from pathlib import Path
from unittest import mock

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

import fetch_risk_free_rate as frr  # noqa: E402
import _config as cfg  # noqa: E402


class TestFredFailFast(unittest.TestCase):
    def test_retry_budget_is_small(self) -> None:
        # Must stay small so a blocked FRED can't hang the refresh for minutes.
        self.assertLessEqual(frr.MAX_ATTEMPTS, 3)

    def test_default_read_timeout_is_short(self) -> None:
        # The per-attempt read timeout was 60s; cap it so the worst case is
        # tens of seconds, not minutes.
        default_timeout = inspect.signature(frr.fetch_dgs3mo).parameters["timeout"].default
        self.assertLessEqual(default_timeout, 30)

    def test_worst_case_wait_is_bounded(self) -> None:
        # Belt-and-suspenders: attempts x timeout (the dominant term) must be
        # comfortably under two minutes even in the all-timeout worst case.
        default_timeout = inspect.signature(frr.fetch_dgs3mo).parameters["timeout"].default
        self.assertLess(frr.MAX_ATTEMPTS * default_timeout, 120)

    def test_gives_up_after_exactly_max_attempts(self) -> None:
        # On persistent timeouts it retries exactly MAX_ATTEMPTS times then
        # raises — no unbounded loop. time.sleep is stubbed so the test is fast.
        calls = {"n": 0}

        def fake_get(*_a, **_k):
            calls["n"] += 1
            raise requests.Timeout("read timed out")

        with mock.patch.object(frr.requests, "get", side_effect=fake_get), \
                mock.patch.object(frr.time, "sleep", lambda *_a, **_k: None):
            with self.assertRaises(requests.RequestException):
                frr.fetch_dgs3mo()
        self.assertEqual(calls["n"], frr.MAX_ATTEMPTS)


# --- FRED API fallback ---------------------------------------------------
# On some networks (Norton TLS interception) the keyless fredgraph host
# (fred.stlouisfed.org) read-times-out, while FRED's API host
# (api.stlouisfed.org) is reachable. When a FRED_API_KEY is configured the
# fetcher uses the API FIRST (fast, reachable), then falls back to the graph
# CSV; with no key it uses only the graph CSV (prior behavior).

_API_JSON = (
    '{"observations":['
    '{"date":"2026-06-03","value":"4.19"},'
    '{"date":"2026-06-04","value":"."},'   # FRED missing-obs marker -> dropped
    '{"date":"2026-06-05","value":"4.21"}'
    ']}'
)
_GRAPH_CSV = "observation_date,DGS3MO\n2026-06-03,4.10\n2026-06-04,.\n2026-06-05,4.12\n"


class _Resp:
    """Minimal stand-in for a requests.Response (text + no-op raise_for_status)."""
    def __init__(self, text):
        self.text = text

    def raise_for_status(self):
        pass


class TestFredApiFallback(unittest.TestCase):
    def test_parse_api_json_drops_missing_and_scales(self):
        df = frr._parse_api_json(_API_JSON)
        self.assertEqual(list(df["date"].dt.strftime("%Y-%m-%d")),
                         ["2026-06-03", "2026-06-05"])  # the '.' row is dropped
        self.assertAlmostEqual(df["rate_annual"].iloc[-1], 0.0421)  # 4.21% -> decimal

    def test_parse_graph_csv_drops_missing_and_scales(self):
        df = frr._parse_graph_csv(_GRAPH_CSV)
        self.assertEqual(list(df["date"].dt.strftime("%Y-%m-%d")),
                         ["2026-06-03", "2026-06-05"])
        self.assertAlmostEqual(df["rate_annual"].iloc[-1], 0.0412)

    def test_api_used_first_when_key_present(self):
        seen = []

        def fake_get(url, params=None, timeout=None, headers=None):
            seen.append(url)
            return _Resp(_API_JSON)

        with mock.patch.object(frr.requests, "get", side_effect=fake_get):
            df = frr.fetch_dgs3mo(api_key="KEY")
        self.assertIn("api.stlouisfed.org", seen[0])      # API endpoint tried first
        self.assertAlmostEqual(df["rate_annual"].iloc[-1], 0.0421)

    def test_falls_back_to_graph_when_api_unreachable(self):
        def fake_get(url, params=None, timeout=None, headers=None):
            if "api.stlouisfed.org" in url:
                raise requests.Timeout("blocked")
            return _Resp(_GRAPH_CSV)

        with mock.patch.object(frr.requests, "get", side_effect=fake_get), \
                mock.patch.object(frr.time, "sleep", lambda *_a, **_k: None):
            df = frr.fetch_dgs3mo(api_key="KEY")
        self.assertAlmostEqual(df["rate_annual"].iloc[-1], 0.0412)  # graph CSV value

    def test_no_key_uses_graph_only(self):
        seen = []

        def fake_get(url, params=None, timeout=None, headers=None):
            seen.append(url)
            return _Resp(_GRAPH_CSV)

        with mock.patch.object(frr.requests, "get", side_effect=fake_get):
            frr.fetch_dgs3mo()  # no key
        self.assertTrue(seen and all("api.stlouisfed.org" not in u for u in seen))
        self.assertIn("fred.stlouisfed.org", seen[0])


class TestFredApiKeyResolution(unittest.TestCase):
    def test_empty_when_unset(self):
        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(cfg, "load_env", lambda *_a, **_k: {}):
            self.assertEqual(cfg.get_fred_api_key(), "")

    def test_read_from_environment(self):
        with mock.patch.dict(os.environ, {"FRED_API_KEY": "abc123"}, clear=True):
            self.assertEqual(cfg.get_fred_api_key(), "abc123")


if __name__ == "__main__":
    unittest.main()

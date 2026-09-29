"""Tests for parsers/fetch_atm_iv_history.py.

Three pure surfaces:
  * `front_month_expiry` — calendar 3rd-Friday picker. Doesn't hit Polygon.
  * `atm_strike` — round spot to nearest $5 multiple (reliable strike grid
    on both SPY and NVDA).
  * `invert_atm_iv` — wrap options_pricer.implied_vol with the put-side
    convention this parser uses, returning NaN on bad inputs rather than
    raising.

The Polygon fetch loop is not unit-tested — it's exercised in the
end-to-end sample-fetch dry run before merging the PR.
"""
import math
import sys
import unittest
from datetime import date
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from fetch_atm_iv_history import (  # noqa: E402
    FetchError,
    RegressionError,
    WorkItem,
    _fetch_work_items_parallel,
    _get_json,
    assert_not_regressive,
    atm_strike,
    fetch_option_close,
    fetch_unadjusted_spot_range,
    front_month_expiry,
    invert_atm_iv,
    merge_history,
    missing_days,
    next_n_monthly_expiries,
    plan_work_items,
)
from iv_constant_maturity import CM_CSV_COLS  # noqa: E402


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


class _UrlGet:
    """Thread-safe fake `get`, keyed by a substring match against the URL so
    it's order-independent (safe under concurrency). `routes` maps a URL
    substring → either a _Resp or an Exception (raised). Unmatched URLs get
    `default`. Counts calls under a lock."""

    def __init__(self, routes, default=None):
        import threading
        self.routes = routes
        self.default = default or _Resp(200, {"results": []})
        self.calls = 0
        self._lock = threading.Lock()

    def __call__(self, url, params=None, timeout=None):
        with self._lock:
            self.calls += 1
        for needle, item in self.routes.items():
            if needle in url:
                if isinstance(item, Exception):
                    raise item
                return item
        return self.default


class TestGetJsonRetry(unittest.TestCase):
    """_get_json: retry transient failures with backoff, raise FetchError on
    persistent / non-transient ones. The fix for the silent partial backfill —
    HTTP failures must surface, not masquerade as no-data."""

    def test_success_returns_json_without_sleeping(self):
        get = _ScriptedGet([_Resp(200, {"results": [{"c": 1.0}]})])
        sleep = _SleepRec()
        out = _get_json("u", {}, get=get, sleep=sleep)
        self.assertEqual(out["results"][0]["c"], 1.0)
        self.assertEqual(sleep.delays, [])
        self.assertEqual(get.calls, 1)

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

    def test_retries_on_503(self):
        get = _ScriptedGet([_Resp(503), _Resp(200, {"ok": 1})])
        out = _get_json("u", {}, get=get, sleep=_SleepRec())
        self.assertEqual(out["ok"], 1)

    def test_raises_fetcherror_on_persistent_429(self):
        get = _ScriptedGet([_Resp(429)] * 10)
        with self.assertRaises(FetchError):
            _get_json("u", {}, get=get, sleep=_SleepRec(), max_retries=3)
        self.assertEqual(get.calls, 4)  # 1 initial + 3 retries

    def test_raises_immediately_on_non_transient_403(self):
        # 403 = entitlement/tier problem. Must fail LOUD and not retry —
        # this is exactly the kind of wall that silently dropped SPY.
        get = _ScriptedGet([_Resp(403)])
        sleep = _SleepRec()
        with self.assertRaises(FetchError):
            _get_json("u", {}, get=get, sleep=sleep, max_retries=3)
        self.assertEqual(get.calls, 1)
        self.assertEqual(sleep.delays, [])

    def test_retries_connection_error_then_succeeds(self):
        get = _ScriptedGet([requests.ConnectionError("boom"),
                            _Resp(200, {"ok": 1})])
        out = _get_json("u", {}, get=get, sleep=_SleepRec())
        self.assertEqual(out["ok"], 1)
        self.assertEqual(get.calls, 2)

    def test_raises_after_exhausting_connection_retries(self):
        get = _ScriptedGet([requests.ConnectionError("boom")] * 10)
        with self.assertRaises(FetchError):
            _get_json("u", {}, get=get, sleep=_SleepRec(), max_retries=2)
        self.assertEqual(get.calls, 3)


class TestFetchOptionCloseErrorHandling(unittest.TestCase):
    """fetch_option_close must distinguish genuine no-data (None) from an
    HTTP failure (FetchError) — conflating them was the root-cause bug."""

    def test_returns_none_on_genuine_no_data(self):
        get = _ScriptedGet([_Resp(200, {"results": []})])
        out = fetch_option_close("O:SPY260717P00590000", date(2020, 1, 2),
                                 "k", "b", get=get, sleep=_SleepRec())
        self.assertIsNone(out)

    def test_returns_close_on_data(self):
        get = _ScriptedGet([_Resp(200, {"results": [{"c": 3.5}]})])
        out = fetch_option_close("O:SPY260717P00590000", date(2020, 1, 2),
                                 "k", "b", get=get, sleep=_SleepRec())
        self.assertAlmostEqual(out, 3.5)

    def test_raises_on_http_error_instead_of_returning_none(self):
        get = _ScriptedGet([_Resp(403)])
        with self.assertRaises(FetchError):
            fetch_option_close("O:SPY260717P00590000", date(2020, 1, 2),
                               "k", "b", get=get, sleep=_SleepRec())


class TestFetchUnadjustedSpotErrorHandling(unittest.TestCase):
    """fetch_unadjusted_spot_range must raise on HTTP failure rather than
    return an empty frame — an empty frame silently skipped the whole
    underlying (this is why SPY vanished from the backfill)."""

    def test_raises_on_http_error_instead_of_empty_frame(self):
        get = _ScriptedGet([_Resp(403)])
        with self.assertRaises(FetchError):
            fetch_unadjusted_spot_range("SPY", date(2016, 1, 1),
                                        date(2016, 2, 1), "k", "b",
                                        get=get, sleep=_SleepRec())

    def test_returns_frame_on_success(self):
        payload = {"results": [{"t": 1451779200000, "c": 200.0}]}
        get = _ScriptedGet([_Resp(200, payload)])
        df = fetch_unadjusted_spot_range("SPY", date(2016, 1, 1),
                                         date(2016, 2, 1), "k", "b",
                                         get=get, sleep=_SleepRec())
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(float(df.iloc[0]["close"]), 200.0)

    def test_empty_results_is_not_an_error(self):
        # A 200 with no bars (e.g. a gap with no trading) is genuine
        # no-data → empty frame, not a raise.
        get = _ScriptedGet([_Resp(200, {"results": []})])
        df = fetch_unadjusted_spot_range("SPY", date(2016, 1, 1),
                                         date(2016, 2, 1), "k", "b",
                                         get=get, sleep=_SleepRec())
        self.assertTrue(df.empty)


class TestFrontMonthExpiry(unittest.TestCase):
    """Front-month = first 3rd-Friday strictly after the as-of date."""

    def test_first_of_month_picks_this_months_third_friday(self):
        # Jan 2026: 3rd Friday is Jan 16.
        self.assertEqual(front_month_expiry(date(2026, 1, 1)), date(2026, 1, 16))

    def test_day_before_expiry_picks_this_months_third_friday(self):
        self.assertEqual(front_month_expiry(date(2026, 1, 15)), date(2026, 1, 16))

    def test_on_expiry_day_rolls_to_next_month(self):
        # Strictly-after semantics: on Jan 16 (Friday) we look ahead to Feb's
        # 3rd Friday (Feb 20), avoiding the DTE≈0 IV noise spike.
        self.assertEqual(front_month_expiry(date(2026, 1, 16)), date(2026, 2, 20))

    def test_late_month_rolls_to_next_month(self):
        self.assertEqual(front_month_expiry(date(2026, 1, 30)), date(2026, 2, 20))

    def test_year_boundary(self):
        # Dec 2026: 3rd Friday is Dec 18. Dec 19 onward rolls to Jan 2027's
        # 3rd Friday = Jan 15.
        self.assertEqual(front_month_expiry(date(2026, 12, 19)), date(2027, 1, 15))

    def test_juneteenth_friday_rolls_to_thursday(self):
        # 2026-06-19 is the 3rd Friday AND Juneteenth (federal market
        # holiday since 2021). The monthly expiry shifts to Thursday
        # 2026-06-18 — Polygon confirms this is the listed expiry.
        self.assertEqual(front_month_expiry(date(2026, 5, 26)), date(2026, 6, 18))

    def test_good_friday_rolls_to_thursday(self):
        # 2025-04-18 was the 3rd Friday AND Good Friday (NYSE closed).
        # Listed expiry was Thursday 2025-04-17.
        self.assertEqual(front_month_expiry(date(2025, 4, 1)), date(2025, 4, 17))

    def test_holiday_shifted_expiry_day_rolls_forward(self):
        # On the rolled Thursday itself, strict-after semantics already
        # roll us to next month. Jun 18 2026 -> Jul 17 2026.
        self.assertEqual(front_month_expiry(date(2026, 6, 18)), date(2026, 7, 17))


class TestNextNMonthlyExpiries(unittest.TestCase):
    """next_n_monthly_expiries(as_of, n) = the next n listed monthly expiries
    strictly after as_of, ascending, reusing the holiday-aware picker."""

    def test_returns_exactly_n(self):
        self.assertEqual(len(next_n_monthly_expiries(date(2026, 1, 1), 4)), 4)
        self.assertEqual(len(next_n_monthly_expiries(date(2026, 1, 1), 1)), 1)

    def test_first_equals_front_month_expiry(self):
        # The n-expiry walk must start exactly where the single-expiry picker
        # does — same strictly-after + holiday semantics.
        for as_of in (date(2026, 1, 1), date(2026, 5, 26), date(2026, 12, 19)):
            got = next_n_monthly_expiries(as_of, 4)
            self.assertEqual(got[0], front_month_expiry(as_of))

    def test_known_sequence_from_jan(self):
        self.assertEqual(
            next_n_monthly_expiries(date(2026, 1, 1), 4),
            [date(2026, 1, 16), date(2026, 2, 20),
             date(2026, 3, 20), date(2026, 4, 17)],
        )

    def test_strictly_increasing_and_after_as_of(self):
        got = next_n_monthly_expiries(date(2026, 3, 10), 5)
        self.assertTrue(all(e > date(2026, 3, 10) for e in got))
        self.assertEqual(got, sorted(got))
        self.assertEqual(len(set(got)), len(got))  # no dupes

    def test_preserves_juneteenth_shift_midsequence(self):
        # Window straddling June 2026: the Jun expiry must be Thursday 6/18
        # (Juneteenth shift), not the nominal 3rd Friday 6/19.
        got = next_n_monthly_expiries(date(2026, 5, 26), 4)
        self.assertEqual(
            got,
            [date(2026, 6, 18), date(2026, 7, 17),
             date(2026, 8, 21), date(2026, 9, 18)],
        )

    def test_crosses_year_boundary(self):
        # Dec 18 2026 is the listed Dec expiry; on 12/19 we roll into 2027.
        self.assertEqual(
            next_n_monthly_expiries(date(2026, 12, 19), 3),
            [date(2027, 1, 15), date(2027, 2, 19), date(2027, 3, 19)],
        )

    def test_n5_always_brackets_90_days(self):
        # The practical guarantee the CM-90 derive step relies on: across a
        # full year of as-of dates, 5 monthlies always straddle the 90-day
        # target (front month < 90 < furthest). n=4 is NOT enough — e.g.
        # just before a February expiry the 4th monthly lands at only ~85
        # DTE (short Feb), so the derive step grabs 5.
        for month in range(1, 13):
            as_of = date(2026, month, 14)
            dtes = [(e - as_of).days for e in
                    next_n_monthly_expiries(as_of, 5)]
            self.assertLess(min(dtes), 90, f"front not < 90 for {as_of}")
            self.assertGreaterEqual(max(dtes), 90,
                                    f"5 monthlies don't reach 90 for {as_of}")

    def test_n4_can_fall_short_of_90_near_february(self):
        # Documents WHY the derive step uses 5, not 4: the day before the
        # Feb expiry, 4 monthlies only reach mid-May (~85 DTE).
        dtes = [(e - date(2026, 2, 19)).days
                for e in next_n_monthly_expiries(date(2026, 2, 19), 4)]
        self.assertLess(max(dtes), 90)


class TestAtmStrike(unittest.TestCase):
    """Round spot to nearest $5 — listed strike on both SPY and NVDA in the
    near-money range."""

    def test_below_midpoint_rounds_down(self):
        self.assertEqual(atm_strike(742.0), 740.0)

    def test_above_midpoint_rounds_up(self):
        self.assertEqual(atm_strike(748.0), 750.0)

    def test_at_multiple_stays(self):
        self.assertEqual(atm_strike(745.0), 745.0)

    def test_midpoint_rounds_up(self):
        # 742.5 sits exactly between 740 and 745 — convention: round up
        # (so deep-OTM puts get the more-conservative strike).
        self.assertEqual(atm_strike(742.5), 745.0)

    def test_nvda_at_high_spot(self):
        self.assertEqual(atm_strike(214.30), 215.0)


class TestMergeHistory(unittest.TestCase):
    """Idempotent merge protects the deep backfill from casual re-runs.

    Term schema: several expiry rows per (date, underlying), so the dedup
    key is (date, underlying, expiry) — distinct expiries on the same day
    must all survive.
    """

    def _row(self, d, u, iv, expiry="2026-03-20"):
        return {"date": d, "underlying": u, "expiry": expiry,
                "dte_days": 77, "atm_strike": 500.0, "spot": 500.0,
                "close": 5.0, "atm_iv": iv, "fetched_at": "2026-01-01T00:00:00"}

    def test_merge_unions_new_dates(self):
        existing = pd.DataFrame([self._row("2026-01-01", "SPY", 0.20)])
        new = pd.DataFrame([self._row("2026-01-02", "SPY", 0.22)])
        out = merge_history(existing, new)
        self.assertEqual(len(out), 2)
        self.assertEqual(set(out["date"]), {"2026-01-01", "2026-01-02"})

    def test_merge_dedupes_on_date_underlying_expiry_keeping_new(self):
        existing = pd.DataFrame([self._row("2026-01-01", "SPY", 0.20)])
        new = pd.DataFrame([self._row("2026-01-01", "SPY", 0.99)])  # same key
        out = merge_history(existing, new)
        self.assertEqual(len(out), 1)
        # Fresh fetch wins
        self.assertAlmostEqual(float(out.iloc[0]["atm_iv"]), 0.99)

    def test_merge_keeps_distinct_expiries_same_day_underlying(self):
        # The term-structure case: 5 monthlies share one (date, underlying)
        # and must NOT collapse into one row.
        existing = pd.DataFrame([
            self._row("2026-01-01", "SPY", 0.22, expiry="2026-03-20"),
        ])
        new = pd.DataFrame([
            self._row("2026-01-01", "SPY", 0.24, expiry="2026-04-17"),
        ])
        out = merge_history(existing, new)
        self.assertEqual(len(out), 2)
        self.assertEqual(set(out["expiry"]),
                         {"2026-03-20", "2026-04-17"})

    def test_merge_keeps_distinct_underlyings_same_date(self):
        existing = pd.DataFrame([self._row("2026-01-01", "SPY", 0.20)])
        new = pd.DataFrame([self._row("2026-01-01", "NVDA", 0.55)])
        out = merge_history(existing, new)
        self.assertEqual(len(out), 2)

    def test_merge_empty_existing_returns_new(self):
        new = pd.DataFrame([self._row("2026-01-01", "SPY", 0.20)])
        out = merge_history(pd.DataFrame(), new)
        self.assertEqual(len(out), 1)


class TestMissingDays(unittest.TestCase):
    def test_missing_days_excludes_already_present(self):
        existing = pd.DataFrame([
            {"date": "2026-01-01", "underlying": "SPY"},
            {"date": "2026-01-02", "underlying": "SPY"},
        ])
        candidates = [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3)]
        missing = missing_days(existing, candidates, "SPY")
        self.assertEqual(missing, [date(2026, 1, 3)])

    def test_missing_days_per_underlying(self):
        # SPY has 1/1; NVDA has nothing → NVDA still needs 1/1.
        existing = pd.DataFrame([{"date": "2026-01-01", "underlying": "SPY"}])
        candidates = [date(2026, 1, 1)]
        self.assertEqual(missing_days(existing, candidates, "NVDA"),
                         [date(2026, 1, 1)])

    def test_missing_days_empty_existing_returns_all(self):
        candidates = [date(2026, 1, 1), date(2026, 1, 2)]
        self.assertEqual(
            missing_days(pd.DataFrame(), candidates, "SPY"), candidates,
        )


class TestAssertNotRegressive(unittest.TestCase):
    """Write-guard: refuse to overwrite a complete on-disk dataset with one
    that loses an underlying or shrinks a per-underlying row count. The last
    line of defense against the silent partial-backfill overwrite."""

    def _cm(self, counts: dict[str, int]) -> pd.DataFrame:
        rows = []
        for u, n in counts.items():
            for i in range(n):
                rows.append({"date": f"2020-01-{i+1:02d}", "underlying": u,
                             "atm_iv": 0.2, "quality": "interp",
                             "target_days": 90, "fetched_at": "x"})
        return pd.DataFrame(rows, columns=CM_CSV_COLS)

    def test_passes_when_new_grows(self):
        existing = self._cm({"SPY": 100, "NVDA": 100})
        new = self._cm({"SPY": 120, "NVDA": 110})
        assert_not_regressive(existing, new)  # no raise

    def test_passes_when_equal(self):
        existing = self._cm({"SPY": 100})
        assert_not_regressive(existing, self._cm({"SPY": 100}))

    def test_raises_when_underlying_disappears(self):
        # The SPY-vanished scenario.
        existing = self._cm({"SPY": 2510, "NVDA": 2506})
        new = self._cm({"NVDA": 2263})
        with self.assertRaises(RegressionError):
            assert_not_regressive(existing, new)

    def test_raises_when_underlying_shrinks(self):
        # The NVDA-lost-a-year scenario.
        existing = self._cm({"SPY": 2510, "NVDA": 2506})
        new = self._cm({"SPY": 2510, "NVDA": 2263})
        with self.assertRaises(RegressionError):
            assert_not_regressive(existing, new)

    def test_no_existing_file_is_fine(self):
        assert_not_regressive(pd.DataFrame(columns=CM_CSV_COLS),
                              self._cm({"SPY": 10}))

    def test_guard_only_counts_rows_not_schema(self):
        # The guard is schema-agnostic on its own — it just compares row
        # counts per underlying. (main() is responsible for only INVOKING it
        # when the prior file is a CM file; see the `quality` check there.)
        # Here: a legacy front-month frame with MORE rows must still trip if
        # the new CM frame has fewer — proving the guard itself is honest.
        legacy = pd.DataFrame({
            "date": ["2020-01-01"] * 2510,
            "underlying": ["SPY"] * 2510, "atm_iv": [0.2] * 2510,
        })
        with self.assertRaises(RegressionError):
            assert_not_regressive(legacy, self._cm({"SPY": 2263}))


class TestInvertAtmIv(unittest.TestCase):
    """Thin wrapper around options_pricer.implied_vol with NaN-on-bad-input
    instead of raises, and the put-side convention this parser locks in."""

    def test_roundtrip_recovers_known_iv(self):
        # Price a put at known σ, then invert and confirm we get σ back.
        # SPY $745 ATM put, 30 days out, σ=20%, r=4%, q=1.5%.
        from options_pricer import price_and_greeks  # noqa: E402

        spot, strike, T, r, q, sigma = 745.0, 745.0, 30 / 365.0, 0.04, 0.015, 0.20
        priced = price_and_greeks(spot, strike, T, r, q, sigma, "put")
        iv = invert_atm_iv(
            close=priced["price"], spot=spot, strike=strike,
            dte_days=30, r=r, q=q,
        )
        self.assertTrue(math.isfinite(iv))
        self.assertAlmostEqual(iv, 0.20, places=3)

    def test_zero_dte_returns_nan(self):
        self.assertTrue(math.isnan(invert_atm_iv(
            close=1.0, spot=745.0, strike=745.0, dte_days=0,
            r=0.04, q=0.015,
        )))

    def test_zero_close_returns_nan(self):
        self.assertTrue(math.isnan(invert_atm_iv(
            close=0.0, spot=745.0, strike=745.0, dte_days=30,
            r=0.04, q=0.015,
        )))


def _spot_frame(underlying, dates_closes):
    """Build a [date, symbol, close] spot frame like fetch_unadjusted_
    spot_range returns. dates_closes: list of (iso_date, close)."""
    return pd.DataFrame(
        [{"date": pd.Timestamp(d), "symbol": underlying, "close": c}
         for d, c in dates_closes],
        columns=["date", "symbol", "close"],
    )


class TestPlanWorkItems(unittest.TestCase):
    """plan_work_items: pure (no I/O) work-list builder. Turns already-
    fetched spot frames + the incremental cache into one WorkItem per
    (underlying, day, expiry) that still needs fetching."""

    def _rfr(self):
        return pd.Series(dtype=float)  # empty → default r used

    def test_one_item_per_expiry_per_uncached_day(self):
        spot = {"SPY": _spot_frame("SPY", [("2026-01-02", 500.0)])}
        items = plan_work_items(spot, existing=pd.DataFrame(),
                                rfr=self._rfr(), n_expiries=5, q=0.015)
        self.assertEqual(len(items), 5)  # 5 expiries for the one day
        self.assertTrue(all(isinstance(it, WorkItem) for it in items))
        self.assertEqual({it.underlying for it in items}, {"SPY"})
        self.assertEqual({it.day for it in items}, {date(2026, 1, 2)})
        # 5 distinct expiries, all strictly after the day
        self.assertEqual(len({it.expiry for it in items}), 5)
        self.assertTrue(all(it.expiry > date(2026, 1, 2) for it in items))

    def test_strike_is_atm_rounded(self):
        spot = {"SPY": _spot_frame("SPY", [("2026-01-02", 742.0)])}
        items = plan_work_items(spot, existing=pd.DataFrame(),
                                rfr=self._rfr(), n_expiries=1, q=0.015)
        self.assertEqual(items[0].strike, 740.0)  # 742 -> nearest $5

    def test_skips_cached_days(self):
        # 1/2 already on disk for SPY → only 1/5 should be planned.
        existing = pd.DataFrame([
            {"date": "2026-01-02", "underlying": "SPY", "expiry": "2026-02-20"},
        ])
        spot = {"SPY": _spot_frame("SPY",
                                   [("2026-01-02", 500.0), ("2026-01-05", 505.0)])}
        items = plan_work_items(spot, existing=existing,
                                rfr=self._rfr(), n_expiries=5, q=0.015)
        self.assertEqual({it.day for it in items}, {date(2026, 1, 5)})

    def test_multiple_underlyings(self):
        spot = {
            "SPY": _spot_frame("SPY", [("2026-01-02", 500.0)]),
            "NVDA": _spot_frame("NVDA", [("2026-01-02", 130.0)]),
        }
        items = plan_work_items(spot, existing=pd.DataFrame(),
                                rfr=self._rfr(), n_expiries=2, q=0.015)
        self.assertEqual(len(items), 4)  # 2 underlyings × 2 expiries

    def test_risk_free_rate_resolved_per_day(self):
        rfr = pd.Series({pd.Timestamp("2026-01-02"): 0.051})
        spot = {"SPY": _spot_frame("SPY", [("2026-01-02", 500.0)])}
        items = plan_work_items(spot, existing=pd.DataFrame(),
                                rfr=rfr, n_expiries=1, q=0.015)
        self.assertAlmostEqual(items[0].r, 0.051)

    def test_empty_spot_yields_no_items(self):
        items = plan_work_items({"SPY": _spot_frame("SPY", [])},
                                existing=pd.DataFrame(),
                                rfr=self._rfr(), n_expiries=5, q=0.015)
        self.assertEqual(items, [])


def _work_item(underlying, iso_day, iso_expiry, strike, spot, r=0.04, q=0.015):
    from datetime import date as _date
    y, m, d = (int(x) for x in iso_day.split("-"))
    ey, em, ed = (int(x) for x in iso_expiry.split("-"))
    return WorkItem(underlying=underlying, day=_date(y, m, d),
                    expiry=_date(ey, em, ed), strike=strike, spot=spot,
                    r=r, q=q)


class TestFetchWorkItemsParallel(unittest.TestCase):
    """_fetch_work_items_parallel: fan out fetch_option_close across threads,
    preserving the fail-loud contract (any HTTP failure aborts the whole
    run — the #117 bug must not reappear under concurrency)."""

    def _items(self, n):
        # n SPY items, distinct expiries so their option tickers differ.
        exps = ["2026-02-20", "2026-03-20", "2026-04-17", "2026-05-15",
                "2026-06-18", "2026-07-17"]
        return [_work_item("SPY", "2026-01-02", exps[i % len(exps)],
                           500.0 + i, 500.0) for i in range(n)]

    def test_all_success_returns_row_per_item(self):
        # Every option close resolves → one term row per work item.
        get = _UrlGet({}, default=_Resp(200, {"results": [{"c": 5.0}]}))
        rows = _fetch_work_items_parallel(
            self._items(6), "k", "b", max_workers=4, get=get,
            sleep=_SleepRec(),
        )
        self.assertEqual(len(rows), 6)
        self.assertEqual({r["underlying"] for r in rows}, {"SPY"})
        self.assertTrue(all("atm_iv" in r for r in rows))

    def test_none_close_drops_silently(self):
        # Genuine no-data (200, empty results) → that item drops, no raise.
        get = _UrlGet({}, default=_Resp(200, {"results": []}))
        rows = _fetch_work_items_parallel(
            self._items(6), "k", "b", max_workers=4, get=get,
            sleep=_SleepRec(),
        )
        self.assertEqual(rows, [])

    def test_one_http_failure_aborts_whole_run(self):
        # THE critical invariant: a 403 on ANY contract must propagate as
        # FetchError and abort — never silently produce a partial dataset.
        # Item i=3 has strike 503.0 → ticker ...P00503000. Route that one
        # contract to a 403; everything else 200.
        get = _UrlGet(
            {"P00503000": _Resp(403)},
            default=_Resp(200, {"results": [{"c": 5.0}]}),
        )
        with self.assertRaises(FetchError):
            _fetch_work_items_parallel(
                self._items(6), "k", "b", max_workers=4, get=get,
                sleep=_SleepRec(),
            )

    def test_results_independent_of_worker_count(self):
        get1 = _UrlGet({}, default=_Resp(200, {"results": [{"c": 5.0}]}))
        get8 = _UrlGet({}, default=_Resp(200, {"results": [{"c": 5.0}]}))
        r1 = _fetch_work_items_parallel(self._items(12), "k", "b",
                                        max_workers=1, get=get1,
                                        sleep=_SleepRec())
        r8 = _fetch_work_items_parallel(self._items(12), "k", "b",
                                        max_workers=8, get=get8,
                                        sleep=_SleepRec())
        self.assertEqual(len(r1), len(r8))

    def test_progress_callback_invoked(self):
        seen = []
        get = _UrlGet({}, default=_Resp(200, {"results": [{"c": 5.0}]}))
        _fetch_work_items_parallel(
            self._items(6), "k", "b", max_workers=4, get=get,
            sleep=_SleepRec(), progress=lambda done, total: seen.append((done, total)),
        )
        self.assertTrue(seen)
        self.assertEqual(seen[-1][0], 6)   # final done == item count
        self.assertEqual(seen[-1][1], 6)   # total == item count


if __name__ == "__main__":
    unittest.main()

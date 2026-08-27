"""Offline tests for parsers/fetch_single_name_iv_history.py (spec test 12c).
All network goes through an injected get_json_fn — no HTTP anywhere."""
import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from fetch_single_name_iv_history import (  # noqa: E402
    EntitlementError,
    build_iv_history,
    load_or_refresh_iv_history,
    occ_put_ticker,
    standard_expiry,
)
from options_pricer import black_scholes, implied_vol  # noqa: E402


def _contracts(expiry: str, strikes) -> list[dict]:
    return [{"expiration_date": expiry, "strike_price": k,
             "contract_type": "put"} for k in strikes]


def _bars(dates_closes) -> list[dict]:
    return [{"t": int(pd.Timestamp(d).value // 10**6), "c": c}
            for d, c in dates_closes]


class FakePolygon:
    """Canned Polygon: reference months + per-contract bars, with a call log."""

    def __init__(self, months: dict, bars: dict, forbid: bool = False,
                forbid_aggs: bool = False):
        self.months = months            # "YYYY-MM" -> results list
        self.bars = bars                # occ ticker -> results list
        self.forbid = forbid
        self.forbid_aggs = forbid_aggs
        self.calls: list[tuple[str, str]] = []

    def __call__(self, url: str, params: dict, timeout: int = 30) -> dict:
        if self.forbid:
            raise EntitlementError("403 NOT_AUTHORIZED")
        if "/v3/reference/options/contracts" in url:
            ym = str(params.get("expiration_date.gte", ""))[:7]
            self.calls.append(("ref", ym))
            return {"results": self.months.get(ym, [])}
        if "/v2/aggs/ticker/" in url:
            if self.forbid_aggs:
                raise EntitlementError("403 NOT_AUTHORIZED")
            occ = url.split("/v2/aggs/ticker/")[1].split("/range")[0]
            self.calls.append(("aggs", occ))
            return {"results": self.bars.get(occ, [])}
        raise AssertionError(f"unexpected url: {url}")


def _closes(days_closes) -> pd.Series:
    idx = pd.DatetimeIndex([pd.Timestamp(d) for d, _ in days_closes])
    return pd.Series([c for _, c in days_closes], index=idx)


def _r_series() -> pd.Series:
    idx = pd.bdate_range("2026-01-02", "2026-12-31")
    return pd.Series(0.04, index=idx)


def _scenario():
    """Closes over 2026-06-15..2026-06-19 (5 bdays), spot 100 except one deep
    -30% day. Standard July expiry 2026-07-17 (5 strikes) beats the weekly
    2026-07-10 (2 strikes). August expiry exists so DTE targeting must still
    prefer July (|dte-35| smaller)."""
    days = [("2026-06-15", 100.0), ("2026-06-16", 100.0),
            ("2026-06-17", 70.0),   # nearest listed strike 90 -> deep ITM
            ("2026-06-18", 100.0), ("2026-06-19", 100.0)]
    months = {
        "2026-07": (_contracts("2026-07-17", [90, 95, 100, 105, 110])
                    + _contracts("2026-07-10", [100, 105])),
        "2026-08": _contracts("2026-08-21", [90, 100, 110]),
    }
    E = date(2026, 7, 17)
    # priced at the TRUE calendar DTE of the asserted anchor date:
    # 2026-06-15 -> 2026-07-17 is 32 days, so row15's inversion recovers
    # sigma=0.32 exactly; the other dates reuse this price at 31/28 DTE and
    # their (unasserted) IVs simply land near 0.32.
    fair = black_scholes(100.0, 100.0, 32 / 365, 0.04, 0.0, 0.32, "put")["price"]
    k100 = occ_put_ticker("TST", E, 100.0)
    k95 = occ_put_ticker("TST", E, 95.0)
    k90 = occ_put_ticker("TST", E, 90.0)
    bars = {
        # K=100 bar MISSING on 06-18 -> second-nearest (95) must carry it
        k100: _bars([("2026-06-15", fair), ("2026-06-16", fair),
                     ("2026-06-19", fair)]),
        k95: _bars([("2026-06-18", fair * 0.6)]),
        # 06-17: spot 70, nearest strike 90 -> below-intrinsic print 10.0
        # (intrinsic-forward ~ 19.9) -> implied_vol NaN -> row dropped
        k90: _bars([("2026-06-17", 10.0)]),
    }
    return _closes(days), months, bars


class TestBuildIvHistory(unittest.TestCase):
    def test_selection_inversion_and_drops(self):
        closes, months, bars = _scenario()
        fake = FakePolygon(months, bars)
        df, status = build_iv_history("TST", closes, _r_series(), 0.0,
                                      key="k", base="http://x",
                                      get_json_fn=fake, log=lambda *_: None)
        self.assertEqual(status, "ok")
        got_dates = set(df["date"].astype(str))
        self.assertIn("2026-06-15", got_dates)
        self.assertIn("2026-06-18", got_dates)          # via 2nd-nearest 95
        self.assertNotIn("2026-06-17", got_dates)       # below-intrinsic drop
        self.assertTrue((df["expiry"].astype(str) == "2026-07-17").all())
        row = df[df["date"].astype(str) == "2026-06-18"].iloc[0]
        self.assertEqual(row["strike"], 95.0)
        row15 = df[df["date"].astype(str) == "2026-06-15"].iloc[0]
        self.assertAlmostEqual(row15["iv"], 0.32, places=2)

    def test_standard_expiry_pick(self):
        e, strikes = standard_expiry({
            date(2026, 7, 17): [90, 95, 100, 105, 110],
            date(2026, 7, 10): [100, 105]})
        self.assertEqual(e, date(2026, 7, 17))
        self.assertEqual(len(strikes), 5)

    def test_probe_stops_after_empty_months(self):
        # closes spanning ~10 months, listings only at the new end and NO
        # bars anywhere (no row ever appends, so the dead-bars wall — which
        # requires at least one row — stays inert): the lazy backward walk
        # must give up after EMPTY_MONTHS_STOP consecutive fresh empty
        # months instead of probing to the beginning of time.
        idx = pd.bdate_range("2025-09-01", "2026-06-19")
        closes = pd.Series(100.0, index=idx)
        months = {
            "2026-07": _contracts("2026-07-17", [90, 95, 100, 105, 110]),
            "2026-06": _contracts("2026-06-19", [90, 100, 110]),
        }
        fake = FakePolygon(months, {})
        build_iv_history("TST", closes, _r_series(), 0.0, key="k",
                         base="http://x", get_json_fn=fake,
                         log=lambda *_: None)
        ref_months = [m for kind, m in fake.calls if kind == "ref"]
        # lazy walk (newest DATE first): D=2026-06-19 discovers 07 (data) +
        # 08 (empty), D=2026-06-10 discovers 06 (data), then month(D+20)
        # crossing each earlier boundary discovers 2026-05..2025-12 = six
        # consecutive fresh empties -> stop.
        self.assertIn("2025-12", ref_months)      # the 6th consecutive empty
        self.assertNotIn("2025-11", ref_months)   # never probed past the wall

    def test_dead_bars_wall_stops_backfill(self):
        # listings exist for 8 months but bars only in the newest ~2 —
        # the walk must stop ~BARS_DEAD_DAYS_STOP days into the dead zone
        # instead of probing bars to the beginning of the series.
        from fetch_single_name_iv_history import BARS_DEAD_DAYS_STOP  # noqa: F401
        idx = pd.bdate_range("2025-11-03", "2026-06-19")
        closes = pd.Series(100.0, index=idx)
        months = {}
        for ym, exp in [("2025-12", "2025-12-19"), ("2026-01", "2026-01-16"),
                        ("2026-02", "2026-02-20"), ("2026-03", "2026-03-20"),
                        ("2026-04", "2026-04-17"), ("2026-05", "2026-05-15"),
                        ("2026-06", "2026-06-19"), ("2026-07", "2026-07-17")]:
            months[ym] = _contracts(exp, [90, 95, 100, 105, 110])
        # bars only for the 2026-06/07 expiries, covering the newest dates
        bars = {}
        for exp in (date(2026, 6, 19), date(2026, 7, 17)):
            for k in (95, 100, 105):
                bars[occ_put_ticker("TST", exp, k)] = _bars(
                    [(d.strftime("%Y-%m-%d"), 5.0)
                     for d in pd.bdate_range("2026-05-01", "2026-06-19")])
        fake = FakePolygon(months, bars)
        df, status = build_iv_history("TST", closes, _r_series(), 0.0,
                                      key="k", base="http://x",
                                      get_json_fn=fake, log=lambda *_: None)
        self.assertEqual(status, "ok")
        self.assertGreater(len(df), 0)
        # the walk must NOT have probed bars for the oldest months' contracts:
        # rows exist for 2026-05-01..06-19; the dead streak starts at
        # 2026-04-30 and reaches 42 at 2026-03-04 (20 March + 22 April
        # bdays), so no expiry older than 2026-04-17 is ever bar-probed.
        aggs_occ = {occ for kind, occ in fake.calls if kind == "aggs"}
        dead_old = occ_put_ticker("TST", date(2025, 12, 19), 100.0)
        self.assertNotIn(dead_old, aggs_occ)

    def test_entitlement_unavailable(self):
        closes, months, bars = _scenario()
        fake = FakePolygon(months, bars, forbid=True)
        df, status = build_iv_history("TST", closes, _r_series(), 0.0,
                                      key="k", base="http://x",
                                      get_json_fn=fake, log=lambda *_: None)
        self.assertEqual(status, "unavailable")
        self.assertTrue(df.empty)

    def test_inversion_roundtrip(self):
        px = black_scholes(100.0, 95.0, 0.25, 0.04, 0.005, 0.32, "put")["price"]
        iv = implied_vol(px, 100.0, 95.0, 0.25, 0.04, 0.005, "put")
        self.assertAlmostEqual(iv, 0.32, places=3)

    def test_mid_bars_403_cold_cache_unavailable(self):
        closes, months, bars = _scenario()
        fake = FakePolygon(months, bars, forbid_aggs=True)
        df, status = build_iv_history("TST", closes, _r_series(), 0.0,
                                      key="k", base="http://x",
                                      get_json_fn=fake, log=lambda *_: None)
        self.assertEqual(status, "unavailable")
        self.assertTrue(df.empty)

    def test_rate_lookup_asof_semantics(self):
        from fetch_single_name_iv_history import _rate_lookup
        r = pd.Series([0.03, 0.05],
                      index=pd.DatetimeIndex(["2026-01-05", "2026-06-01"]))
        r_at = _rate_lookup(r)
        self.assertEqual(r_at(date(2026, 1, 1)), 0.03)   # before series -> first
        self.assertEqual(r_at(date(2026, 3, 1)), 0.03)   # ffill
        self.assertEqual(r_at(date(2026, 6, 1)), 0.05)   # on the date
        self.assertEqual(r_at(date(2026, 12, 1)), 0.05)  # after -> last
        self.assertEqual(_rate_lookup(pd.Series(dtype=float))(date(2026, 1, 1)), 0.04)


class TestLoadOrRefresh(unittest.TestCase):
    def test_incremental_appends_only_new_dates(self):
        import tempfile
        closes, months, bars = _scenario()
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            fake1 = FakePolygon(months, bars)
            res1 = load_or_refresh_iv_history(
                data_dir, "TST", closes.iloc[:3], _r_series(), 0.0,
                key="k", base="http://x", get_json_fn=fake1,
                log=lambda *_: None)
            n1 = len(res1["iv"])
            fake2 = FakePolygon(months, bars)
            res2 = load_or_refresh_iv_history(
                data_dir, "TST", closes, _r_series(), 0.0,
                key="k", base="http://x", get_json_fn=fake2,
                log=lambda *_: None)
            self.assertGreater(len(res2["iv"]), n1)
            # second run must not re-ask reference for already-cached dates'
            # months beyond the tail neighborhood: cached max is 2026-06-17,
            # so every ref call is for months >= 2026-07 window of the tail
            ref2 = [m for kind, m in fake2.calls if kind == "ref"]
            self.assertTrue(all(m >= "2026-07" for m in ref2))
            self.assertEqual(res2["status"], "thin")   # 4 rows < MIN_ROWS
            self.assertEqual(res2["first_covered"],
                             res2["iv"].index.min().date())

    def test_cache_survives_refresh_failure(self):
        import tempfile
        closes, months, bars = _scenario()
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            load_or_refresh_iv_history(data_dir, "TST", closes.iloc[:3],
                                       _r_series(), 0.0, key="k",
                                       base="http://x",
                                       get_json_fn=FakePolygon(months, bars),
                                       log=lambda *_: None)
            res = load_or_refresh_iv_history(
                data_dir, "TST", closes, _r_series(), 0.0, key="k",
                base="http://x",
                get_json_fn=FakePolygon(months, bars, forbid=True),
                log=lambda *_: None)
            self.assertGreater(len(res["iv"]), 0)      # stale cache served
            self.assertEqual(res["status"], "thin")

    def test_status_ok_needs_span_and_density(self):
        # pins the fetcher's ok|thin gate AND its equality with the engine's
        # true|proxy gate, so the two modules cannot drift silently.
        from fetch_single_name_iv_history import MIN_ROWS, MIN_SPAN_DAYS
        from single_name_hedge import _IV_MIN_ROWS, _IV_MIN_SPAN_DAYS
        self.assertEqual(MIN_ROWS, 250)
        self.assertEqual(MIN_SPAN_DAYS, 540)
        self.assertEqual(MIN_ROWS, _IV_MIN_ROWS)
        self.assertEqual(MIN_SPAN_DAYS, _IV_MIN_SPAN_DAYS)

    def test_mid_bars_403_warm_cache_logs_entitlement(self):
        import tempfile
        closes, months, bars = _scenario()
        with tempfile.TemporaryDirectory() as td:
            data_dir = Path(td)
            load_or_refresh_iv_history(data_dir, "TST", closes.iloc[:3],
                                       _r_series(), 0.0, key="k",
                                       base="http://x",
                                       get_json_fn=FakePolygon(months, bars),
                                       log=lambda *_: None)
            lines: list = []
            res = load_or_refresh_iv_history(
                data_dir, "TST", closes, _r_series(), 0.0, key="k",
                base="http://x",
                get_json_fn=FakePolygon(months, bars, forbid_aggs=True),
                log=lambda *a: lines.append(" ".join(str(x) for x in a)))
            self.assertGreater(len(res["iv"]), 0)   # cache still served
            self.assertTrue(any("entitlement" in ln for ln in lines))


class TestFetchChainWrapper(unittest.TestCase):
    def test_flatten_passthrough_no_network(self):
        import fetch_options_chains as foc
        contract = {
            "details": {"ticker": "O:TST270115P00095000",
                        "contract_type": "put",
                        "exercise_style": "american",
                        "strike_price": 95.0,
                        "expiration_date": "2027-01-15"},
            "greeks": {"delta": -0.25},
            "implied_volatility": 0.35,
            "open_interest": 10,
            "day": {"close": 4.0},
            "last_quote": {"bid": 3.9, "ask": 4.1},
        }
        orig = foc._fetch_chain
        foc._fetch_chain = lambda u, lo, hi, k, b: ([contract], 123.45, "DELAYED")
        try:
            rows, spot = foc.fetch_chain("TST", 30, 300,
                                         key="k", base="http://x")
        finally:
            foc._fetch_chain = orig
        self.assertEqual(spot, 123.45)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["strike"], 95.0)
        self.assertEqual(rows[0]["polygon_ask"], 4.1)
        self.assertEqual(rows[0]["contract_type"], "put")
        self.assertEqual(rows[0]["underlying"], "TST")


if __name__ == "__main__":
    unittest.main()

# tests/test_risk_bundle.py
"""Unit + equivalence tests for parsers/risk_bundle.py — the single-source
risk-series bundle both UIs import (Phase D finale, Weak Point #4)."""
import ast
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "parsers"))

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "synth_data"

import config_local as cfg  # noqa: E402
import risk_bundle as rb  # noqa: E402
from treasury_proxy import treasury_proxy  # noqa: E402


class TestImportHygiene(unittest.TestCase):
    """risk_bundle must stay importable by BOTH UIs: engine-level imports only
    (the circular-import / Streamlit-coupling guard from the spec)."""

    def test_no_ui_layer_imports(self):
        src = (ROOT / "parsers" / "risk_bundle.py").read_text(encoding="utf-8")
        banned = {"streamlit", "terminal", "app", "theme", "plotly"}
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            for n in names:
                self.assertNotIn(
                    n.split(".")[0], banned,
                    f"UI-layer import {n!r} found in risk_bundle.py")


class TestBuildSnapshotWeights(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2026-01-02", periods=5, freq="B")
        self.daily = pd.DataFrame(
            {s: np.linspace(100.0, 101.0, 5)
             for s in ("AAA", "BBB", "SPY", "SGOV")}, index=idx)
        self.snap = pd.DataFrame([
            {"bucket": "B1", "asset_class": "equity_stock", "account_id": "X1",
             "symbol": "AAA", "description": "Alpha Corp",
             "statement_date": pd.Timestamp("2026-05-31"),
             "market_value": 600.0},
            {"bucket": "B1", "asset_class": "cash", "account_id": "X1",
             "symbol": "CASH", "description": "Sweep",
             "statement_date": pd.Timestamp("2026-05-31"),
             "market_value": 100.0},
            {"bucket": "B2", "asset_class": "option_call", "account_id": "X1",
             "symbol": "AAA260117C100", "description": "Call",
             "statement_date": pd.Timestamp("2026-05-31"),
             "market_value": 50.0},
            {"bucket": "B2", "asset_class": "equity_stock", "account_id": "X1",
             "symbol": "BBB", "description": "Beta Corp",
             "statement_date": pd.Timestamp("2026-05-31"),
             "market_value": 400.0},
        ])

    def test_drops_cash_and_options_and_normalizes(self):
        w, wmv = rb.build_snapshot_weights(self.snap, None, None, self.daily)
        self.assertEqual(set(w.index), {"AAA", "BBB"})
        self.assertAlmostEqual(float(w.sum()), 1.0, places=12)
        self.assertAlmostEqual(float(w["AAA"]), 0.6, places=12)
        self.assertAlmostEqual(float(wmv["AAA"]), 600.0, places=9)

    def test_none_filters_equal_full_choice_lists(self):
        full_b = sorted(self.snap["bucket"].unique())
        full_c = sorted(self.snap["asset_class"].unique())
        w_none, mv_none = rb.build_snapshot_weights(
            self.snap, None, None, self.daily)
        w_full, mv_full = rb.build_snapshot_weights(
            self.snap, full_b, full_c, self.daily)
        pd.testing.assert_series_equal(w_none, w_full)
        pd.testing.assert_series_equal(mv_none, mv_full)

    def test_bucket_filter_narrows(self):
        w, _ = rb.build_snapshot_weights(self.snap, ["B1"], None, self.daily)
        self.assertEqual(set(w.index), {"AAA"})
        self.assertAlmostEqual(float(w["AAA"]), 1.0, places=12)

    def test_tlh_sleeve_folds_to_spy(self):
        snap = self.snap.copy()
        snap.loc[snap["symbol"] == "AAA", "account_id"] = cfg.TLH_ACCOUNT_ID
        w, _ = rb.build_snapshot_weights(snap, None, None, self.daily)
        self.assertIn("SPY", w.index)
        self.assertNotIn("AAA", w.index)

    def test_ladder_rung_uses_duration_proxy(self):
        snap = self.snap.copy()
        desc = "UNITED STATES TREAS NTS 4.25% 05/15/2033"
        mask = snap["symbol"] == "BBB"
        snap.loc[mask, "bucket"] = "JPM Treasury Ladder"
        snap.loc[mask, "symbol"] = "CUSIP123"
        snap.loc[mask, "description"] = desc
        expect = treasury_proxy(desc, pd.Timestamp("2026-05-31"))
        daily = self.daily.copy()
        daily[expect] = 100.0  # proxy must be price-covered or it SGOV-falls
        w, _ = rb.build_snapshot_weights(snap, None, None, daily)
        self.assertIn(expect, w.index)
        self.assertNotIn("CUSIP123", w.index)

    def test_uncovered_symbol_falls_back_to_sgov(self):
        snap = self.snap.copy()
        snap.loc[snap["symbol"] == "BBB", "symbol"] = "NOPRICE"
        w, _ = rb.build_snapshot_weights(snap, None, None, self.daily)
        self.assertIn("SGOV", w.index)
        self.assertNotIn("NOPRICE", w.index)

    def test_empty_inputs(self):
        w, wmv = rb.build_snapshot_weights(
            self.snap.iloc[0:0], None, None, self.daily)
        self.assertTrue(w.empty and wmv.empty)
        w2, _ = rb.build_snapshot_weights(self.snap, None, None, pd.DataFrame())
        self.assertTrue(w2.empty)


class TestSpyDailyReturns(unittest.TestCase):
    def setUp(self):
        self.idx = pd.date_range("2026-01-02", periods=6, freq="B")
        self.daily = pd.DataFrame(
            {"SPY": [100.0, 101.0, 102.0, 101.0, 103.0, 104.0]},
            index=self.idx)

    def test_tr_series_preferred(self):
        cal = pd.date_range(self.idx.min(), self.idx.max(), freq="D")
        tr = pd.Series(np.linspace(500.0, 505.0, len(cal)), index=cal)
        got = rb.spy_daily_returns(self.daily, tr)
        exp = tr.reindex(self.idx, method="ffill").pct_change().dropna()
        pd.testing.assert_series_equal(got, exp)

    def test_price_only_fallback(self):
        got = rb.spy_daily_returns(self.daily, pd.Series(dtype=float))
        exp = self.daily["SPY"].pct_change().dropna()
        pd.testing.assert_series_equal(got, exp)

    def test_no_daily_prices_or_spy(self):
        self.assertTrue(
            rb.spy_daily_returns(pd.DataFrame(), pd.Series(dtype=float)).empty)
        no_spy = self.daily.rename(columns={"SPY": "AAA"})
        self.assertTrue(
            rb.spy_daily_returns(no_spy, pd.Series(dtype=float)).empty)


class TestSpyOverlapWindow(unittest.TestCase):
    """spy_rets in the bundle must not predate port_rets: SPY head history
    the portfolio never lived through (e.g. a pre-inception crash) fed the
    VaR/CVaR/worst-day tiles until 2026-07 (TK feedback batch D)."""

    def test_bundle_truncates_spy_head(self):
        from terminal import holdings_service as hs
        from terminal.performance_service import _prepare_portfolio_twr
        frames = hs.load_frames(FIXTURE)
        twr = _prepare_portfolio_twr(frames.twr_portfolio)
        bench = hs._bench_tr_series(frames)
        daily = frames.daily_prices
        latest_dt = frames.positions["statement_date"].max()
        # Graft a synthetic SPY head predating every portfolio return AND
        # the fixture's bench series (keep indexes sorted, no overlap).
        first = daily.index.min()
        if bench is not None and not bench.empty:
            first = min(first, bench.index.min())
        pre_idx = pd.bdate_range(end=first - pd.Timedelta(days=1), periods=40)
        pre = pd.DataFrame(np.nan, index=pre_idx, columns=daily.columns)
        pre["SPY"] = np.linspace(90.0, 100.0, 40)
        daily2 = pd.concat([pre, daily]).sort_index()
        # bench_tr is CALENDAR-daily (weekends included — statement month-ends
        # land on them), so the grafted head must be calendar too.
        pre_cal = pd.date_range(end=first - pd.Timedelta(days=1), periods=56,
                                freq="D")
        bench2 = (pd.concat([pd.Series(np.linspace(450.0, 500.0, 56),
                                       index=pre_cal), bench]).sort_index()
                  if bench is not None and not bench.empty else bench)
        b = rb.build_risk_series_bundle(
            positions=frames.positions,
            positions_monthly=frames.positions_monthly,
            latest_dt=latest_dt, bucket_filter=None, class_filter=None,
            account_active=False, class_active=False,
            daily_prices=daily2, bench_tr=bench2, twr_portfolio=twr)
        self.assertFalse(b["port_rets"].empty)
        self.assertFalse(b["spy_rets"].empty)
        self.assertGreaterEqual(b["spy_rets"].index.min(),
                                b["port_rets"].index.min())


class TestSynthesizeMonthlyReturns(unittest.TestCase):
    def setUp(self):
        idx = pd.date_range("2026-01-02", "2026-03-31", freq="B")
        self.port = pd.Series(0.001, index=idx)
        self.twr = pd.DataFrame({
            "statement_date": pd.to_datetime(
                ["2026-01-31", "2026-02-28", "2026-03-31"]),
            "prev_stmt_date": pd.to_datetime(
                [pd.NaT, "2026-01-31", "2026-02-28"]),
        })

    def test_first_month_nan_anchor(self):
        m = rb.synthesize_monthly_returns(self.port, self.twr)
        self.assertEqual(len(m), 3)
        self.assertTrue(np.isnan(m.iloc[0]))  # wealth-index baseline contract

    def test_window_compounding(self):
        m = rb.synthesize_monthly_returns(self.port, self.twr)
        w = self.port[(self.port.index > pd.Timestamp("2026-01-31"))
                      & (self.port.index <= pd.Timestamp("2026-02-28"))]
        self.assertAlmostEqual(float(m.iloc[1]),
                               float((1.0 + w).prod() - 1.0), places=12)

    def test_empty_window_is_nan(self):
        twr = pd.concat([self.twr, pd.DataFrame({
            "statement_date": [pd.Timestamp("2026-04-30")],
            "prev_stmt_date": [pd.Timestamp("2026-04-01")]})],
            ignore_index=True)
        m = rb.synthesize_monthly_returns(self.port, twr)  # no April bars
        self.assertTrue(np.isnan(m.iloc[-1]))

    def test_empty_inputs(self):
        self.assertTrue(rb.synthesize_monthly_returns(
            pd.Series(dtype=float), self.twr).empty)
        self.assertTrue(rb.synthesize_monthly_returns(
            self.port, pd.DataFrame()).empty)


class TestDailyPortfolioReturns(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from terminal import holdings_service as hs
        cls.frames = hs.load_frames(FIXTURE)

    def test_matches_manual_pipeline_on_fixture(self):
        from monthly_normalize import monthly_normalize
        from risk_metrics import (synthesize_portfolio_returns_historical,
                                  weights_per_snap_monthly)
        daily = self.frames.daily_prices
        wps = weights_per_snap_monthly(
            monthly_normalize(self.frames.positions),
            lambda s: rb.build_snapshot_weights(s, None, None, daily)[0])
        exp = (synthesize_portfolio_returns_historical(wps, daily)
               if wps else pd.Series(dtype=float))
        got = rb.daily_portfolio_returns(
            self.frames.positions, daily, None, None)
        pd.testing.assert_series_equal(got, exp)

    def test_empty_daily_prices(self):
        self.assertTrue(rb.daily_portfolio_returns(
            self.frames.positions, pd.DataFrame(), None, None).empty)


class TestBuildRiskSeriesBundle(unittest.TestCase):
    """Engine bundle == terminal rs._bundle on the fixture, key by key.

    Historically a true independent-implementation equivalence proof (it ran
    against the OLD inline rs._bundle body before the Task-4 re-point);
    since then it is the wiring pin — rs._bundle's input prep must keep
    feeding the shared engine the canonical inputs."""

    # Hand-maintained: every series-valued bundle key must be listed here or
    # _compare silently skips it (test_engine_key_set only pins key NAMES).
    SERIES_KEYS = ["weights", "port_rets", "spy_rets", "monthly",
                   "spy_monthly", "wealth_index", "dd_full_pct",
                   "spy_dd_full_pct"]

    @classmethod
    def setUpClass(cls):
        from monthly_normalize import slice_as_of_month
        from terminal import holdings_service as hs
        from terminal import risk_service as rs
        from terminal.performance_service import _prepare_portfolio_twr
        cls.rs = rs
        cls.frames = hs.load_frames(FIXTURE)
        cls.twr = _prepare_portfolio_twr(cls.frames.twr_portfolio)
        cls.bench = hs._bench_tr_series(cls.frames)
        cls.latest_dt = cls.frames.positions["statement_date"].max()
        snap = slice_as_of_month(cls.frames.positions_monthly, cls.latest_dt)
        cls.all_buckets = sorted(snap["bucket"].astype(str).unique())
        cls.all_classes = sorted(snap["asset_class"].astype(str).unique())

    def _engine(self, bucket_filter, class_filter, account_active,
                class_active):
        return rb.build_risk_series_bundle(
            positions=self.frames.positions,
            positions_monthly=self.frames.positions_monthly,
            latest_dt=self.latest_dt,
            bucket_filter=bucket_filter, class_filter=class_filter,
            account_active=account_active, class_active=class_active,
            daily_prices=self.frames.daily_prices,
            bench_tr=self.bench, twr_portfolio=self.twr)

    def _compare(self, bucket_filter, class_filter, account_active,
                 class_active):
        old = self.rs._bundle(self.frames, bucket_filter, class_filter,
                              account_active, class_active)
        new = self._engine(bucket_filter, class_filter, account_active,
                           class_active)
        for k in self.SERIES_KEYS:
            pd.testing.assert_series_equal(
                new[k], old[k], check_names=False, obj=f"bundle[{k}]")
        pd.testing.assert_frame_equal(new["latest_snap"], old["latest_snap"])
        pd.testing.assert_frame_equal(new["synthesis_gaps"],
                                      old["synthesis_gaps"])
        self.assertEqual(new["monthly_source"], old["monthly_source"])
        self.assertAlmostEqual(new["nav_latest"], old["nav_latest"], places=9)
        return new

    def test_unfiltered_equivalence_and_twr_branch(self):
        new = self._compare(self.all_buckets, self.all_classes, False, False)
        self.assertEqual(new["monthly_source"], "twr")

    def test_account_filtered_equivalence_synthetic_branch(self):
        new = self._compare(self.all_buckets[:1], self.all_classes,
                            True, False)
        self.assertEqual(new["monthly_source"], "synthetic")

    def test_class_filtered_equivalence_synthetic_branch(self):
        modelable = [c for c in self.all_classes
                     if c != "cash" and not c.startswith("option")][:1]
        new = self._compare(self.all_buckets, modelable, False, True)
        self.assertFalse(new["weights"].empty)  # degenerate filter would
        self.assertEqual(new["monthly_source"], "synthetic")  # prove nothing

    def test_engine_key_set(self):
        new = self._engine(self.all_buckets, self.all_classes, False, False)
        self.assertEqual(set(new.keys()), {
            "latest_snap", "weights", "weights_mv", "port_rets", "spy_rets",
            "monthly", "monthly_source", "spy_monthly", "wealth_index",
            "dd_full_pct", "spy_dd_full_pct", "nav_latest", "synthesis_gaps",
            "filter_state"})
        self.assertEqual(new["filter_state"],
                         {"account_active": False, "class_active": False})

    def test_latest_dt_none_yields_empty_universe(self):
        new = rb.build_risk_series_bundle(
            positions=self.frames.positions,
            positions_monthly=self.frames.positions_monthly,
            latest_dt=None,
            bucket_filter=self.all_buckets, class_filter=self.all_classes,
            account_active=False, class_active=False,
            daily_prices=self.frames.daily_prices,
            bench_tr=self.bench, twr_portfolio=self.twr)
        self.assertTrue(new["latest_snap"].empty)
        self.assertEqual(new["nav_latest"], 0.0)


if __name__ == "__main__":
    unittest.main()

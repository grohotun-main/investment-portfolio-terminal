# tests/test_terminal_risk.py
import dataclasses
import json
import math
import os
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "parsers"))

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "synth_data"

from terminal import holdings_service as hs
from terminal import risk_service as rs
from risk_metrics import (compute_calmar, compute_sharpe, compute_sortino,
                          window_drawdown_pct)


def _deep_close(a, b, *, rel=1e-6, abs_=1e-9, path="root"):
    """Structural-exact, float-TOLERANT deep compare (same as the Factor golden:
    the rolling beta/alpha + synthesis pass through many float ops that aren't
    bit-reproducible across BLAS builds, so pin structure + formatted strings
    exactly but compare raw floats with a relative tolerance)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return None if a is b else f"{path}: {a!r} != {b!r}"
    if isinstance(a, dict) and isinstance(b, dict):
        if a.keys() != b.keys():
            return f"{path}: key mismatch {set(a) ^ set(b)}"
        for k in a:
            m = _deep_close(a[k], b[k], rel=rel, abs_=abs_, path=f"{path}.{k}")
            if m:
                return m
        return None
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            m = _deep_close(x, y, rel=rel, abs_=abs_, path=f"{path}[{i}]")
            if m:
                return m
        return None
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return (None if math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)
                else f"{path}: {a!r} !~ {b!r}")
    return None if a == b else f"{path}: {a!r} != {b!r}"


class TestContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = rs.build_risk_view(cls.frames)

    def test_keys(self):
        self.assertEqual(set(self.view.keys()), {
            "meta", "caption", "conc_caption", "state", "filter_note",
            "coverage_gaps", "daily_available", "daily_unavailable_message",
            "risk_adjusted", "drawdown", "concentration", "daily", "beta",
            "quadrant_html"})

    def test_state_available(self):
        self.assertTrue(self.view["state"]["available"])

    def test_five_tiles_each_section(self):
        self.assertEqual(len(self.view["risk_adjusted"]["tiles"]), 5)
        self.assertEqual(len(self.view["drawdown"]["tiles"]), 5)
        if self.view["daily"]:
            self.assertEqual(len(self.view["daily"]["tiles"]), 5)
        if self.view["beta"] and self.view["beta"]["available"]:
            self.assertEqual(len(self.view["beta"]["tiles"]), 5)

    def test_jsonable_no_nan(self):
        body = json.dumps(self.view, allow_nan=False)  # raises if any NaN leaks
        self.assertNotIn("NaN", body)


class TestEngineParity(unittest.TestCase):
    """Tile strings equal an independent recompute of the importable engines on
    the same (unfiltered) bundle inputs — the '1:1 numbers' gate at the data
    layer (mirrors the Factor tab's engine-parity test)."""

    @classmethod
    def setUpClass(cls):
        cls.frames = hs.load_frames(FIXTURE)
        cls.view = rs.build_risk_view(cls.frames)
        rf_series = rs._load_rf(cls.frames.data_dir)
        cls.rf = rf_series if not rf_series.empty else rs.RF_FALLBACK_ANNUAL
        twr = rs._prepare_portfolio_twr(cls.frames.twr_portfolio)
        idx = pd.DatetimeIndex(twr["statement_date"].values)
        cls.monthly = pd.Series(twr["return_pct"].astype(float).values, index=idx)

    def test_risk_adjusted_tiles(self):
        r_all = self.monthly.dropna()
        r_1y, r_3y = r_all.tail(12), r_all.tail(36)
        dd_3y = window_drawdown_pct(r_3y)
        exp = [
            f"{compute_sharpe(r_1y, self.rf):.2f}",
            f"{compute_sharpe(r_3y, self.rf):.2f}",
            f"{compute_sortino(r_1y, self.rf):.2f}",
            f"{compute_sortino(r_3y, self.rf):.2f}",
            f"{compute_calmar(r_3y, dd_3y):.2f}",
        ]
        got = [t["value"] for t in self.view["risk_adjusted"]["tiles"]]
        self.assertEqual(got, exp)

    def test_annualized_vol_tile(self):
        r_3y = self.monthly.dropna().tail(36)
        ann_vol = float(r_3y.std(ddof=1) * np.sqrt(12) * 100.0) if len(r_3y) >= 2 else np.nan
        vol_tile = self.view["drawdown"]["tiles"][4]
        self.assertEqual(vol_tile["label"], "Annualized vol (3Y)")
        self.assertEqual(vol_tile["value"],
                         f"{ann_vol:.1f}%" if math.isfinite(ann_vol) else "—")

    def test_concentration_matches_engine(self):
        from risk_metrics import compute_concentration
        from monthly_normalize import slice_as_of_month
        pm = self.frames.positions_monthly
        latest_dt = self.frames.positions["statement_date"].max()
        snap = slice_as_of_month(pm, latest_dt)
        is_opt = snap["asset_class"].astype(str).str.startswith("option")
        snap = snap[(snap["asset_class"] != "cash") & (~is_opt)].copy()
        snap.loc[snap["asset_class"] == "tax_loss_harvesting", "symbol"] = "SPY"
        snap.loc[snap["bucket"] == "Treasury Ladder", "symbol"] = "SGOV"
        snap = snap.dropna(subset=["symbol"])
        conc = compute_concentration(snap.groupby("symbol")["market_value"].sum())
        tiles = self.view["concentration"]["tiles"]
        if conc["n_positions"] > 0:
            self.assertEqual(tiles[0]["value"], f"{conc['effective_n']:.1f}")
            self.assertEqual(tiles[1]["value"], f"{conc['max_pct']:.1f}%")
            self.assertEqual(tiles[2]["value"], f"{conc['top5_pct']:.1f}%")


class TestChartStructureLongSeries(unittest.TestCase):
    """The synth fixture's daily history is too short to exercise the rolling /
    histogram / scatter chart paths. Drive the section builders directly with a
    constructed 400-day return series so those paths run in CI (not only in the
    real-data live smoke)."""

    def _bundle(self):
        rng = pd.date_range("2024-01-01", periods=400, freq="B")
        # deterministic, non-degenerate daily returns (no Math.random analog)
        i = np.arange(400)
        spy = pd.Series(0.0004 + 0.01 * np.sin(i / 7.0), index=rng)
        port = pd.Series(0.0003 + 0.008 * np.sin(i / 7.0 + 0.3)
                         + 0.002 * np.cos(i / 3.0), index=rng)
        monthly = pd.Series([0.01, -0.02, 0.03, np.nan, 0.015],
                            index=pd.date_range("2024-01-31", periods=5, freq="ME"))
        return {"port_rets": port, "spy_rets": spy, "monthly": monthly,
                "spy_monthly": pd.Series(dtype=float),
                "wealth_index": pd.Series(dtype=float),
                "dd_full_pct": pd.Series(dtype=float),
                "spy_dd_full_pct": pd.Series(dtype=float),
                "nav_latest": 0.0, "bench_tr": pd.Series(dtype=float)}

    def test_daily_section_charts_populate(self):
        b = self._bundle()
        daily = rs._daily_vol(b)
        self.assertTrue(daily["distribution"]["available"])
        self.assertEqual(len(daily["distribution"]["port_counts"]), 50)
        self.assertEqual(len(daily["distribution"]["fit"]), 200)
        self.assertTrue(daily["rolling_vol"]["available"])
        self.assertTrue(daily["rolling_vol"]["series"][0]["points"])

    def test_distribution_sparse_bin_dates(self):
        """Sparse bins (1..SPARSE_BIN_DATES_MAX obs) list their dates and the
        lists reconcile exactly with the counts; dense bins carry no entry."""
        b = self._bundle()
        b["port_rets"] = b["port_rets"].copy()
        b["port_rets"].iloc[10] = -0.045   # lone -4.5% day -> a 1-obs tail bin
        dist = rs._daily_vol(b)["distribution"]
        for key, counts in (("port_dates", dist["port_counts"]),
                            ("spy_dates", dist["spy_counts"])):
            dates = dist[key]
            self.assertIsInstance(dates, dict)
            for bstr, dts in dates.items():
                c = counts[int(bstr)]
                self.assertGreaterEqual(c, 1)
                self.assertLessEqual(c, rs.SPARSE_BIN_DATES_MAX)
                self.assertEqual(len(dts), c)
                for d in dts:
                    self.assertRegex(d, r"^\d{4}-\d{2}-\d{2}$")
            for i, c in enumerate(counts):   # every sparse bin has its entry
                if 1 <= c <= rs.SPARSE_BIN_DATES_MAX:
                    self.assertIn(str(i), dates)
        planted = str(b["port_rets"].index[10].date())
        self.assertTrue(any(planted in dts
                            for dts in dist["port_dates"].values()),
                        "planted outlier date missing from sparse-bin dates")

    def test_beta_section_charts_populate(self):
        b = self._bundle()
        beta = rs._beta(b)
        self.assertTrue(beta["available"])
        self.assertTrue(beta["scatter"]["available"])
        self.assertTrue(beta["rolling_beta"]["available"])
        self.assertTrue(beta["rolling_alpha"]["available"])
        # all overlay series share ONE common index length (drawOverlayChart
        # aligns purely by position) — ragged series would misalign the chart.
        lens = {len(s["points"]) for s in beta["rolling_beta"]["series"]}
        self.assertEqual(len(lens), 1)

    def test_rolling_sharpe_populates(self):
        b = self._bundle()
        ra = rs._risk_adjusted(b, rs.RF_FALLBACK_ANNUAL)
        self.assertTrue(ra["rolling_sharpe"]["available"])
        lens = {len(s["points"]) for s in ra["rolling_sharpe"]["series"]}
        self.assertEqual(len(lens), 1)


class TestFilteredSynthesis(unittest.TestCase):
    """An Account / Asset-class filter flips the monthly source to 'synthetic'
    (statement TWR has no per-slice series) — verify the bundle path switches."""

    def test_filter_switches_to_synthetic(self):
        frames = hs.load_frames(FIXTURE)
        view = rs.build_risk_view(frames)
        opts = view["meta"]["classes"]
        if not opts:
            self.skipTest("no class options on fixture")
        filtered = rs.build_risk_view(frames, asset_class=opts[0]["id"])
        self.assertTrue(filtered["meta"]["class_filter_active"])
        self.assertIsNotNone(filtered["filter_note"])


class TestEmptyStates(unittest.TestCase):
    def test_no_twr(self):
        frames = hs.load_frames(FIXTURE)
        frames2 = dataclasses.replace(
            frames, twr_portfolio=frames.twr_portfolio.iloc[0:0])
        view = rs.build_risk_view(frames2)
        self.assertFalse(view["state"]["available"])
        self.assertEqual(view["state"]["unavailable"], "no_twr")
        self.assertIsNone(view["risk_adjusted"])


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["APP_DATA_DIR"] = str(FIXTURE)
        from fastapi.testclient import TestClient
        from terminal.server import app
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls):
        os.environ.pop("APP_DATA_DIR", None)

    def test_risk_ok(self):
        r = self.client.get("/api/risk")
        self.assertEqual(r.status_code, 200)
        self.assertIn("risk_adjusted", r.json())

    def test_unknown_account_422(self):
        r = self.client.get("/api/risk", params={"account": "nope"})
        self.assertEqual(r.status_code, 422)

    def test_missing_data_dir_503(self):
        os.environ["APP_DATA_DIR"] = str(ROOT / "tests" / "no_such_dir")
        try:
            r = self.client.get("/api/risk")
            self.assertEqual(r.status_code, 503)
        finally:
            os.environ["APP_DATA_DIR"] = str(FIXTURE)


class TestGolden(unittest.TestCase):
    GOLDEN = (Path(__file__).resolve().parent / "fixtures"
              / "terminal_risk_golden.json")

    def test_matches_golden(self):
        frames = hs.load_frames(FIXTURE)
        view = rs.build_risk_view(frames)
        self.assertTrue(self.GOLDEN.exists(),
                        "golden snapshot missing — regenerate intentionally")
        expected = json.loads(self.GOLDEN.read_text(encoding="utf-8"))
        mismatch = _deep_close(view, expected)
        self.assertIsNone(mismatch,
                          f"risk view diverges from golden at {mismatch}")


if __name__ == "__main__":
    unittest.main()

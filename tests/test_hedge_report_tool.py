"""Offline end-to-end for tools/hedge_report.py (spec test 13)."""
import sys
import unittest
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))
sys.path.insert(0, str(ROOT / "tools"))

from hedge_report import (assert_sale_covering_chain, build_report_data,  # noqa: E402
                          render_html)

FIXTURE = ROOT / "tests" / "fixtures" / "single_name_chain_fixture.csv"
TODAY = date(2026, 7, 6)
SELL_BY = date(2027, 1, 15)


def _chain_rows():
    df = pd.read_csv(FIXTURE)
    return df.where(pd.notna(df), None).to_dict("records")


def _hist_payload():
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2019-01-02", periods=1900)
    rets = rng.normal(0.0005, 0.022, len(idx))
    price = pd.Series(200.0 * np.cumprod(1.0 + rets), index=idx)
    price = price / price.iloc[-1] * 200.0   # end exactly at spot 200
    dser = pd.Series([0.40] * 4,
                     index=pd.DatetimeIndex(pd.to_datetime(
                         ["2025-09-10", "2025-12-10", "2026-03-11",
                          "2026-06-10"])))
    return {"status": "ok", "price": price, "tr": price.copy(),
            "dser": dser, "asof": price.index[-1],
            "n_days": len(price), "stale": False, "msg": ""}


def _iv_payload():
    idx = pd.bdate_range("2024-01-02", periods=600)
    iv = pd.Series(np.linspace(0.30, 0.44, len(idx)), index=idx)
    return {"iv": iv, "first_covered": idx[0].date(), "status": "ok"}


class TestBuildReportData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data = build_report_data(
            ticker="TST", shares=1800, sell_by=SELL_BY, today=TODAY,
            floors=[0.10, 0.20, 0.30, 0.40, 0.50],
            kicker_otm=0.45, kicker_budget_pct=0.10,
            chain_rows=_chain_rows(), spot=200.0, hist=_hist_payload(),
            rf=0.04, vix=None, iv_payload=_iv_payload())

    def test_menu_solves_expected_strikes(self):
        menu = self.data["menu"]
        self.assertEqual(len(menu), 5)
        strikes = [p["strike"] for p in menu]
        # 30% floor: the ask-less K=150 (last 7.00, stale) beats K=160 @ 9.00
        self.assertEqual(strikes, [210.0, 180.0, 150.0, 140.0, 120.0])
        self.assertTrue(all(p["expiry"] == "2027-01-15" for p in menu))
        self.assertTrue(all(p["contracts"] == 18 for p in menu))
        costs = [p["total_cost"] for p in menu]
        self.assertEqual(costs, sorted(costs, reverse=True))

    def test_vol_and_odds_present(self):
        self.assertEqual(self.data["vol"]["iv_source"], "true")
        self.assertIsNotNone(self.data["vol"]["iv_percentile"])
        d = self.data["odds"][0.20]
        self.assertIn("prob", d)
        self.assertIn(d["source"], ("empirical", "gpd", "none"))

    def test_kicker_and_marks(self):
        k = self.data["kicker"]
        self.assertIsNotNone(k)
        self.assertLessEqual(k["strike"], 200.0 * 0.55)
        self.assertGreaterEqual(len(self.data["crash_marks"]), 18)

    def test_no_collar_anywhere(self):
        html = render_html(self.data).lower()
        self.assertNotIn("collar", html)
        self.assertNotIn("covered call", html)

    def test_stale_dominated_print_excluded_with_warning(self):
        rows = _chain_rows() + [{
            "underlying": "TST", "contract_ticker": "O:TST270115P00215000",
            "contract_type": "put", "strike": 215.0,
            "expiration_date": "2027-01-15", "polygon_bid": None,
            "polygon_ask": None, "polygon_price": 5.00,   # << 210's 28.00
            "polygon_iv": 0.30, "polygon_delta": -0.6,
            "polygon_open_interest": 3}]
        data = build_report_data(
            ticker="TST", shares=1800, sell_by=SELL_BY, today=TODAY,
            floors=[0.10], kicker_otm=0.45, kicker_budget_pct=0.10,
            chain_rows=rows, spot=200.0, hist=_hist_payload(),
            rf=0.04, vix=None, iv_payload=None)
        # without the filter the 215 print (all-in loss ~ -5%) would win
        self.assertEqual(data["menu"][0]["strike"], 210.0)
        self.assertTrue(any("unexecutable bargain" in w
                            for w in data["warnings"]))


class TestRenderHtml(unittest.TestCase):
    def test_sections_and_hygiene(self):
        data = build_report_data(
            ticker="TST", shares=1800, sell_by=SELL_BY, today=TODAY,
            floors=[0.10, 0.20, 0.30, 0.40, 0.50],
            kicker_otm=0.45, kicker_budget_pct=0.10,
            chain_rows=_chain_rows(), spot=200.0, hist=_hist_payload(),
            rf=0.04, vix=None, iv_payload=_iv_payload())
        html = render_html(data)
        for marker in ("Your position today", "The insurance menu",
                       "pricey day to buy insurance", "The trade tickets",
                       "If the crash comes early", "crash kicker",
                       "How sure are these numbers", "Taxes", "Glossary",
                       "tax professional", "sell the puts",
                       "O:TST270115P00180000"):
            self.assertIn(marker, html)
        self.assertNotIn(">nan<", html.lower())
        self.assertNotIn(">none<", html.lower())
        self.assertIn("plotly", html.lower())   # inlined chart lib

    def test_odd_lot_warning(self):
        data = build_report_data(
            ticker="TST", shares=1850, sell_by=SELL_BY, today=TODAY,
            floors=[0.20], kicker_otm=0.45, kicker_budget_pct=0.10,
            chain_rows=_chain_rows(), spot=200.0, hist=_hist_payload(),
            rf=0.04, vix=None, iv_payload=None)
        self.assertTrue(any("50 shares" in w for w in data["warnings"]))
        self.assertEqual(data["vol"]["iv_source"], "proxy")


class TestDedicatedIvSolveRows(unittest.TestCase):
    def test_far_sale_uses_dedicated_window(self):
        rows = [r for r in _chain_rows()
                if str(r["expiration_date"]) != "2026-08-21"]
        solve_rows = [r for r in _chain_rows()
                      if str(r["expiration_date"]) == "2026-08-21"]
        data = build_report_data(
            ticker="TST", shares=1800, sell_by=SELL_BY, today=TODAY,
            floors=[0.20], kicker_otm=0.45, kicker_budget_pct=0.10,
            chain_rows=rows, spot=200.0, hist=_hist_payload(),
            rf=0.04, vix=None, iv_payload=_iv_payload(),
            iv_solve_rows=solve_rows)
        self.assertIsNotNone(data["vol"]["iv_percentile"])

    def test_no_solve_candidate_degrades_honestly(self):
        rows = [r for r in _chain_rows()
                if str(r["expiration_date"]) != "2026-08-21"]
        data = build_report_data(
            ticker="TST", shares=1800, sell_by=SELL_BY, today=TODAY,
            floors=[0.20], kicker_otm=0.45, kicker_budget_pct=0.10,
            chain_rows=rows, spot=200.0, hist=_hist_payload(),
            rf=0.04, vix=None, iv_payload=_iv_payload())
        self.assertEqual(data["vol"]["iv_source"], "true")
        self.assertIsNone(data["vol"]["iv_percentile"])
        html = render_html(data)
        self.assertIn("couldn't be solved from the chain", html)
        # crash marks price off the package's own IV, not the crash anchor
        self.assertGreater(len(data["crash_marks"]), 0)
        self.assertAlmostEqual(float(data["crash_marks"]["iv_used"].min()),
                               0.34, places=6)


class TestSaleCoveringChainGuard(unittest.TestCase):
    def test_empty_chain_exits_3(self):
        with self.assertRaises(SystemExit) as cm:
            assert_sale_covering_chain([], "TST", SELL_BY)
        self.assertEqual(cm.exception.code, 3)

    def test_no_covering_expiry_exits_3_listing_found(self):
        rows = [{"expiration_date": "2026-08-21"},
                {"expiration_date": "2026-12-18"}]
        import io
        from contextlib import redirect_stdout
        buf = io.StringIO()
        with redirect_stdout(buf), self.assertRaises(SystemExit) as cm:
            assert_sale_covering_chain(rows, "TST", SELL_BY)
        self.assertEqual(cm.exception.code, 3)
        out = buf.getvalue()
        self.assertIn("2026-08-21", out)
        self.assertIn("2026-12-18", out)

    def test_covering_expiry_passes(self):
        rows = [{"expiration_date": "2027-01-15"}]
        assert_sale_covering_chain(rows, "TST", SELL_BY)   # no raise


if __name__ == "__main__":
    unittest.main()

"""Tests for parsers/crash_betas.py."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from crash_betas import (  # noqa: E402
    compute_window_return,
    compute_crash_betas,
)


def _mk_prices(dates, **series):
    return pd.DataFrame(series, index=pd.DatetimeIndex(dates))


class WindowReturnTests(unittest.TestCase):
    def test_simple_pct_change_over_window(self):
        df = _mk_prices(
            ["2020-02-20", "2020-03-23"],
            SPY=[100.0, 70.0],
        )
        r = compute_window_return(df["SPY"], "2020-02-20", "2020-03-23")
        self.assertAlmostEqual(r, -0.30, places=4)

    def test_missing_start_uses_first_available_after(self):
        # Start date is a weekend; should pick up the next available trading day.
        df = _mk_prices(
            ["2020-02-24", "2020-03-23"],  # 2020-02-22/23 was weekend
            SPY=[95.0, 70.0],
        )
        r = compute_window_return(df["SPY"], "2020-02-20", "2020-03-23")
        self.assertAlmostEqual(r, (70.0 - 95.0) / 95.0, places=4)

    def test_returns_nan_when_window_empty(self):
        df = _mk_prices(["2020-02-20"], SPY=[100.0])
        r = compute_window_return(df["SPY"], "2025-01-01", "2025-01-15")
        self.assertTrue(np.isnan(r))


class ComputeCrashBetasTests(unittest.TestCase):
    def test_median_across_two_windows(self):
        # In window 1, NVDA falls 50% vs SPY 25% → beta_window = 2.0
        # In window 2, NVDA falls 20% vs SPY 10% → beta_window = 2.0
        # median across windows = 2.0
        dates = ["2020-02-20", "2020-03-23", "2022-01-01", "2022-06-30"]
        df = _mk_prices(
            dates,
            SPY=[100.0, 75.0, 100.0, 90.0],
            NVDA=[100.0, 50.0, 100.0, 80.0],
        )
        windows = [("2020-02-20", "2020-03-23"),
                   ("2022-01-01", "2022-06-30")]
        out = compute_crash_betas(df, tickers=["NVDA"], spy_ticker="SPY",
                                   windows=windows)
        self.assertAlmostEqual(out["NVDA"], 2.0, places=4)

    def test_spy_self_beta_is_one(self):
        dates = ["2020-02-20", "2020-03-23"]
        df = _mk_prices(dates, SPY=[100.0, 70.0])
        out = compute_crash_betas(df, tickers=["SPY"], spy_ticker="SPY",
                                   windows=[("2020-02-20", "2020-03-23")])
        self.assertAlmostEqual(out["SPY"], 1.0, places=4)

    def test_missing_ticker_returns_nan(self):
        dates = ["2020-02-20", "2020-03-23"]
        df = _mk_prices(dates, SPY=[100.0, 70.0])
        out = compute_crash_betas(df, tickers=["NVDA"], spy_ticker="SPY",
                                   windows=[("2020-02-20", "2020-03-23")])
        self.assertTrue(np.isnan(out["NVDA"]))

    def test_skips_window_when_spy_return_is_zero(self):
        # Division by zero guard: a window where SPY didn't move shouldn't poison
        # the median. The test exercises a 2-window setup where one is degenerate.
        dates = ["2020-02-20", "2020-03-23", "2022-01-01", "2022-06-30"]
        df = _mk_prices(
            dates,
            SPY=[100.0, 75.0, 100.0, 100.0],  # window 2: 0% SPY return
            NVDA=[100.0, 50.0, 100.0, 80.0],
        )
        windows = [("2020-02-20", "2020-03-23"),
                   ("2022-01-01", "2022-06-30")]
        out = compute_crash_betas(df, tickers=["NVDA"], spy_ticker="SPY",
                                   windows=windows)
        # Only window 1 contributes; beta = 2.0
        self.assertAlmostEqual(out["NVDA"], 2.0, places=4)


if __name__ == "__main__":
    unittest.main()

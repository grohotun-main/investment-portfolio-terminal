# tests/test_crash_scenarios.py
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "parsers"))

from crash_betas import portfolio_crash_scenarios


def _crash_prices():
    idx = pd.bdate_range("2020-01-02", periods=120)
    spy = pd.Series(100.0, index=idx)
    spy.loc[idx >= "2020-02-20"] = 80.0          # -20% step inside the window
    doubled = pd.Series(100.0, index=idx)
    doubled.loc[idx >= "2020-02-20"] = 60.0      # -40% => beta 2 on the window
    return pd.DataFrame({"SPY": spy, "DBL": doubled})


WIN = [("2020-02-19", "2020-03-23")]


class TestPortfolioCrashScenarios(unittest.TestCase):
    def test_implied_drop_weighted_beta_times_spy(self):
        px = _crash_prices()
        w = pd.Series({"SPY": 0.5, "DBL": 0.5})
        out = portfolio_crash_scenarios(px, w, WIN)
        self.assertTrue(out["available"])
        sc = out["scenarios"][0]
        self.assertAlmostEqual(sc["spy_drop_pct"], -20.0, places=4)
        # betas: SPY=1, DBL=2 -> implied = (0.5*1 + 0.5*2) * -20 = -30
        self.assertAlmostEqual(sc["implied_drop_pct"], -30.0, places=4)
        self.assertEqual(sc["window"], "2020-02-19→2020-03-23")
        self.assertEqual(out["excluded_weight_pct"], 0.0)

    def test_nan_beta_excluded_and_renormalized(self):
        px = _crash_prices()
        w = pd.Series({"SPY": 0.4, "DBL": 0.4, "GHOST": 0.2})
        out = portfolio_crash_scenarios(px, w, WIN)
        sc = out["scenarios"][0]
        # GHOST (no prices) excluded; renormalized 0.5/0.5 -> -30 again
        self.assertAlmostEqual(sc["implied_drop_pct"], -30.0, places=4)
        self.assertAlmostEqual(out["excluded_weight_pct"], 20.0, places=6)
        self.assertEqual(out["n_excluded"], 1)

    def test_unusable_windows_honest(self):
        px = _crash_prices()
        w = pd.Series({"SPY": 1.0})
        out = portfolio_crash_scenarios(px, w, [("2050-01-01", "2050-02-01")])
        self.assertFalse(out["available"])

    def test_no_spy_column_everything_excluded(self):
        idx = pd.bdate_range("2020-01-02", periods=30)
        px = pd.DataFrame({"AAA": pd.Series(100.0, index=idx)})
        w = pd.Series({"AAA": 0.6, "BBB": 0.4})
        out = portfolio_crash_scenarios(px, w, WIN)
        self.assertFalse(out["available"])
        self.assertAlmostEqual(out["excluded_weight_pct"], 100.0, places=6)
        self.assertEqual(out["n_excluded"], 2)

    def test_no_negative_spy_window_honest(self):
        idx = pd.bdate_range("2020-01-02", periods=120)
        spy = pd.Series(100.0, index=idx)
        spy.loc[idx >= "2020-02-20"] = 120.0      # SPY UP in the window
        px = pd.DataFrame({"SPY": spy})
        out = portfolio_crash_scenarios(px, pd.Series({"SPY": 1.0}), WIN)
        self.assertFalse(out["available"])
        self.assertEqual(out["scenarios"], [])


if __name__ == "__main__":
    unittest.main()

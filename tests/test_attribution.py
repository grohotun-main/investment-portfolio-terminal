# tests/test_attribution.py
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "parsers"))

from attribution import position_return_contribution


def _prices():
    idx = pd.bdate_range("2025-01-02", periods=300)
    a = pd.Series(np.linspace(100, 120, len(idx)), index=idx)   # +20% total
    b = pd.Series(np.linspace(50, 45, len(idx)), index=idx)     # -10% total
    return pd.DataFrame({"AAA": a, "BBB": b})


class TestPositionReturnContribution(unittest.TestCase):
    def test_contrib_is_weight_times_window_return(self):
        px = _prices()
        w = pd.Series({"AAA": 0.6, "BBB": 0.4})
        out = position_return_contribution(px, w, {"60d": 60})
        df = out["60d"]
        for sym in ("AAA", "BBB"):
            win = px[sym].tail(60)
            r = win.iloc[-1] / win.iloc[0] - 1.0
            self.assertAlmostEqual(df.loc[sym, "return_pct"], r * 100, places=6)
            self.assertAlmostEqual(df.loc[sym, "contrib_pp"],
                                   float(w[sym]) * r * 100, places=6)
            self.assertAlmostEqual(df.loc[sym, "weight_pct"],
                                   float(w[sym]) * 100, places=6)

    def test_sorted_by_contrib_desc_and_ytd(self):
        px = _prices()
        w = pd.Series({"AAA": 0.5, "BBB": 0.5})
        out = position_return_contribution(px, w, {"ytd": "ytd"})
        df = out["ytd"]
        self.assertEqual(list(df.index), ["AAA", "BBB"])   # +contrib first
        self.assertGreater(df.loc["AAA", "contrib_pp"], 0)
        self.assertLess(df.loc["BBB", "contrib_pp"], 0)

    def test_missing_symbol_excluded_and_disclosed(self):
        px = _prices()
        w = pd.Series({"AAA": 0.5, "BBB": 0.3, "ZZZ": 0.2})
        out = position_return_contribution(px, w, {"60d": 60})
        df = out["60d"]
        self.assertNotIn("ZZZ", df.index)
        self.assertAlmostEqual(df.attrs["excluded_weight_pct"], 20.0,
                               places=6)

    def test_nan_weight_dropped_and_counted(self):
        px = _prices()
        w = pd.Series({"AAA": 0.7, "BBB": float("nan")})
        out = position_return_contribution(px, w, {"60d": 60})
        df = out["60d"]
        self.assertNotIn("BBB", df.index)
        self.assertEqual(df.attrs["n_dropped_nan_weights"], 1)
        self.assertAlmostEqual(df.attrs["excluded_weight_pct"], 0.0,
                               places=9)

    def test_empty_inputs(self):
        out = position_return_contribution(pd.DataFrame(), pd.Series(dtype=float),
                                           {"60d": 60})
        self.assertTrue(out["60d"].empty)


if __name__ == "__main__":
    unittest.main()

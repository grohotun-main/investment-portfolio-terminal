# tests/test_risk_metrics_active.py
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "parsers"))

from risk_metrics import rolling_active_stats


def _months(vals, start="2023-01-31"):
    idx = pd.date_range(start, periods=len(vals), freq="ME")
    return pd.Series(vals, index=idx)


class TestRollingActiveStats(unittest.TestCase):
    def test_always_ahead_hits_100(self):
        port = _months([0.02] * 24)
        bench = _months([0.01] * 24)
        out = rolling_active_stats(port, bench, window=12)
        self.assertTrue(out["available"])
        self.assertEqual(out["hit_rate_pct"], 100.0)
        self.assertEqual(out["n_windows"], 13)          # 24 - 12 + 1
        self.assertAlmostEqual(out["tracking_error_pct"], 0.0, places=9)
        self.assertIsNone(out["information_ratio"])     # TE == 0

    def test_mixed_hit_rate_and_ir_sign(self):
        rng = np.random.default_rng(7)
        bench = _months(rng.normal(0.008, 0.03, 36))
        port = bench + 0.002 + pd.Series(
            rng.normal(0, 0.01, 36), index=bench.index)
        out = rolling_active_stats(port, bench, window=12)
        self.assertTrue(out["available"])
        self.assertGreater(out["hit_rate_pct"], 50.0)
        self.assertGreater(out["information_ratio"], 0.0)
        self.assertGreater(out["tracking_error_pct"], 0.0)

    def test_too_short_honest(self):
        out = rolling_active_stats(_months([0.01] * 8), _months([0.0] * 8),
                                   window=12)
        self.assertFalse(out["available"])
        self.assertEqual(out["n_months"], 8)

    def test_misaligned_inner_join(self):
        port = _months([0.01] * 20)
        bench = _months([0.0] * 20, start="2023-05-31")   # 16-month overlap
        out = rolling_active_stats(port, bench, window=12)
        self.assertTrue(out["available"])
        self.assertEqual(out["n_windows"], 5)             # 16 - 12 + 1


if __name__ == "__main__":
    unittest.main()

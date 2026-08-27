# tests/test_turbulence.py
import unittest
import numpy as np
import pandas as pd
from parsers import turbulence as tb


class TurbulenceTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        idx = pd.bdate_range("2020-01-01", periods=400)
        calm = rng.normal(0, 0.01, size=(399, 3))
        self.rets = pd.DataFrame(calm, index=idx[1:], columns=["SPY", "GLD", "TLT"])

    def test_index_length_and_finiteness(self):
        t = tb.turbulence_index(self.rets)
        self.assertEqual(len(t), len(self.rets))
        self.assertTrue(np.isfinite(t.to_numpy()).all())

    def test_spike_row_is_high_percentile(self):
        spiked = self.rets.copy()
        spiked.iloc[-1] = [0.15, -0.15, 0.15]      # joint extreme move
        out = tb.turbulence_now(spiked)
        self.assertGreaterEqual(out["percentile"], 90.0)
        self.assertEqual(out["regime"], "abnormal")


class VolRegimeTests(unittest.TestCase):
    def test_high_vol_tail_is_stressed(self):
        idx = pd.bdate_range("2020-01-01", periods=300)
        calm = np.full(250, 100.0) + np.linspace(0, 1, 250)
        # last 50 days: violent swings -> high realized vol
        rng = np.random.default_rng(2)
        wild = 100.0 * (1 + rng.normal(0, 0.05, size=50)).cumprod()
        price = pd.Series(np.concatenate([calm, wild]), index=idx)
        reg = tb.vol_regime(price, window=21, q=0.80)
        self.assertEqual(reg.iloc[-1], "stressed")
        self.assertEqual(reg.iloc[100], "calm")

    def test_flat_price_all_calm(self):
        """Fix 2: constant price → zero vol → guard returns all-calm."""
        idx = pd.bdate_range("2020-01-01", periods=100)
        price = pd.Series(np.full(100, 50.0), index=idx)
        reg = tb.vol_regime(price, window=21, q=0.80)
        self.assertTrue((reg == "calm").all(), f"expected all calm, got: {reg.value_counts().to_dict()}")

    def test_vol_regime_leading_boundary(self):
        """Fix 1: for window=21, first 21 entries (iloc[:21]) must all be calm."""
        rng = np.random.default_rng(3)
        idx = pd.bdate_range("2020-01-01", periods=200)
        # mix of calm + stressed so thr > 0 (non-degenerate data)
        base = np.full(150, 100.0) + np.linspace(0, 1, 150)
        wild = 100.0 * (1 + rng.normal(0, 0.05, size=50)).cumprod()
        price = pd.Series(np.concatenate([base, wild]), index=idx)
        reg = tb.vol_regime(price, window=21, q=0.80)
        leading = reg.iloc[:21]
        self.assertTrue(
            (leading == "calm").all(),
            f"expected first 21 entries calm, got: {leading.value_counts().to_dict()}",
        )
        self.assertEqual(reg.iloc[20], "calm")


class TurbulenceEdgeTests(unittest.TestCase):
    def test_empty_frame_returns_unknown(self):
        """Fix 3 / existing guard: empty DataFrame → regime == 'unknown'."""
        out = tb.turbulence_now(pd.DataFrame())
        self.assertEqual(out["regime"], "unknown")

    def test_one_row_frame_does_not_raise(self):
        """Fix 4: 1-row frame must return an empty Series without LinAlgError."""
        df = pd.DataFrame({"A": [0.01], "B": [-0.01]}, index=pd.bdate_range("2020-01-01", periods=1))
        result = tb.turbulence_index(df)
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 0)

    def test_constant_return_frame_not_abnormal(self):
        """Fix 3: perfectly constant cross-asset returns → regime != 'abnormal' (should be 'calm')."""
        idx = pd.bdate_range("2020-01-01", periods=50)
        df = pd.DataFrame(
            np.tile([0.001, -0.001, 0.0], (50, 1)),
            index=idx,
            columns=["A", "B", "C"],
        )
        out = tb.turbulence_now(df)
        self.assertNotEqual(out["regime"], "abnormal",
                            f"constant-return frame should not be 'abnormal', got {out}")


class VolRegimeConstantsTests(unittest.TestCase):
    def test_constants_are_the_defaults(self):
        idx = pd.bdate_range("2020-01-01", periods=120)
        rng = np.random.default_rng(7)
        price = pd.Series(100.0 * (1 + rng.normal(0, 0.02, 120)).cumprod(), index=idx)
        self.assertEqual(tb.DIP_VOL_REGIME_WINDOW, 21)
        self.assertAlmostEqual(tb.DIP_VOL_REGIME_Q, 0.80)
        # vol_regime() with no overrides must equal an explicit call with the constants
        pd.testing.assert_series_equal(
            tb.vol_regime(price),
            tb.vol_regime(price, window=tb.DIP_VOL_REGIME_WINDOW, q=tb.DIP_VOL_REGIME_Q),
        )

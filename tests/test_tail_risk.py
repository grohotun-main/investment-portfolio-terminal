# tests/test_tail_risk.py
import unittest
import numpy as np
from parsers import tail_risk as tr


class GpdTailTests(unittest.TestCase):
    def setUp(self):
        # exponential-ish losses: GPD with xi ~ 0 should fit
        rng = np.random.default_rng(0)
        self.losses = rng.exponential(scale=0.05, size=500)

    def test_fit_returns_params_and_confidence(self):
        fit = tr.fit_gpd_tail(self.losses, threshold_q=0.90)
        self.assertIn("xi", fit)
        self.assertIn("beta", fit)
        self.assertGreater(fit["n_exceedances"], 0)
        self.assertTrue(fit["confident"])           # 500*0.10 = 50 exceedances

    def test_low_sample_is_not_confident(self):
        fit = tr.fit_gpd_tail(self.losses[:40], threshold_q=0.90)
        self.assertFalse(fit["confident"])          # 40*0.10 = 4 exceedances

    def test_tail_quantile_more_extreme_than_threshold(self):
        fit = tr.fit_gpd_tail(self.losses, threshold_q=0.90)
        q99 = tr.tail_loss_quantile(fit, p=0.01)
        self.assertGreater(q99, fit["threshold"])   # 1% loss worse than the 90% threshold

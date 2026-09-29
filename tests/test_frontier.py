# tests/test_frontier.py
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "parsers"))

from frontier import ASSUMED_BETA, capm_expected_returns, LAMBDA_LADDER, trace_frontier


def _ff_daily(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """Daily Fama-French frame in the ff_factors_daily.csv schema."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "mkt_rf": rng.normal(0.0004, 0.01, n),
        "smb": rng.normal(0.0, 0.004, n),
        "hml": rng.normal(0.0, 0.004, n),
        "rmw": rng.normal(0.0, 0.003, n),
        "cma": rng.normal(0.0, 0.003, n),
        "mom": rng.normal(0.0, 0.004, n),
        "rf": np.full(n, 0.00008),
    })


def _prices_with_betas(ff: pd.DataFrame, betas: dict[str, float],
                       seed: int = 1) -> pd.DataFrame:
    """Wide price frame whose returns are rf + beta*mkt_rf + small noise."""
    rng = np.random.default_rng(seed)
    idx = pd.to_datetime(ff["date"])
    out = {}
    for sym, b in betas.items():
        r = (ff["rf"].to_numpy() + b * ff["mkt_rf"].to_numpy()
             + rng.normal(0.0, 0.0004, len(ff)))
        out[sym] = 100.0 * np.cumprod(1.0 + r)
    return pd.DataFrame(out, index=idx)


class TestCapmExpectedReturns(unittest.TestCase):
    def setUp(self):
        self.ff = _ff_daily()
        self.px = _prices_with_betas(self.ff, {"HIGH": 1.5, "LOW": 0.5})

    def test_mu_is_rf_plus_beta_times_erp(self):
        res = capm_expected_returns(self.px, self.ff, ["HIGH", "LOW"],
                                    rf_annual=0.04, erp=0.05)
        self.assertIsNone(res["error"])
        self.assertEqual(res["assumed"], [])
        self.assertAlmostEqual(res["betas"]["HIGH"], 1.5, delta=0.15)
        self.assertAlmostEqual(res["betas"]["LOW"], 0.5, delta=0.15)
        for sym in ("HIGH", "LOW"):
            self.assertAlmostEqual(res["mu"][sym],
                                   0.04 + res["betas"][sym] * 0.05, places=12)

    def test_erp_scales_the_premium_linearly(self):
        a = capm_expected_returns(self.px, self.ff, ["HIGH"],
                                  rf_annual=0.04, erp=0.05)
        b = capm_expected_returns(self.px, self.ff, ["HIGH"],
                                  rf_annual=0.04, erp=0.10)
        self.assertAlmostEqual(b["mu"]["HIGH"] - 0.04,
                               2.0 * (a["mu"]["HIGH"] - 0.04), places=12)

    def test_unpriced_symbol_assumes_beta_one(self):
        res = capm_expected_returns(self.px, self.ff, ["HIGH", "GHOST"],
                                    rf_annual=0.04, erp=0.05)
        self.assertIsNone(res["error"])
        self.assertEqual(res["assumed"], ["GHOST"])
        self.assertEqual(res["betas"]["GHOST"], ASSUMED_BETA)
        self.assertAlmostEqual(res["mu"]["GHOST"], 0.04 + 0.05, places=12)

    def test_priced_but_too_thin_assumes_beta_one(self):
        px = self.px.copy()
        px["THIN"] = np.nan
        px.iloc[-60:, px.columns.get_loc("THIN")] = 100.0 * np.cumprod(
            1.0 + np.full(60, 0.001))
        res = capm_expected_returns(px, self.ff, ["HIGH", "THIN"],
                                    rf_annual=0.04, erp=0.05)
        self.assertEqual(res["assumed"], ["THIN"])
        self.assertEqual(res["betas"]["THIN"], ASSUMED_BETA)

    def test_missing_factors_is_error_not_raise(self):
        res = capm_expected_returns(self.px, pd.DataFrame(), ["HIGH"],
                                    rf_annual=0.04, erp=0.05)
        self.assertIsNotNone(res["error"])
        self.assertIn("fetch_ff_factors", res["error"])
        self.assertTrue(res["mu"].empty)

    def test_no_symbols_is_error_not_raise(self):
        res = capm_expected_returns(self.px, self.ff, [],
                                    rf_annual=0.04, erp=0.05)
        self.assertIsNotNone(res["error"])

    def test_mu_index_order_follows_symbols(self):
        res = capm_expected_returns(self.px, self.ff, ["LOW", "GHOST", "HIGH"],
                                    rf_annual=0.04, erp=0.05)
        self.assertEqual(list(res["mu"].index), ["LOW", "GHOST", "HIGH"])


def _book(n_days=400, seed=5):
    """Mature 3-name book + one young name, with a plausible mu."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2022-01-03", periods=n_days)
    cols = {}
    for sym, drift, vol in (("SPY", 0.0004, 0.010), ("AAA", 0.0005, 0.013),
                            ("BBB", 0.0002, 0.006)):
        cols[sym] = 100.0 * np.cumprod(1.0 + rng.normal(drift, vol, n_days))
    px = pd.DataFrame(cols, index=idx)
    young = np.full(n_days, np.nan)
    young[-60:] = 50.0 * np.cumprod(1.0 + rng.normal(0.0004, 0.01, 60))
    px["YOUNG"] = young
    weights = pd.Series({"SPY": 0.40, "AAA": 0.25, "BBB": 0.25, "YOUNG": 0.10})
    class_of = {"SPY": "equity", "AAA": "equity", "BBB": "fixed_income",
                "YOUNG": "equity"}
    mu = pd.Series({"SPY": 0.085, "AAA": 0.110, "BBB": 0.045, "YOUNG": 0.090})
    return px, weights, class_of, mu


class TestTraceFrontier(unittest.TestCase):
    def setUp(self):
        self.px, self.weights, self.class_of, self.mu = _book()

    def _trace(self, **kw):
        kw.setdefault("name_cap", 0.60)
        kw.setdefault("class_floors", {})
        return trace_frontier(self.px, self.weights, self.class_of, self.mu,
                              **kw)

    def test_shape_and_point_count(self):
        out = self._trace()
        self.assertIsNone(out["error"])
        self.assertEqual(list(out["points"].columns),
                         ["lam", "vol", "exp_return", "effective_n",
                          "max_weight", "converged", "weights"])
        self.assertEqual(len(out["points"]), len(LAMBDA_LADDER))
        self.assertEqual(out["skipped"], [])

    def test_points_run_from_min_variance_to_max_return(self):
        pts = self._trace()["points"]
        vols = pts["vol"].to_numpy(dtype=float)
        rets = pts["exp_return"].to_numpy(dtype=float)
        self.assertTrue(np.all(np.isfinite(vols)))
        self.assertTrue(np.all(np.isfinite(rets)))
        self.assertTrue(np.all(np.diff(vols) >= -1e-9),
                        f"vol not monotone along the ladder: {vols}")
        self.assertTrue(np.all(np.diff(rets) >= -1e-9),
                        f"return not monotone along the ladder: {rets}")

    def test_first_point_matches_the_min_variance_suggestion(self):
        from min_variance import suggest_min_variance_grid
        out = self._trace()
        mv_out = suggest_min_variance_grid(
            self.px, self.weights, self.class_of, name_cap=0.60,
            class_floors={})
        first = out["points"].iloc[0]
        # The ladder's top lambda is finite (~316 * mature_budget), so the mu-
        # scalarized solve at that point is a close approximation of the
        # mu=None min-variance corner, not bit-identical to it -- Task 2's own
        # convergence test (test_high_risk_aversion_converges_to_min_variance)
        # needed risk_aversion=1e5 to call two solves "converged", so places=6
        # (< 5e-7) on vol is tighter than the fixed LAMBDA_LADDER can deliver.
        # Observed gap on this book is ~3.3e-6; delta=1e-5 keeps a real margin
        # while still catching a wrong-direction/wrong-scale regression.
        self.assertAlmostEqual(float(first["vol"]), float(mv_out["vol"]),
                               delta=1e-5)

    def test_markers_are_priced_through_mu(self):
        out = self._trace()
        keys = {m["key"] for m in out["markers"]}
        self.assertEqual(keys, {"min_variance", "risk_parity"})
        for m in out["markers"]:
            self.assertTrue(np.isfinite(m["vol"]))
            self.assertTrue(np.isfinite(m["exp_return"]))
            self.assertGreater(m["exp_return"], 0.0)

    def test_current_book_return_includes_the_young_holding(self):
        out = self._trace()
        expected = float(sum(self.weights[s] * self.mu[s]
                             for s in self.weights.index))
        self.assertAlmostEqual(out["current"]["exp_return"], expected,
                               places=12)

    def test_infeasible_cap_skips_every_point_without_erroring(self):
        out = self._trace(name_cap=0.10)      # 0.10 * 4 names < 1
        self.assertIsNone(out["error"])
        self.assertTrue(out["points"].empty)
        self.assertEqual(len(out["skipped"]), len(LAMBDA_LADDER))
        self.assertIsInstance(out["skipped"][0][2], str)

    def test_unbuildable_covariance_is_error(self):
        out = trace_frontier(pd.DataFrame(), self.weights, self.class_of,
                             self.mu, name_cap=0.60, class_floors={})
        self.assertIsNotNone(out["error"])
        self.assertTrue(out["points"].empty)

    def test_missing_mu_entry_is_error_not_raise(self):
        mu = self.mu.drop("BBB")
        out = trace_frontier(self.px, self.weights, self.class_of, mu,
                             name_cap=0.60, class_floors={})
        self.assertIsNotNone(out["error"])

    def test_on_point_progress_hook_fires_per_solve(self):
        seen = []
        self._trace(on_point=lambda done, total: seen.append((done, total)))
        self.assertEqual(len(seen), len(LAMBDA_LADDER))
        self.assertEqual(seen[-1], (len(LAMBDA_LADDER), len(LAMBDA_LADDER)))

    def test_a_class_floor_costs_expected_return(self):
        """The floor forces weight into the low-mu sleeve, so the reachable
        maximum return must drop. (Written before point rows carried per-name
        weights; the max_weight/exp_return proxies still observe the constraint
        directly.)"""
        free = self._trace()["points"]
        floored = self._trace(class_floors={"fixed_income": 0.30})["points"]
        self.assertFalse(floored.empty)
        self.assertLess(float(floored["exp_return"].max()),
                        float(free["exp_return"].max()))
        self.assertTrue(all(float(r["max_weight"]) <= 0.60 + 1e-6
                            for _, r in floored.iterrows()))

    def test_points_carry_full_book_weights(self):
        pts = self._trace()["points"]
        self.assertIn("weights", pts.columns)
        self.assertFalse(pts.empty)
        for _, r in pts.iterrows():
            w = r["weights"]
            self.assertIsInstance(w, dict)
            self.assertTrue(np.isclose(sum(w.values()), 100.0, atol=1e-6), w)


class TestConcentrationShared(unittest.TestCase):
    def test_concentration_is_opt_curve_single_copy(self) -> None:
        # O4a dedup pin: frontier imports opt_curve._concentration — a
        # re-forked local copy would drift (the pre-O4a state).
        import frontier as fr
        import opt_curve as oc
        self.assertIs(fr._concentration, oc._concentration)


if __name__ == "__main__":
    unittest.main()

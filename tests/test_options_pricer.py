"""Tests for parsers/options_pricer.py.

The pricer is the math engine for hedging analytics. Bugs here propagate
silently into hedge-sizing and would only be caught by Polygon-comparison or
by losing real money. So this suite is heavier than the other parsers':

  * Closed-form Hull textbook values lock in BS price + Greeks.
  * Put-call parity (must hold to ~1e-12) catches sign / formula errors.
  * Finite-difference Greeks vs analytic Greeks catch wrong derivatives.
  * American put >= European put + parity-violation checks catch tree bugs.
  * Implied-vol round-trip across many parameter regimes catches solver bugs.

All tests are pure-Python; no network, no scipy.
"""
import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from options_pricer import (  # noqa: E402
    DEFAULT_N_STEPS,
    binomial_american,
    black_scholes,
    implied_vol,
    price_and_greeks,
)


# ---------- Black-Scholes closed-form ------------------------------------

class TestBlackScholesClosedForm(unittest.TestCase):
    """Hull-textbook values + classic analytic checks. Tolerance 1e-3 on
    price, 1e-4 on Greeks (the formulas are exact; tolerance accommodates
    numerical noise in the erf-based normal CDF)."""

    def test_atm_call_hull_table(self) -> None:
        # Hull "Options, Futures..." standard textbook example:
        # S=100, K=100, T=1, r=5%, q=0, σ=20%  ->  call price = 10.4506
        r = black_scholes(100.0, 100.0, 1.0, 0.05, 0.0, 0.20, "call")
        self.assertAlmostEqual(r["price"], 10.4506, places=3)

    def test_atm_put_hull_table(self) -> None:
        # Same params, put price = 5.5735
        r = black_scholes(100.0, 100.0, 1.0, 0.05, 0.0, 0.20, "put")
        self.assertAlmostEqual(r["price"], 5.5735, places=3)

    def test_otm_call_with_dividend_regression(self) -> None:
        # Regression reference for the dividend-yield path. The ATM Hull
        # test above validates the formula end-to-end against a published
        # source; this locks the q-term behavior. If a refactor accidentally
        # drops e^(-qT) somewhere, this fires.
        r = black_scholes(100.0, 110.0, 0.5, 0.04, 0.02, 0.25, "call")
        self.assertAlmostEqual(r["price"], 3.7044, places=3)

    def test_dividend_decreases_call_price(self) -> None:
        # Qualitative: raising q (more dividends paid out) reduces a call's
        # value because more wealth leaks out of the stock pre-expiry.
        low_q  = black_scholes(100, 100, 1.0, 0.05, 0.00, 0.20, "call")["price"]
        high_q = black_scholes(100, 100, 1.0, 0.05, 0.04, 0.20, "call")["price"]
        self.assertGreater(low_q, high_q)

    def test_call_delta_in_zero_to_one(self) -> None:
        for K in (80, 100, 120):
            d = black_scholes(100, K, 1.0, 0.05, 0.0, 0.30, "call")["delta"]
            self.assertGreaterEqual(d, 0.0)
            self.assertLessEqual(d, 1.0)

    def test_put_delta_in_minus_one_to_zero(self) -> None:
        for K in (80, 100, 120):
            d = black_scholes(100, K, 1.0, 0.05, 0.0, 0.30, "put")["delta"]
            self.assertGreaterEqual(d, -1.0)
            self.assertLessEqual(d, 0.0)

    def test_gamma_strictly_positive(self) -> None:
        # Convexity: gamma > 0 for any non-degenerate option
        for opt in ("call", "put"):
            g = black_scholes(100, 100, 0.5, 0.05, 0.0, 0.25, opt)["gamma"]
            self.assertGreater(g, 0.0)

    def test_vega_strictly_positive(self) -> None:
        for opt in ("call", "put"):
            v = black_scholes(100, 100, 0.5, 0.05, 0.0, 0.25, opt)["vega"]
            self.assertGreater(v, 0.0)

    def test_call_theta_negative_no_dividend(self) -> None:
        # With q=0, a long call always has theta < 0
        t = black_scholes(100, 100, 1.0, 0.05, 0.0, 0.20, "call")["theta"]
        self.assertLess(t, 0.0)


# ---------- Put-call parity ----------------------------------------------

class TestPutCallParity(unittest.TestCase):
    """C - P = S*e^(-qT) - K*e^(-rT) must hold for European options."""

    def _check(self, S, K, T, r, q, sigma):
        c = black_scholes(S, K, T, r, q, sigma, "call")["price"]
        p = black_scholes(S, K, T, r, q, sigma, "put")["price"]
        lhs = c - p
        rhs = S * math.exp(-q * T) - K * math.exp(-r * T)
        self.assertAlmostEqual(lhs, rhs, places=10)

    def test_parity_atm_no_div(self):
        self._check(100, 100, 1.0, 0.05, 0.0, 0.20)

    def test_parity_otm_with_div(self):
        self._check(95, 110, 0.5, 0.04, 0.02, 0.25)

    def test_parity_deep_itm_call(self):
        self._check(150, 100, 2.0, 0.03, 0.015, 0.35)

    def test_parity_high_vol(self):
        self._check(100, 100, 1.0, 0.05, 0.0, 1.50)  # 150% vol — single names


# ---------- Greeks vs finite difference ---------------------------------

class TestGreeksFiniteDifference(unittest.TestCase):
    """Bump-and-revalue the price; numerical derivative must match analytic
    Greek to ~1e-3. Catches sign errors and misplaced terms in the formulas."""

    BASE = dict(S=100.0, K=100.0, T=0.5, r=0.04, q=0.02, sigma=0.25, opt="call")

    def _bs(self, **overrides):
        p = {**self.BASE, **overrides}
        return black_scholes(p["S"], p["K"], p["T"], p["r"], p["q"], p["sigma"], p["opt"])

    def test_delta_matches_finite_diff(self):
        eps = 0.01
        for opt in ("call", "put"):
            analytic = self._bs(opt=opt)["delta"]
            up   = self._bs(opt=opt, S=self.BASE["S"] + eps)["price"]
            down = self._bs(opt=opt, S=self.BASE["S"] - eps)["price"]
            numerical = (up - down) / (2 * eps)
            self.assertAlmostEqual(analytic, numerical, places=3,
                                   msg=f"opt={opt}")

    def test_gamma_matches_finite_diff(self):
        eps = 0.05
        for opt in ("call", "put"):
            analytic = self._bs(opt=opt)["gamma"]
            up   = self._bs(opt=opt, S=self.BASE["S"] + eps)["price"]
            mid  = self._bs(opt=opt)["price"]
            down = self._bs(opt=opt, S=self.BASE["S"] - eps)["price"]
            numerical = (up - 2 * mid + down) / (eps * eps)
            self.assertAlmostEqual(analytic, numerical, places=3,
                                   msg=f"opt={opt}")

    def test_vega_matches_finite_diff(self):
        eps = 1e-4
        for opt in ("call", "put"):
            analytic = self._bs(opt=opt)["vega"]
            up   = self._bs(opt=opt, sigma=self.BASE["sigma"] + eps)["price"]
            down = self._bs(opt=opt, sigma=self.BASE["sigma"] - eps)["price"]
            numerical = (up - down) / (2 * eps)
            # Vega is large in absolute terms (~30s) — use places=2
            self.assertAlmostEqual(analytic, numerical, places=2,
                                   msg=f"opt={opt}")

    def test_rho_matches_finite_diff(self):
        eps = 1e-5
        for opt in ("call", "put"):
            analytic = self._bs(opt=opt)["rho"]
            up   = self._bs(opt=opt, r=self.BASE["r"] + eps)["price"]
            down = self._bs(opt=opt, r=self.BASE["r"] - eps)["price"]
            numerical = (up - down) / (2 * eps)
            self.assertAlmostEqual(analytic, numerical, places=2,
                                   msg=f"opt={opt}")


# ---------- Binomial-tree (American) ------------------------------------

class TestBinomialAmerican(unittest.TestCase):
    """American option behaviors that distinguish it from European."""

    def test_american_put_at_least_european(self):
        # Always: an American option dominates the European equivalent.
        # The early-exercise premium is strictly positive when interest rates
        # make early exercise potentially valuable (puts) or when dividends
        # make it valuable (calls).
        for K in (90, 100, 110):
            ep = black_scholes(100, K, 1.0, 0.05, 0.0, 0.30, "put")["price"]
            ap = binomial_american(100, K, 1.0, 0.05, 0.0, 0.30, "put")["price"]
            self.assertGreaterEqual(ap + 1e-4, ep, msg=f"K={K}")

    def test_american_call_no_div_equals_european(self):
        # Classical result: an American CALL on a non-dividend-paying stock is
        # never optimally exercised early, so its price equals the European.
        ec = black_scholes(100, 100, 1.0, 0.05, 0.0, 0.30, "call")["price"]
        ac = binomial_american(100, 100, 1.0, 0.05, 0.0, 0.30, "call")["price"]
        self.assertAlmostEqual(ac, ec, places=1)  # tree discretization noise

    def test_american_itm_put_has_positive_early_exercise_premium(self):
        # ITM put + high rate -> meaningful early-exercise premium
        ep = black_scholes(85, 100, 1.0, 0.08, 0.0, 0.25, "put")["price"]
        ap = binomial_american(85, 100, 1.0, 0.08, 0.0, 0.25, "put")["price"]
        self.assertGreater(ap - ep, 0.05,
                           msg="early-exercise premium should be nontrivial here")

    def test_american_delta_in_range(self):
        d = binomial_american(100, 100, 0.5, 0.05, 0.0, 0.25, "put")["delta"]
        self.assertGreaterEqual(d, -1.0)
        self.assertLessEqual(d, 0.0)

    def test_american_gamma_positive(self):
        g = binomial_american(100, 100, 0.5, 0.05, 0.0, 0.25, "put")["gamma"]
        self.assertGreater(g, 0.0)


# ---------- Leisen-Reimer specific behaviors ----------------------------

class TestLeisenReimer(unittest.TestCase):
    """LR is the new default method for binomial_american. Tests pin down
    the properties that motivated the swap: Greek convergence is fast and
    stable (no CRR-style oscillation), and the LR American premium agrees
    with European BS to within numerical noise for OTM options (where
    early-exercise value is near zero).
    """

    def test_vega_converges_in_n_steps(self):
        # OTM put, 89 DTE — the moneyness band where CRR oscillates
        # between 0.62 and 0.70 (per 1%) across N=200/500/1000 on the
        # same contract (verified in debug session 2026-05-24). LR must
        # agree to ~3 decimals (per 1%) across N=101..1001.
        S, K, T, r, q, sigma = 745.64, 655.0, 89/365, 0.0368, 0.0067, 0.243
        vegas_per_pct = []
        for N in (101, 201, 501, 1001):
            res = binomial_american(S, K, T, r, q, sigma, "put",
                                    n_steps=N, method="lr")
            vegas_per_pct.append(res["vega"] * 0.01)
        spread = max(vegas_per_pct) - min(vegas_per_pct)
        self.assertLess(spread, 0.001,
                        msg=f"LR vega/pct not converging: {vegas_per_pct}")

    def test_otm_put_american_premium_near_zero(self):
        # OTM puts have negligible early-exercise value (you wouldn't
        # exercise a put with no intrinsic). Our American LR vega should
        # match European BS vega tightly when measured in display units.
        S, K, T, r, q, sigma = 745.64, 655.0, 89/365, 0.0368, 0.0067, 0.243
        bs = black_scholes(S, K, T, r, q, sigma, "put")
        am = binomial_american(S, K, T, r, q, sigma, "put", method="lr")
        # Compare in "per 1%" units (the units a user actually sees)
        self.assertAlmostEqual(am["vega"] * 0.01, bs["vega"] * 0.01,
            places=2,
            msg="LR American vega should match BS European vega for OTM put")
        self.assertAlmostEqual(am["delta"], bs["delta"], places=2)

    def test_itm_put_american_premium_positive(self):
        # Deep ITM put + positive rate: American > European is meaningful.
        S, K, T, r, q, sigma = 85.0, 100.0, 1.0, 0.08, 0.0, 0.25
        ep = black_scholes(S, K, T, r, q, sigma, "put")["price"]
        ap = binomial_american(S, K, T, r, q, sigma, "put", method="lr")["price"]
        self.assertGreater(ap - ep, 0.05,
                           msg="Early-exercise premium should be nontrivial")

    def test_crr_method_still_available(self):
        # Backwards compatibility: method="crr" must still work.
        r = binomial_american(100, 100, 0.5, 0.05, 0.0, 0.25, "put",
                              method="crr")
        self.assertIn("vega", r)
        self.assertGreater(r["vega"], 0)
        self.assertLess(r["delta"], 0)

    def test_lr_and_crr_agree_at_atm(self):
        # ATM is where CRR works well — LR and CRR should give close answers.
        S, K, T, r, q, sigma = 100.0, 100.0, 0.5, 0.04, 0.02, 0.25
        lr  = binomial_american(S, K, T, r, q, sigma, "put", method="lr",
                                n_steps=501)
        crr = binomial_american(S, K, T, r, q, sigma, "put", method="crr",
                                n_steps=501)
        self.assertAlmostEqual(lr["price"], crr["price"], places=2)
        self.assertAlmostEqual(lr["delta"], crr["delta"], places=2)
        # Vega tolerance looser — CRR oscillates even at N=501
        self.assertAlmostEqual(lr["vega"], crr["vega"], places=0)


# ---------- Implied volatility solver -----------------------------------

class TestImpliedVol(unittest.TestCase):
    """Round-trip: price an option with a known σ, back σ out from that price,
    should recover σ to within solver tolerance."""

    def _roundtrip(self, S, K, T, r, q, true_sigma, opt, exercise="european"):
        priced = price_and_greeks(S, K, T, r, q, true_sigma, opt, exercise)
        iv = implied_vol(priced["price"], S, K, T, r, q, opt, exercise)
        self.assertAlmostEqual(iv, true_sigma, places=4,
            msg=f"S={S} K={K} T={T} σ={true_sigma} opt={opt} ex={exercise}")

    def test_european_atm_put_roundtrip(self):
        self._roundtrip(100, 100, 0.5, 0.04, 0.01, 0.25, "put")

    def test_european_otm_call_roundtrip(self):
        self._roundtrip(100, 120, 1.0, 0.05, 0.0, 0.30, "call")

    def test_european_deep_itm_put_roundtrip(self):
        self._roundtrip(80, 110, 0.5, 0.04, 0.0, 0.20, "put")

    def test_european_low_vol_roundtrip(self):
        self._roundtrip(100, 100, 0.25, 0.03, 0.0, 0.08, "put")

    def test_european_high_vol_roundtrip(self):
        # Single-name IVs can be 80-150% — solver must handle the upper range.
        self._roundtrip(100, 100, 0.5, 0.04, 0.0, 1.20, "put")

    def test_american_atm_put_roundtrip(self):
        self._roundtrip(100, 100, 0.5, 0.04, 0.02, 0.25, "put", exercise="american")

    def test_below_intrinsic_returns_nan(self):
        # Market price below intrinsic forward = arbitrage, solver should bail.
        # Forward intrinsic for a put with S=100, K=110, T=0.5, r=0.04, q=0:
        #   K*e^(-rT) - S*e^(-qT) = 110*0.9802 - 100 = 7.82
        iv = implied_vol(2.0, 100, 110, 0.5, 0.04, 0.0, "put", "european")
        self.assertTrue(math.isnan(iv))


# ---------- Edge cases --------------------------------------------------

class TestEdgeCases(unittest.TestCase):
    """Degenerate inputs: must not crash, must return sane values."""

    def test_expiry_call_intrinsic(self):
        r = black_scholes(105, 100, 0.0, 0.05, 0.0, 0.25, "call")
        self.assertEqual(r["price"], 5.0)
        self.assertEqual(r["delta"], 1.0)
        self.assertEqual(r["gamma"], 0.0)
        self.assertEqual(r["vega"], 0.0)

    def test_expiry_put_intrinsic(self):
        r = black_scholes(95, 100, 0.0, 0.05, 0.0, 0.25, "put")
        self.assertEqual(r["price"], 5.0)
        self.assertEqual(r["delta"], -1.0)

    def test_zero_vol_returns_discounted_intrinsic(self):
        # No randomness -> price = max(0, forward - K*exp(-rT))
        r = black_scholes(100, 90, 1.0, 0.05, 0.0, 0.0, "call")
        expected = 100 - 90 * math.exp(-0.05 * 1.0)
        self.assertAlmostEqual(r["price"], expected, places=6)

    def test_dispatcher_european_matches_bs(self):
        bs   = black_scholes(100, 100, 1.0, 0.05, 0.0, 0.20, "call")
        disp = price_and_greeks(100, 100, 1.0, 0.05, 0.0, 0.20, "call", "european")
        self.assertEqual(bs, disp)

    def test_dispatcher_rejects_unknown_exercise(self):
        with self.assertRaises(ValueError):
            price_and_greeks(100, 100, 1.0, 0.05, 0.0, 0.20, "call", "bermudan")

    def test_unknown_opt_type_raises(self):
        with self.assertRaises(ValueError):
            black_scholes(100, 100, 1.0, 0.05, 0.0, 0.20, "straddle")


# ---------- Polygon real-world sanity (kept tight) ----------------------

class TestPolygonRealWorldSpotCheck(unittest.TestCase):
    """Single contract from the Step 0 probe: SPY 25-DTE ATM put.
    Polygon's reported Greeks should agree with ours within ~5% (most
    are within 1-2% — see Step 1 smoke test). This is a sanity check
    that the pricer doesn't drift over future refactors; the full
    chain verification lives in parsers/verify_options_pricer.py."""

    # From _probe_options.py output, 2026-05-24 SPY ATM put,
    # K=746, expiry 2026-06-18, IV reported by Polygon = 0.14694.
    # Inputs: S=745.64, T=25/365, r≈0.043, q≈0.013, opt='put', american.
    S, K, T, r, q, sigma = 745.64, 746.0, 25 / 365, 0.043, 0.013, 0.14694

    def test_delta_within_5pct(self):
        d = binomial_american(self.S, self.K, self.T, self.r, self.q,
                              self.sigma, "put")["delta"]
        polygon = -0.4850
        rel_err = abs(d - polygon) / abs(polygon)
        self.assertLess(rel_err, 0.05, f"delta {d} vs polygon {polygon}")

    def test_gamma_within_5pct(self):
        g = binomial_american(self.S, self.K, self.T, self.r, self.q,
                              self.sigma, "put")["gamma"]
        polygon = 0.014446
        rel_err = abs(g - polygon) / abs(polygon)
        self.assertLess(rel_err, 0.05, f"gamma {g} vs polygon {polygon}")

    def test_vega_within_5pct(self):
        v = binomial_american(self.S, self.K, self.T, self.r, self.q,
                              self.sigma, "put")["vega"]
        # Polygon vega is per 1% — convert ours (per 1.0) by *0.01
        polygon_per_pct = 0.7707
        ours_per_pct = v * 0.01
        rel_err = abs(ours_per_pct - polygon_per_pct) / abs(polygon_per_pct)
        self.assertLess(rel_err, 0.05, f"vega/pct {ours_per_pct} vs polygon {polygon_per_pct}")


if __name__ == "__main__":
    unittest.main()

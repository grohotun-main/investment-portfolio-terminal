"""Tests for parsers/stress_hedge.py.

The stress-hedge calculator is consumed directly by buy-decisions on real
money, so this suite locks in:

  * Accounting identities (upfront cost, notional, cost-of-insurance) —
    closed-form math, no pricer involvement.
  * Breakeven definition (at-expiry intrinsic = premium paid).
  * Scenario application: a no-op scenario reprices ≈ base; sign and
    monotonicity of payoff for puts and calls under spot/vol shocks.
  * Override hook: passing a custom `scenarios=` sequence works.
  * Linearity in contract count.
  * Edge guard: ValueError on expired contracts.

Pure-Python, no network, no scipy.
"""
import math
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from options_pricer import binomial_american  # noqa: E402
from stress_hedge import (  # noqa: E402
    CONTRACT_MULT,
    DEFAULT_SCENARIOS,
    Hedge,
    Scenario,
    evaluate_hedge,
)


def _spy_put_hedge(premium: float = 5.00, n: int = 1,
                   expiry_days: int = 90) -> tuple[Hedge, date]:
    """A representative SPY 5% OTM put hedge used across several tests."""
    today = date(2026, 5, 24)
    return (
        Hedge(
            ticker="SPY",
            option_type="put",
            strike=475.0,
            expiry=today + timedelta(days=expiry_days),
            n_contracts=n,
            premium_per_share=premium,
        ),
        today,
    )


class TestAccountingIdentities(unittest.TestCase):
    """Cost, notional, and cost-of-insurance follow from pure arithmetic
    — no pricer involvement. These should never drift."""

    def test_upfront_cost(self) -> None:
        h, today = _spy_put_hedge(premium=5.00, n=10)
        ev = evaluate_hedge(h, spot_today=500.0, sigma_today=0.18,
                            r=0.045, q=0.013, today=today)
        # 10 contracts * 100 shares * $5.00 = $5,000
        self.assertAlmostEqual(ev.upfront_cost, 5_000.0, places=6)

    def test_notional_protected(self) -> None:
        h, today = _spy_put_hedge(premium=5.00, n=10)
        ev = evaluate_hedge(h, spot_today=500.0, sigma_today=0.18,
                            r=0.045, q=0.013, today=today)
        # 10 * 100 * $500 = $500,000
        self.assertAlmostEqual(ev.notional_protected, 500_000.0, places=6)

    def test_cost_of_insurance_pct(self) -> None:
        h, today = _spy_put_hedge(premium=5.00, n=10)
        ev = evaluate_hedge(h, spot_today=500.0, sigma_today=0.18,
                            r=0.045, q=0.013, today=today)
        # 5_000 / 500_000 = 1%
        self.assertAlmostEqual(ev.cost_of_insurance_pct, 0.01, places=10)

    def test_cost_of_insurance_invariant_to_contract_count(self) -> None:
        # Doubling n doubles upfront AND notional, so % is unchanged.
        h1, today = _spy_put_hedge(premium=5.00, n=1)
        h2, _     = _spy_put_hedge(premium=5.00, n=20)
        ev1 = evaluate_hedge(h1, 500.0, 0.18, 0.045, 0.013, today=today)
        ev2 = evaluate_hedge(h2, 500.0, 0.18, 0.045, 0.013, today=today)
        self.assertAlmostEqual(ev1.cost_of_insurance_pct,
                               ev2.cost_of_insurance_pct, places=12)


class TestBreakeven(unittest.TestCase):

    def test_put_breakeven_is_strike_minus_premium(self) -> None:
        h, today = _spy_put_hedge(premium=5.00)
        ev = evaluate_hedge(h, 500.0, 0.18, 0.045, 0.013, today=today)
        self.assertAlmostEqual(ev.breakeven_spot, 470.0, places=10)
        # spot 500 → 470 = -6% decline
        self.assertAlmostEqual(ev.breakeven_decline_pct, 0.06, places=10)

    def test_call_breakeven_is_strike_plus_premium(self) -> None:
        today = date(2026, 5, 24)
        h = Hedge(ticker="SPY", option_type="call", strike=525.0,
                  expiry=today + timedelta(days=90), n_contracts=1,
                  premium_per_share=3.00)
        ev = evaluate_hedge(h, 500.0, 0.18, 0.045, 0.013, today=today)
        self.assertAlmostEqual(ev.breakeven_spot, 528.0, places=10)


class TestScenarioApplication(unittest.TestCase):

    def test_default_scenarios_count(self) -> None:
        h, today = _spy_put_hedge()
        ev = evaluate_hedge(h, 500.0, 0.18, 0.045, 0.013, today=today)
        self.assertEqual(len(ev.scenarios), len(DEFAULT_SCENARIOS))
        # Names preserved in order
        self.assertEqual(
            tuple(s.scenario.name for s in ev.scenarios),
            tuple(s.name for s in DEFAULT_SCENARIOS),
        )

    def test_noop_scenario_reprices_to_base(self) -> None:
        # A (spot_mult=1.0, vol_mult=1.0) scenario must reprice to the
        # base value within tree-precision.
        h, today = _spy_put_hedge(premium=5.00)
        noop = (Scenario("Today", 1.0, 1.0),)
        ev = evaluate_hedge(h, 500.0, 0.18, 0.045, 0.013, today=today,
                            scenarios=noop)
        self.assertEqual(len(ev.scenarios), 1)
        self.assertAlmostEqual(ev.scenarios[0].repriced_per_share,
                               ev.base_per_share, places=10)
        # And the shocked (spot, vol) equal today's
        self.assertAlmostEqual(ev.scenarios[0].shocked_spot, 500.0)
        self.assertAlmostEqual(ev.scenarios[0].shocked_vol, 0.18)

    def test_put_pnl_positive_under_crash(self) -> None:
        # A long OTM put MUST gain value under the COVID-style scenario.
        h, today = _spy_put_hedge(premium=5.00, n=10)
        ev = evaluate_hedge(h, 500.0, 0.18, 0.045, 0.013, today=today)
        covid = next(s for s in ev.scenarios
                     if s.scenario.name == "COVID-style")
        self.assertGreater(covid.pnl_per_share, 0.0)
        self.assertGreater(covid.pnl_total, 0.0)
        # Sanity check arithmetic: pnl_total = n * 100 * pnl_per_share
        self.assertAlmostEqual(
            covid.pnl_total,
            10 * CONTRACT_MULT * covid.pnl_per_share,
            places=6,
        )

    def test_put_payoff_monotonic_in_crash_severity(self) -> None:
        # Worse crashes pay more (DEFAULT_SCENARIOS are ordered by severity).
        h, today = _spy_put_hedge(premium=5.00, n=1)
        ev = evaluate_hedge(h, 500.0, 0.18, 0.045, 0.013, today=today)
        payoffs = [s.repriced_per_share for s in ev.scenarios]
        self.assertEqual(payoffs, sorted(payoffs),
                         msg="Payoffs should increase with scenario severity")

    def test_call_pnl_positive_under_rally(self) -> None:
        today = date(2026, 5, 24)
        h = Hedge(ticker="SPY", option_type="call", strike=525.0,
                  expiry=today + timedelta(days=90), n_contracts=1,
                  premium_per_share=3.00)
        rally = (Scenario("Strong rally", 1.10, 1.5),)
        ev = evaluate_hedge(h, 500.0, 0.18, 0.045, 0.013, today=today,
                            scenarios=rally)
        self.assertGreater(ev.scenarios[0].pnl_per_share, 0.0)

    def test_pnl_pct_consistent_with_total(self) -> None:
        h, today = _spy_put_hedge(premium=5.00, n=7)
        ev = evaluate_hedge(h, 500.0, 0.18, 0.045, 0.013, today=today)
        for s in ev.scenarios:
            self.assertAlmostEqual(
                s.pnl_pct_of_notional,
                s.pnl_total / ev.notional_protected,
                places=12,
            )


class TestCustomScenarios(unittest.TestCase):

    def test_override_uses_supplied_scenarios_not_defaults(self) -> None:
        h, today = _spy_put_hedge()
        custom = (
            Scenario("Tiny dip", 0.99, 1.1),
            Scenario("Big spike", 1.05, 0.5),
        )
        ev = evaluate_hedge(h, 500.0, 0.18, 0.045, 0.013, today=today,
                            scenarios=custom)
        self.assertEqual([s.scenario.name for s in ev.scenarios],
                         ["Tiny dip", "Big spike"])
        # Shocks were applied correctly
        self.assertAlmostEqual(ev.scenarios[0].shocked_spot, 500.0 * 0.99)
        self.assertAlmostEqual(ev.scenarios[0].shocked_vol, 0.18 * 1.1)
        self.assertAlmostEqual(ev.scenarios[1].shocked_spot, 500.0 * 1.05)
        self.assertAlmostEqual(ev.scenarios[1].shocked_vol, 0.18 * 0.5)


class TestBaseReprice(unittest.TestCase):

    def test_base_matches_direct_pricer_call(self) -> None:
        # base_per_share should equal binomial_american price at the same
        # inputs (no scenarios applied to base).
        h, today = _spy_put_hedge(premium=5.00)
        ev = evaluate_hedge(h, 500.0, 0.18, 0.045, 0.013, today=today)
        direct = binomial_american(
            spot=500.0, strike=475.0, T=90/365.0, r=0.045, q=0.013,
            sigma=0.18, opt="put", n_steps=200, method="lr",
        )
        self.assertAlmostEqual(ev.base_per_share, direct["price"], places=10)


class TestEdgeCases(unittest.TestCase):

    def test_expired_hedge_raises(self) -> None:
        today = date(2026, 5, 24)
        h = Hedge(ticker="SPY", option_type="put", strike=475.0,
                  expiry=today - timedelta(days=1), n_contracts=1,
                  premium_per_share=5.00)
        with self.assertRaises(ValueError):
            evaluate_hedge(h, 500.0, 0.18, 0.045, 0.013, today=today)

    def test_expiry_today_is_allowed(self) -> None:
        # T=0 → pricer returns intrinsic; no scenarios should crash.
        today = date(2026, 5, 24)
        h = Hedge(ticker="SPY", option_type="put", strike=525.0,
                  expiry=today, n_contracts=1, premium_per_share=25.00)
        ev = evaluate_hedge(h, 500.0, 0.18, 0.045, 0.013, today=today)
        self.assertEqual(ev.T_years, 0.0)
        # Intrinsic at base: max(525 - 500, 0) = 25
        self.assertAlmostEqual(ev.base_per_share, 25.0, places=10)


class TestAtmBaselineMode(unittest.TestCase):
    """ATM-baseline mode applies vol_mult to ATM IV, then re-adds the
    absolute vol-point skew. Fixes the wing-skew bias documented in the
    module docstring."""

    def test_atm_mode_requires_sigma_atm(self) -> None:
        h, today = _spy_put_hedge()
        with self.assertRaises(ValueError) as cm:
            evaluate_hedge(h, 500.0, 0.18, 0.045, 0.013, today=today,
                           vol_baseline="atm")
        self.assertIn("sigma_atm", str(cm.exception))

    def test_atm_mode_rejects_nonpositive_sigma_atm(self) -> None:
        h, today = _spy_put_hedge()
        with self.assertRaises(ValueError):
            evaluate_hedge(h, 500.0, 0.18, 0.045, 0.013, today=today,
                           vol_baseline="atm", sigma_atm=0.0)
        with self.assertRaises(ValueError):
            evaluate_hedge(h, 500.0, 0.18, 0.045, 0.013, today=today,
                           vol_baseline="atm", sigma_atm=-0.1)

    def test_unknown_vol_baseline_raises(self) -> None:
        h, today = _spy_put_hedge()
        with self.assertRaises(ValueError):
            evaluate_hedge(h, 500.0, 0.18, 0.045, 0.013, today=today,
                           vol_baseline="proportional")  # type: ignore[arg-type]

    def test_atm_mode_records_metadata(self) -> None:
        h, today = _spy_put_hedge()
        ev = evaluate_hedge(h, 500.0, 0.38, 0.045, 0.013, today=today,
                            vol_baseline="atm", sigma_atm=0.16)
        self.assertEqual(ev.vol_baseline, "atm")
        self.assertAlmostEqual(ev.sigma_atm, 0.16, places=12)

    def test_contract_mode_clears_sigma_atm_in_result(self) -> None:
        # Even if a caller passes sigma_atm in contract mode, it should
        # not leak onto the result (it wasn't used).
        h, today = _spy_put_hedge()
        ev = evaluate_hedge(h, 500.0, 0.18, 0.045, 0.013, today=today,
                            vol_baseline="contract", sigma_atm=0.16)
        self.assertEqual(ev.vol_baseline, "contract")
        self.assertIsNone(ev.sigma_atm)

    def test_atm_equals_contract_when_contract_iv_equals_atm(self) -> None:
        # ATM contract: sigma_today == sigma_atm → skew=0 → both modes
        # produce identical scenario IVs and identical P&L.
        h, today = _spy_put_hedge(premium=5.00, n=5)
        ev_c = evaluate_hedge(h, 500.0, 0.18, 0.045, 0.013, today=today,
                              vol_baseline="contract")
        ev_a = evaluate_hedge(h, 500.0, 0.18, 0.045, 0.013, today=today,
                              vol_baseline="atm", sigma_atm=0.18)
        for sc_c, sc_a in zip(ev_c.scenarios, ev_a.scenarios):
            self.assertAlmostEqual(sc_c.shocked_vol, sc_a.shocked_vol, places=12)
            self.assertAlmostEqual(sc_c.repriced_per_share,
                                   sc_a.repriced_per_share, places=12)
            self.assertAlmostEqual(sc_c.pnl_total, sc_a.pnl_total, places=8)

    def test_atm_mode_noop_scenario_recovers_today(self) -> None:
        # The (1.0, 1.0) scenario must yield the contract's own IV under
        # atm mode: sigma_atm*1.0 + (sigma_today - sigma_atm) = sigma_today.
        h, today = _spy_put_hedge()
        noop = (Scenario("Today", 1.0, 1.0),)
        ev = evaluate_hedge(h, 500.0, 0.38, 0.045, 0.013, today=today,
                            scenarios=noop,
                            vol_baseline="atm", sigma_atm=0.16)
        self.assertAlmostEqual(ev.scenarios[0].shocked_vol, 0.38, places=12)
        self.assertAlmostEqual(ev.scenarios[0].repriced_per_share,
                               ev.base_per_share, places=10)

    def test_atm_mode_absolute_skew_arithmetic(self) -> None:
        # Explicit numbers: contract IV 38%, ATM 16% → skew = 22 vol pts.
        # 5x scenario: ATM_new = 80%, contract_new = 80 + 22 = 102%.
        h, today = _spy_put_hedge()
        crash = (Scenario("Crash", 0.65, 5.0),)
        ev = evaluate_hedge(h, 500.0, 0.38, 0.045, 0.013, today=today,
                            scenarios=crash,
                            vol_baseline="atm", sigma_atm=0.16)
        self.assertAlmostEqual(ev.scenarios[0].shocked_vol, 1.02, places=10)

    def test_atm_mode_lower_stressed_iv_for_wing(self) -> None:
        # Wing contract (sigma_today > sigma_atm): atm mode gives lower
        # stressed IV than contract mode for any vol_mult > 1.
        h, today = _spy_put_hedge()
        crash = (Scenario("Crash", 0.65, 5.0),)
        ev_c = evaluate_hedge(h, 500.0, 0.38, 0.045, 0.013, today=today,
                              scenarios=crash, vol_baseline="contract")
        ev_a = evaluate_hedge(h, 500.0, 0.38, 0.045, 0.013, today=today,
                              scenarios=crash,
                              vol_baseline="atm", sigma_atm=0.16)
        self.assertLess(ev_a.scenarios[0].shocked_vol,
                        ev_c.scenarios[0].shocked_vol)
        # Contract mode: 0.38 * 5 = 1.90; atm mode: 0.16 * 5 + 0.22 = 1.02.
        # ~88 vol-pts lower.
        self.assertAlmostEqual(
            ev_c.scenarios[0].shocked_vol - ev_a.scenarios[0].shocked_vol,
            0.88, places=10,
        )

    def test_atm_mode_lower_payoff_for_wing_under_crash(self) -> None:
        # Same setup as above: atm mode's lower stressed IV should give
        # a smaller (still positive) P&L on the long put.
        h, today = _spy_put_hedge(premium=5.00, n=10)
        crash = (Scenario("Crash", 0.65, 5.0),)
        ev_c = evaluate_hedge(h, 500.0, 0.38, 0.045, 0.013, today=today,
                              scenarios=crash, vol_baseline="contract")
        ev_a = evaluate_hedge(h, 500.0, 0.38, 0.045, 0.013, today=today,
                              scenarios=crash,
                              vol_baseline="atm", sigma_atm=0.16)
        self.assertGreater(ev_a.scenarios[0].pnl_total, 0.0)
        self.assertLess(ev_a.scenarios[0].pnl_total,
                        ev_c.scenarios[0].pnl_total)

    def test_atm_mode_floors_negative_sigma(self) -> None:
        # Exotic: vol-collapse scenario (vol_mult=0.1) on an ITM contract
        # where sigma_today > sigma_atm and the skew makes sig_shock
        # mathematically positive but tiny. Just verify no crash and
        # output stays positive.
        h, today = _spy_put_hedge()
        collapse = (Scenario("Vol collapse", 1.0, 0.1),)
        # sigma_atm=0.30, sigma_today=0.10 → skew=-0.20.
        # 0.30*0.1 + (-0.20) = 0.03 - 0.20 = -0.17 → floored to 1e-4.
        ev = evaluate_hedge(h, 500.0, 0.10, 0.045, 0.013, today=today,
                            scenarios=collapse,
                            vol_baseline="atm", sigma_atm=0.30)
        self.assertGreater(ev.scenarios[0].shocked_vol, 0.0)


class TestMtmBreakeven(unittest.TestCase):
    """MTM breakeven is the spot at which today's mark-to-market equals
    the upfront premium. Verify: (1) it's always closer to today's spot
    than the at-expiry breakeven (time value), (2) reprices at the solved
    spot match the premium within tolerance, (3) degenerate inputs
    surface as None, not nonsense numbers."""

    def test_put_mtm_be_above_at_expiry_be(self) -> None:
        # Universal for any OTM put with T>0: at spot = K-premium (the
        # at-expiry BE) the option's intrinsic equals premium and its
        # American value is strictly above intrinsic (time value > 0),
        # so we need a HIGHER spot to bring value back down to premium.
        # Hence mtm_be > breakeven_spot.
        h, today = _spy_put_hedge(premium=5.00)
        ev = evaluate_hedge(h, 500.0, 0.18, 0.045, 0.013, today=today)
        self.assertIsNotNone(ev.mtm_breakeven_spot)
        self.assertGreater(ev.mtm_breakeven_spot, ev.breakeven_spot)

    def test_mtm_be_position_tracks_premium_vs_base(self) -> None:
        # For a put: value(S) is decreasing in S. If premium < base
        # (paid less than today's mid → already in profit), mtm_be > spot.
        # If premium > base, mtm_be < spot.
        h_cheap, today = _spy_put_hedge(premium=4.00)   # likely below base
        h_dear, _      = _spy_put_hedge(premium=10.00)  # likely above base
        ev_cheap = evaluate_hedge(h_cheap, 500.0, 0.18, 0.045, 0.013, today=today)
        ev_dear  = evaluate_hedge(h_dear,  500.0, 0.18, 0.045, 0.013, today=today)
        # Confirm setup assumption: base sits between the two premiums.
        self.assertLess(ev_cheap.hedge.premium_per_share, ev_cheap.base_per_share)
        self.assertGreater(ev_dear.hedge.premium_per_share, ev_dear.base_per_share)
        # Now the directional claim:
        self.assertGreater(ev_cheap.mtm_breakeven_spot, ev_cheap.spot_today)
        self.assertLess(ev_dear.mtm_breakeven_spot,  ev_dear.spot_today)

    def test_put_mtm_be_reprices_to_premium(self) -> None:
        # Round-trip check: at the solved spot, the LR-priced option
        # value must equal the premium within tolerance.
        h, today = _spy_put_hedge(premium=5.00)
        ev = evaluate_hedge(h, 500.0, 0.18, 0.045, 0.013, today=today)
        priced = binomial_american(
            spot=ev.mtm_breakeven_spot, strike=h.strike, T=ev.T_years,
            r=0.045, q=0.013, sigma=0.18, opt="put",
            n_steps=200, method="lr",
        )
        self.assertAlmostEqual(priced["price"], h.premium_per_share, places=3)

    def test_call_mtm_be_below_at_expiry_be(self) -> None:
        # Symmetric to puts: at S = K + premium, intrinsic equals
        # premium, American call value strictly above → need LOWER S to
        # bring value down. mtm_be < breakeven_spot for an OTM call with T>0.
        today = date(2026, 5, 24)
        h = Hedge(ticker="SPY", option_type="call", strike=525.0,
                  expiry=today + timedelta(days=90), n_contracts=1,
                  premium_per_share=3.00)
        ev = evaluate_hedge(h, 500.0, 0.18, 0.045, 0.013, today=today)
        self.assertIsNotNone(ev.mtm_breakeven_spot)
        self.assertLess(ev.mtm_breakeven_spot, ev.breakeven_spot)

    def test_decline_pct_formula_consistent(self) -> None:
        # decline_pct must equal (spot_today - mtm_be) / spot_today
        # regardless of direction. Just verify the arithmetic.
        h, today = _spy_put_hedge(premium=5.00)
        ev = evaluate_hedge(h, 500.0, 0.18, 0.045, 0.013, today=today)
        self.assertAlmostEqual(
            ev.mtm_breakeven_decline_pct,
            (ev.spot_today - ev.mtm_breakeven_spot) / ev.spot_today,
            places=12,
        )

    def test_returns_none_when_premium_above_max_value(self) -> None:
        # Pathological: premium > strike for a put has no positive-spot
        # solution (American put value ≤ strike). Solver should return None,
        # not silently give a wrong answer.
        today = date(2026, 5, 24)
        h = Hedge(ticker="SPY", option_type="put", strike=100.0,
                  expiry=today + timedelta(days=30), n_contracts=1,
                  premium_per_share=200.0)  # impossible: paid more than strike
        ev = evaluate_hedge(h, 500.0, 0.20, 0.045, 0.013, today=today)
        self.assertIsNone(ev.mtm_breakeven_spot)
        self.assertIsNone(ev.mtm_breakeven_decline_pct)

    def test_returns_none_at_expiry(self) -> None:
        # T=0 means no time value left; "MTM" breakeven collapses to
        # at-expiry breakeven (already captured by breakeven_spot).
        # Solver returns None to signal "use the at-expiry number".
        today = date(2026, 5, 24)
        h = Hedge(ticker="SPY", option_type="put", strike=525.0,
                  expiry=today, n_contracts=1, premium_per_share=25.00)
        ev = evaluate_hedge(h, 500.0, 0.18, 0.045, 0.013, today=today)
        self.assertIsNone(ev.mtm_breakeven_spot)
        self.assertIsNone(ev.mtm_breakeven_decline_pct)

    def test_mtm_be_independent_of_n_contracts(self) -> None:
        # Per-share quantity → MTM BE depends only on (K, T, r, q, σ,
        # premium_per_share, opt_type). Doubling contracts must NOT
        # shift MTM BE.
        h1, today = _spy_put_hedge(premium=5.00, n=1)
        h10, _    = _spy_put_hedge(premium=5.00, n=10)
        ev1  = evaluate_hedge(h1,  500.0, 0.18, 0.045, 0.013, today=today)
        ev10 = evaluate_hedge(h10, 500.0, 0.18, 0.045, 0.013, today=today)
        self.assertAlmostEqual(ev1.mtm_breakeven_spot,
                               ev10.mtm_breakeven_spot, places=3)


if __name__ == "__main__":
    unittest.main()

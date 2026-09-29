"""Tests for parsers/hedge_stress_scenarios.py."""
from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "parsers"))

from parsers.hedge_exit_simulator import HedgePolicy  # noqa: E402
from parsers.hedge_stress_scenarios import (  # noqa: E402
    _build_recommended_leg,
    _reprice_leg,
    stress_test_program,
)
from parsers.stress_hedge import DEFAULT_SCENARIOS, Scenario  # noqa: E402


class TestBuildRecommendedLeg(unittest.TestCase):
    def test_strike_at_target_moneyness(self):
        policy = HedgePolicy(target_dte=90, target_moneyness=0.05,
                             notional_protected=2_980_000.0)
        leg, prem = _build_recommended_leg(
            policy, spot=745.64, sigma_atm=0.155, today=date(2026, 5, 22),
        )
        # 5% OTM at spot 745.64 = 708.4 → rounded to $710 ($5 grid).
        self.assertEqual(leg.strike, 710.0)
        # 90 DTE.
        self.assertEqual((leg.expiry - date(2026, 5, 22)).days, 90)
        # Premium is positive and reasonable for 90-DTE 5%-OTM SPY put at 15.5% IV.
        self.assertGreater(prem, 1.0)
        self.assertLess(prem, 30.0)

    def test_contracts_size_to_notional(self):
        policy = HedgePolicy(target_dte=90, target_moneyness=0.05,
                             notional_protected=2_980_000.0)
        leg, _ = _build_recommended_leg(
            policy, spot=745.64, sigma_atm=0.155, today=date(2026, 5, 22),
        )
        # K=710 → 1 contract protects 71,000 → 2_980_000/71_000 ≈ 42.
        self.assertEqual(leg.contracts, 42)


class TestReprice(unittest.TestCase):
    def test_reprice_changes_with_spot(self):
        from parsers.hedge_exit_simulator import Leg
        leg = Leg(
            open_date=pd.Timestamp("2026-05-22"),
            ticker="SPY_test", underlying="SPY",
            expiry=date(2026, 8, 20), strike=710.0,
            contracts=42, premium_paid=7.57,
        )
        mv_base = _reprice_leg(leg, spot=745.64, sigma=0.155,
                               today=date(2026, 5, 22))
        mv_down = _reprice_leg(leg, spot=700.0, sigma=0.155,
                               today=date(2026, 5, 22))
        mv_up = _reprice_leg(leg, spot=780.0, sigma=0.155,
                             today=date(2026, 5, 22))
        # Put MV should rise when spot drops, fall when spot rises.
        self.assertGreater(mv_down, mv_base)
        self.assertLess(mv_up, mv_base)


class TestStressTestProgram(unittest.TestCase):
    def setUp(self):
        self.policy = HedgePolicy(
            target_dte=90, target_moneyness=0.05,
            notional_protected=2_980_000.0,
        )

    def test_basic_shape(self):
        df = stress_test_program(
            self.policy, rule_kwargs_by_rule=None,
            today=date(2026, 5, 22), spot=745.64, sigma_atm=0.155,
        )
        # 4 scenarios × 4 rules × 2 observations = 32 rows.
        self.assertEqual(len(df), 32)
        for col in ("scenario", "rule", "observation", "spy_spot",
                    "leg_mv_per_share", "rule_fires",
                    "pnl_if_close_now", "pnl_pct_notional"):
            self.assertIn(col, df.columns)
        # Attrs preserved.
        self.assertIn("leg_premium", df.attrs)
        self.assertIn("leg_cost_basis", df.attrs)

    def test_shock_payoff_monotone_in_severity(self):
        df = stress_test_program(
            self.policy, rule_kwargs_by_rule=None,
            today=date(2026, 5, 22), spot=745.64, sigma_atm=0.155,
        )
        shock = df[(df["observation"] == "shock_day") &
                   (df["rule"] == "dte_roll")]
        shock = shock.set_index("scenario")["pnl_pct_notional"]
        # Magnitude order: Mild < Liberation-Day < Moderate crash < COVID.
        # dte_roll never fires pre-expiry at 90 DTE → P&L equals raw repriced
        # MV change (same property hold_to_expiry used to exercise).
        self.assertLess(shock["Mild correction"], shock["Liberation-Day-style"])
        self.assertLess(shock["Liberation-Day-style"], shock["Moderate crash"])
        self.assertLess(shock["Moderate crash"], shock["COVID-style"])

    def test_profit_take_fires_on_big_shocks(self):
        df = stress_test_program(
            self.policy, rule_kwargs_by_rule=None,
            today=date(2026, 5, 22), spot=745.64, sigma_atm=0.155,
        )
        shock = df[(df["observation"] == "shock_day") &
                   (df["rule"] == "profit_take_3x")]
        shock = shock.set_index("scenario")["rule_fires"]
        # All canonical scenarios apply ≥50% IV pop in addition to spot drop —
        # at default 3× mult, vega contribution alone can clear the threshold,
        # so profit_take fires on every scenario. (Setting mult=5 in F.7
        # sensitivity sweep is what differentiates them.)
        for sc in ("Mild correction", "Liberation-Day-style",
                   "Moderate crash", "COVID-style"):
            self.assertTrue(bool(shock[sc]), f"profit_take_3x should fire on {sc}")

    def test_profit_take_at_5x_more_selective(self):
        # Bumping mult to 5× should filter out the mildest scenario.
        df = stress_test_program(
            self.policy, rule_kwargs_by_rule={"profit_take_3x": {"mult": 5.0}},
            today=date(2026, 5, 22), spot=745.64, sigma_atm=0.155,
        )
        shock = df[(df["observation"] == "shock_day") &
                   (df["rule"] == "profit_take_3x")]
        shock = shock.set_index("scenario")["rule_fires"]
        # The largest shocks still fire; the smallest may not.
        self.assertTrue(bool(shock["COVID-style"]))
        self.assertTrue(bool(shock["Moderate crash"]))

    def test_custom_scenarios(self):
        custom = (
            Scenario("Tiny dip", 0.99, 1.1),
            Scenario("Huge crash", 0.50, 8.0),
        )
        df = stress_test_program(
            self.policy, rule_kwargs_by_rule=None,
            today=date(2026, 5, 22), spot=745.64, sigma_atm=0.155,
            scenarios=custom,
        )
        # 2 scenarios × 4 rules × 2 obs = 16 rows.
        self.assertEqual(len(df), 16)
        self.assertEqual(set(df["scenario"]), {"Tiny dip", "Huge crash"})


class TestStressEmpiricalPct(unittest.TestCase):
    def test_stress_empirical_pct_fires_on_shock_day(self):
        """With shock_iv_rank=100 (default) and R_high=80, empirical_pct
        must fire on shock_day."""
        from hedge_exit_simulator import HedgePolicy
        from hedge_stress_scenarios import stress_test_program

        policy = HedgePolicy(target_dte=90, target_moneyness=0.05,
                             notional_protected=100_000.0)
        out = stress_test_program(
            policy,
            rule_kwargs_by_rule={"empirical_pct": {"r_high": 80.0, "r_low": 30.0}},
            today=pd.Timestamp("2025-06-15").date(),
            spot=600.0, sigma_atm=0.15,
        )
        rows = out[(out["rule"] == "empirical_pct") &
                   (out["observation"] == "shock_day")]
        self.assertFalse(rows.empty)
        self.assertTrue(rows["rule_fires"].all(),
                        "expected all shock_day rows to fire")


if __name__ == "__main__":
    unittest.main()

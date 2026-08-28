"""Tests for parsers/hedge_recommender.py.

The engine has pure-math sub-steps (sized contracts given inputs) and
a full integration entry point. Unit tests cover the math; the full
integration is verified manually + by the Streamlit smoke test.
"""
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from hedge_recommender import (  # noqa: E402
    EXIT_RULE_MULTIPLIER,
    TENOR_DAYS,
    TAIL_STRIKE_OTM_PCT,
    CRASH_WINDOWS,
    SCENARIO_DRAWDOWNS,
    compute_excess_share,
    identify_excess_mcr_names,
    size_mode_a_basket,
    PutLeg,
)


class IdentifyExcessMCRTests(unittest.TestCase):
    def test_name_with_mcr_above_threshold_is_excess(self):
        # NVDA: portfolio MCR 12%, SPY natural weight 5% (x1.5 = 7.5%).
        # MCR 12% > 7.5% -> excess.
        per_symbol = pd.DataFrame({
            "pctr_pct":   [12.0, 8.0, 5.0],
            "weight_pct": [10.0, 10.0, 10.0],
        }, index=["NVDA", "AAPL", "MSFT"])
        spy_holdings = pd.DataFrame({
            "ticker":     ["NVDA", "AAPL", "MSFT"],
            "weight_pct": [5.0, 6.5, 5.0],  # NVDA: 5*1.5=7.5 (excess); AAPL: 6.5*1.5=9.75 (not); MSFT: 5*1.5=7.5 (not)
        })
        names = identify_excess_mcr_names(per_symbol, spy_holdings,
                                            threshold_mult=1.5)
        self.assertIn("NVDA", names)
        self.assertNotIn("AAPL", names)
        self.assertNotIn("MSFT", names)

    def test_name_absent_from_spy_uses_zero_natural_weight(self):
        # A holding not in SPY (e.g. an ADR, futures fund) has natural
        # weight = 0; threshold = 0; any positive MCR is "excess".
        per_symbol = pd.DataFrame({
            "pctr_pct":   [3.0],
            "weight_pct": [5.0],
        }, index=["XYZ"])
        spy_holdings = pd.DataFrame({"ticker": ["NVDA"], "weight_pct": [5.0]})
        names = identify_excess_mcr_names(per_symbol, spy_holdings,
                                            threshold_mult=1.5)
        self.assertIn("XYZ", names)

    def test_compute_excess_share_formula(self):
        # NVDA: MCR 12%, SPY natural 5% → excess = 7%; share = 7/12 ≈ 0.583
        # AAPL: MCR 8%, SPY natural 10% → MCR < natural → share = 0 (not excess but called for safety)
        per_symbol = pd.DataFrame({
            "pctr_pct":   [12.0, 8.0],
            "weight_pct": [10.0, 10.0],
        }, index=["NVDA", "AAPL"])
        spy_holdings = pd.DataFrame({
            "ticker":     ["NVDA", "AAPL"],
            "weight_pct": [5.0, 10.0],
        })
        out = compute_excess_share(per_symbol, spy_holdings, ["NVDA", "AAPL"])
        self.assertAlmostEqual(out["NVDA"], 7.0 / 12.0, places=4)
        self.assertAlmostEqual(out["AAPL"], 0.0, places=4)


class SizeModeABasketTests(unittest.TestCase):
    def test_caps_spy_at_target_in_worst_scenario(self):
        # Simple all-SPY portfolio, no excess names, no existing puts.
        # $100k all in SPY, beta=1, spot=$500, cap=10%, worst scenario -25%.
        # Unhedged loss at worst = $25k. Cap target = $10k. Gap = $15k.
        # Strike = 0.90 * $500 = $450.
        # Per-contract intrinsic at worst = (450 - 500*0.75)*100 = (450-375)*100 = $7,500.
        # Contracts needed = ceil(15000 / 7500) = 2.
        basket = size_mode_a_basket(
            portfolio_value=100_000,
            equity_weight=1.0,
            spy_drop_worst=-0.25,
            cap_pct=0.10,
            excess_names=[],
            per_name_drop_worst={"SPY": -0.25},
            per_name_position_mv={"SPY": 100_000},
            per_name_spot={"SPY": 500.0},
            per_name_excess_share={},
            chain_premiums={"SPY": {"strike": 450.0, "premium": 12.0}},
            existing_payoff_worst=0.0,
        )
        spy_leg = next((l for l in basket if l.ticker == "SPY"), None)
        self.assertIsNotNone(spy_leg)
        self.assertEqual(spy_leg.contracts, 2)
        self.assertEqual(spy_leg.role, "Systematic")
        self.assertAlmostEqual(spy_leg.strike, 450.0, places=2)

    def test_existing_payoff_reduces_required_spy_contracts(self):
        # Same as above but existing puts already cover $5k of the gap.
        # Remaining gap = $10k; contracts needed = ceil(10000/7500) = 2 still
        # (rounds up). Try $5,001 of remaining gap to verify ceil behavior.
        basket = size_mode_a_basket(
            portfolio_value=100_000,
            equity_weight=1.0,
            spy_drop_worst=-0.25,
            cap_pct=0.10,
            excess_names=[],
            per_name_drop_worst={"SPY": -0.25},
            per_name_position_mv={"SPY": 100_000},
            per_name_spot={"SPY": 500.0},
            per_name_excess_share={},
            chain_premiums={"SPY": {"strike": 450.0, "premium": 12.0}},
            existing_payoff_worst=9_999.0,  # gap drops from 15k to 5_001
        )
        spy_leg = next((l for l in basket if l.ticker == "SPY"), None)
        # ceil(5001 / 7500) = 1
        self.assertEqual(spy_leg.contracts, 1)

    def test_excess_name_gets_idiosyncratic_leg(self):
        # NVDA is excess (50% of its MCR is idiosyncratic).
        # NVDA holdings $20k, spot $100, beta=2, worst scenario SPY -25% ->
        # NVDA drops -50% to $50. Strike = 0.90 * $100 = $90.
        # Per-contract intrinsic at worst = (90 - 50)*100 = $4,000.
        # Idiosyncratic loss = $20k * -0.50 * 0.50 (excess share) = $5,000.
        # Contracts = ceil(5000 / 4000) = 2.
        basket = size_mode_a_basket(
            portfolio_value=100_000,
            equity_weight=1.0,
            spy_drop_worst=-0.25,
            cap_pct=0.10,
            excess_names=["NVDA"],
            per_name_drop_worst={"SPY": -0.25, "NVDA": -0.50},
            per_name_position_mv={"SPY": 80_000, "NVDA": 20_000},
            per_name_spot={"SPY": 500.0, "NVDA": 100.0},
            per_name_excess_share={"NVDA": 0.50},
            chain_premiums={
                "SPY":  {"strike": 450.0, "premium": 12.0},
                "NVDA": {"strike": 90.0,  "premium": 4.0},
            },
            existing_payoff_worst=0.0,
        )
        nvda_leg = next((l for l in basket if l.ticker == "NVDA"), None)
        self.assertIsNotNone(nvda_leg)
        self.assertEqual(nvda_leg.contracts, 2)
        self.assertEqual(nvda_leg.role, "Idiosyncratic (excess concentration)")

    def test_excess_name_with_nan_drop_is_skipped_not_crash(self):
        # Defense-in-depth: an excess name with no crash history (NaN drop —
        # a listing newer than the last crash window) must NOT crash sizing
        # via math.ceil(NaN) ("cannot convert float NaN to integer"). It is
        # skipped (no idiosyncratic leg — can't size a crash hedge without a
        # crash beta), and its NaN must not poison the SPY systematic sizing
        # either: the SPY leg still sizes from the finite names.
        basket = size_mode_a_basket(
            portfolio_value=100_000,
            equity_weight=1.0,
            spy_drop_worst=-0.25,
            cap_pct=0.10,
            excess_names=["DRAM"],
            per_name_drop_worst={"SPY": -0.25, "DRAM": float("nan")},
            per_name_position_mv={"SPY": 80_000, "DRAM": 20_000},
            per_name_spot={"SPY": 500.0, "DRAM": 30.0},
            per_name_excess_share={"DRAM": 1.0},
            chain_premiums={
                "SPY":  {"strike": 450.0, "premium": 12.0},
                "DRAM": {"strike": 27.0,  "premium": 2.0},
            },
            existing_payoff_worst=0.0,
        )
        # No leg for the no-history name.
        self.assertIsNone(next((l for l in basket if l.ticker == "DRAM"), None))
        # SPY systematic leg still sized (DRAM's NaN drop excluded from the
        # unhedged-loss sum, not propagated into residual_gap).
        spy_leg = next((l for l in basket if l.ticker == "SPY"), None)
        self.assertIsNotNone(spy_leg)
        self.assertGreaterEqual(spy_leg.contracts, 1)


class ConstantsTests(unittest.TestCase):
    def test_locked_constants_present(self):
        # These are referenced by the UI; failing import means UI breaks.
        self.assertEqual(TENOR_DAYS, 180)
        self.assertEqual(len(CRASH_WINDOWS), 4)
        self.assertEqual(SCENARIO_DRAWDOWNS,
                          [-0.05, -0.10, -0.15, -0.20, -0.25])
        # 5-crash re-validation locked the multiplier at 3 (Pareto winner:
        # pay/$drag = 3.96, drag = 2.16%/yr across 2018 Q4 + COVID + 2022 H1
        # + Apr 2025 + Iran 2026). Superseded the 2-year-sample mult=5.
        self.assertEqual(EXIT_RULE_MULTIPLIER, 3)


from hedge_recommender import size_mode_b_basket  # noqa: E402


class SizeModeBBasketTests(unittest.TestCase):
    def test_all_budget_to_spy_when_no_excess_names(self):
        # $100k portfolio, 1.0% budget = $1000/yr; tenor=180d -> $1000 * 180/365 = $493 per cycle.
        # Actually we spend the FULL annual budget on the 6-month put (per spec
        # "Total premium spend equals target budget % of portfolio per year");
        # implementation: spend budget_pct x portfolio on the 6-month basket,
        # which represents the annual spend (basket is rolled twice yearly).
        # SPY @ 0.80 x $500 = $400 strike; premium $5/share.
        # Budget = $1000; per-contract cost = 5 * 100 = $500.
        # Contracts = floor(1000 / 500) = 2.
        basket = size_mode_b_basket(
            portfolio_value=100_000,
            equity_weight=1.0,
            budget_pct=0.010,
            excess_names=[],
            per_name_spot={"SPY": 500.0},
            per_name_excess_mcr_pct={},
            spy_systematic_mcr_pct=80.0,
            chain_premiums={"SPY": {"strike": 400.0, "premium": 5.0}},
        )
        spy_leg = next((l for l in basket if l.ticker == "SPY"), None)
        self.assertIsNotNone(spy_leg)
        self.assertEqual(spy_leg.contracts, 2)
        self.assertAlmostEqual(spy_leg.strike, 400.0, places=2)

    def test_budget_split_by_excess_mcr_share(self):
        # Total excess "risk" = SPY systematic (40 pp) + NVDA excess (10 pp) = 50.
        # SPY share = 40/50 = 0.8; NVDA share = 10/50 = 0.2.
        # Budget = $1000; SPY = $800; NVDA = $200.
        # SPY: $400 strike, $5 premium -> per-contract $500 -> floor(800/500) = 1 contract.
        # NVDA: $80 strike, $2 premium -> per-contract $200 -> floor(200/200) = 1 contract.
        basket = size_mode_b_basket(
            portfolio_value=100_000,
            equity_weight=1.0,
            budget_pct=0.010,
            excess_names=["NVDA"],
            per_name_spot={"SPY": 500.0, "NVDA": 100.0},
            per_name_excess_mcr_pct={"NVDA": 10.0},
            spy_systematic_mcr_pct=40.0,
            chain_premiums={
                "SPY":  {"strike": 400.0, "premium": 5.0},
                "NVDA": {"strike": 80.0,  "premium": 2.0},
            },
        )
        spy = next(l for l in basket if l.ticker == "SPY")
        nvda = next(l for l in basket if l.ticker == "NVDA")
        self.assertEqual(spy.contracts, 1)
        self.assertEqual(nvda.contracts, 1)
        self.assertEqual(nvda.role, "Idiosyncratic (excess concentration)")


from hedge_recommender import (  # noqa: E402
    compute_roll_schedule,
    evaluate_existing_puts_payoff,
    existing_puts_to_table,
)


class EvaluateExistingPutsPayoffTests(unittest.TestCase):
    def test_intrinsic_only_at_each_scenario(self):
        # SPY put strike 475, 2 contracts; SPY spot 500.
        # Scenarios: -5/-10/-15/-20/-25% portfolio -> for an all-SPY equity slice
        #   at beta=1, SPY drops match scenario.
        # SPY @ -10% = $450; intrinsic = max(0, 475 - 450) * 100 = 2500 per contract; x 2 = $5000.
        positions = [
            {"underlying": "SPY", "strike": 475.0,
             "expiry": date(2026, 12, 18), "contracts": 2,
             "cost_basis_per_share": 12.0},
        ]
        per_scenario_spot = {
            -0.05: {"SPY": 475.0},  # 5% drop -> no intrinsic
            -0.10: {"SPY": 450.0},
            -0.15: {"SPY": 425.0},
            -0.20: {"SPY": 400.0},
            -0.25: {"SPY": 375.0},
        }
        payoffs = evaluate_existing_puts_payoff(positions, per_scenario_spot)
        self.assertAlmostEqual(payoffs[-0.05], 0.0, places=2)
        self.assertAlmostEqual(payoffs[-0.10], 5000.0, places=2)
        self.assertAlmostEqual(payoffs[-0.15], 10_000.0, places=2)
        self.assertAlmostEqual(payoffs[-0.25], 20_000.0, places=2)

    def test_missing_underlying_in_scenario_treated_as_zero(self):
        positions = [
            {"underlying": "TICKER_NOT_IN_PORTFOLIO", "strike": 100.0,
             "expiry": date(2026, 12, 18), "contracts": 1,
             "cost_basis_per_share": 5.0},
        ]
        per_scenario_spot = {-0.10: {"SPY": 450.0}}  # no TICKER_NOT_IN_PORTFOLIO entry
        payoffs = evaluate_existing_puts_payoff(positions, per_scenario_spot)
        self.assertAlmostEqual(payoffs[-0.10], 0.0, places=2)

    def test_nan_stress_spot_propagates_nan_not_silent_zero(self):
        # A no-crash-history underlying yields a NaN stressed spot (the ticker
        # IS present in the scenario, unlike the skip case above). The old code
        # did max(0.0, strike - nan) -> 0.0, silently valuing the put's
        # protection at $0 and understating hedge coverage. It must propagate
        # NaN so the scenario reads "uncomputable", mirroring the already-fixed
        # _intrinsic_per_contract sibling.
        positions = [
            {"underlying": "DRAM", "strike": 100.0,
             "expiry": date(2026, 12, 18), "contracts": 1,
             "cost_basis_per_share": 5.0},
        ]
        per_scenario_spot = {-0.10: {"DRAM": float("nan")}}
        payoffs = evaluate_existing_puts_payoff(positions, per_scenario_spot)
        self.assertTrue(np.isnan(payoffs[-0.10]),
                        f"expected NaN for a NaN stressed spot, got {payoffs[-0.10]}")

    def test_one_nan_spot_makes_whole_scenario_total_nan(self):
        # The scenario payoff sums across positions; if any leg's stressed spot
        # is missing the *total* protective floor is unknowable, so it must be
        # NaN rather than a finite number that silently omits that leg.
        positions = [
            {"underlying": "SPY", "strike": 475.0,
             "expiry": date(2026, 12, 18), "contracts": 2,
             "cost_basis_per_share": 12.0},
            {"underlying": "DRAM", "strike": 100.0,
             "expiry": date(2026, 12, 18), "contracts": 1,
             "cost_basis_per_share": 5.0},
        ]
        per_scenario_spot = {-0.10: {"SPY": 450.0, "DRAM": float("nan")}}
        payoffs = evaluate_existing_puts_payoff(positions, per_scenario_spot)
        self.assertTrue(np.isnan(payoffs[-0.10]),
                        f"expected NaN total, got {payoffs[-0.10]}")


class ExistingPutsToTableTests(unittest.TestCase):
    def test_builds_existing_put_dataclass(self):
        positions = [{
            "underlying": "SPY", "strike": 475.0,
            "expiry": date(2026, 12, 18), "contracts": 2,
            "cost_basis_per_share": 12.0, "market_value": 2800.0,
        }]
        worst_payoffs = [20_000.0]
        out = existing_puts_to_table(positions, worst_payoffs)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].ticker, "SPY")
        self.assertEqual(out[0].contracts, 2)
        self.assertAlmostEqual(out[0].worst_case_payoff, 20_000.0, places=2)
        self.assertAlmostEqual(out[0].cost_basis, 2 * 12.0 * 100, places=2)
        self.assertAlmostEqual(out[0].current_value, 2800.0, places=2)

    def test_same_underlying_strike_in_two_brokers_not_collided(self):
        # Two NVDA $135 12/18/26 lots in different brokers used to collide
        # in the dict-keyed worst_payoff_by_position and display the same
        # value on both rows. Parallel-list construction keeps them
        # independent per row.
        positions = [
            {"underlying": "NVDA", "strike": 135.0,
             "expiry": date(2026, 12, 18), "contracts": 11,
             "cost_basis_per_share": 3.0, "market_value": 2585.0},
            {"underlying": "NVDA", "strike": 135.0,
             "expiry": date(2026, 12, 18), "contracts": 25,
             "cost_basis_per_share": 3.0, "market_value": 6125.0},
        ]
        worst_payoffs = [30_800.0, 70_000.0]
        out = existing_puts_to_table(positions, worst_payoffs)
        self.assertEqual(len(out), 2)
        self.assertAlmostEqual(out[0].worst_case_payoff, 30_800.0)
        self.assertAlmostEqual(out[1].worst_case_payoff, 70_000.0)


class ComputeRollScheduleTests(unittest.TestCase):
    def test_roll_by_is_90_days_before_expiry(self):
        roll_by, _ = compute_roll_schedule(
            date(2026, 12, 18), today=date(2026, 5, 28),
        )
        self.assertEqual(roll_by, date(2026, 12, 18) - timedelta(days=90))

    def test_roll_into_is_180_days_after_a_future_roll_by(self):
        roll_by, roll_into = compute_roll_schedule(
            date(2026, 12, 18), today=date(2026, 5, 28),
        )
        # roll_by is in the future -> roll_into anchors on roll_by + 180d.
        self.assertGreater(roll_by, date(2026, 5, 28))
        self.assertEqual(roll_into, roll_by + timedelta(days=180))

    def test_overdue_put_anchors_roll_into_on_today(self):
        # Expiry only 30 days out -> roll_by is already past, so roll_into is
        # measured from today, not the stale roll_by.
        today = date(2026, 5, 28)
        roll_by, roll_into = compute_roll_schedule(
            date(2026, 6, 27), today=today,
        )
        self.assertLess(roll_by, today)
        self.assertEqual(roll_into, today + timedelta(days=180))


from hedge_recommender import build_hedge_basket  # noqa: E402


class BuildHedgeBasketIntegrationTests(unittest.TestCase):
    """Exercises the orchestration with all dependencies mocked.

    Verifies the engine assembles a HedgeRecommendation with the
    expected scenario count, non-empty new-puts when existing puts
    don't cover, and a sensible headline cap.
    """

    def _holdings(self):
        return pd.DataFrame({
            "ticker":        ["SPY", "NVDA", "SGOV"],
            "quantity":      [100, 100, 1000],
            "market_value":  [50_000, 20_000, 30_000],
        })

    def _existing_puts(self):
        return [{
            "underlying": "SPY", "strike": 475.0,
            "expiry": date(2026, 12, 18), "contracts": 1,
            "cost_basis_per_share": 12.0, "market_value": 1400.0,
        }]

    def _per_symbol_mcr(self):
        return pd.DataFrame({
            "pctr_pct":   [40.0, 15.0, 0.0],
            "weight_pct": [50.0, 20.0, 30.0],
        }, index=["SPY", "NVDA", "SGOV"])

    def _spy_holdings(self):
        # NVDA natural weight 5%; portfolio 15% -> excess (15 > 1.5*5=7.5)
        return pd.DataFrame({
            "ticker":     ["SPY", "NVDA"],
            "weight_pct": [100.0, 5.0],
        })

    def _chain_premiums(self, mode: str, cap_pct: float):
        # Spot prices on test day: SPY=500, NVDA=100.
        # Mode A cap=10% -> strikes at 90% spot.
        # Mode B -> strikes at 80% spot.
        strike_mult = (1 - cap_pct) if mode == "A" else (1 - TAIL_STRIKE_OTM_PCT)
        return {
            "SPY":  {"strike": 500 * strike_mult, "premium": 12.0,
                     "expiry": date(2026, 12, 18)},
            "NVDA": {"strike": 100 * strike_mult, "premium": 3.0,
                     "expiry": date(2026, 12, 18)},
        }

    def test_returns_recommendation_with_5_scenarios(self):
        rec = build_hedge_basket(
            mode="A",
            target=0.10,
            holdings=self._holdings(),
            existing_options=self._existing_puts(),
            per_symbol_mcr=self._per_symbol_mcr(),
            spy_holdings=self._spy_holdings(),
            chain_premiums=self._chain_premiums("A", 0.10),
            crash_betas={"SPY": 1.0, "NVDA": 1.5, "SGOV": 0.0},
            today=date(2026, 5, 27),
            spot_prices={"SPY": 500.0, "NVDA": 100.0, "SGOV": 100.0},
        )
        self.assertEqual(rec.mode, "A")
        self.assertEqual(len(rec.scenarios), 5)
        self.assertEqual([s.portfolio_drawdown for s in rec.scenarios],
                          [-0.05, -0.10, -0.15, -0.20, -0.25])

    def test_diagnostics_expose_mcr_breakdown_mode_a(self):
        # The hedge-signal panel reads these from diagnostics regardless of
        # mode. NVDA is the lone excess name (15% > 1.5*5%); SPY systematic
        # = 100 - 15 (NVDA excess) - 0 (SGOV cash) = 85.
        rec = build_hedge_basket(
            mode="A", target=0.10,
            holdings=self._holdings(),
            existing_options=self._existing_puts(),
            per_symbol_mcr=self._per_symbol_mcr(),
            spy_holdings=self._spy_holdings(),
            chain_premiums=self._chain_premiums("A", 0.10),
            crash_betas={"SPY": 1.0, "NVDA": 1.5, "SGOV": 0.0},
            today=date(2026, 5, 27),
            spot_prices={"SPY": 500.0, "NVDA": 100.0, "SGOV": 100.0},
        )
        self.assertEqual(rec.diagnostics["per_name_mcr_pct"], {"NVDA": 15.0})
        self.assertAlmostEqual(rec.diagnostics["spy_systematic_mcr_pct"], 85.0)

    def test_diagnostics_expose_mcr_breakdown_mode_b(self):
        rec = build_hedge_basket(
            mode="B", target=0.02,
            holdings=self._holdings(),
            existing_options=self._existing_puts(),
            per_symbol_mcr=self._per_symbol_mcr(),
            spy_holdings=self._spy_holdings(),
            chain_premiums=self._chain_premiums("B", 0.02),
            crash_betas={"SPY": 1.0, "NVDA": 1.5, "SGOV": 0.0},
            today=date(2026, 5, 27),
            spot_prices={"SPY": 500.0, "NVDA": 100.0, "SGOV": 100.0},
        )
        self.assertEqual(rec.diagnostics["per_name_mcr_pct"], {"NVDA": 15.0})
        self.assertAlmostEqual(rec.diagnostics["spy_systematic_mcr_pct"], 85.0)

    def test_already_covered_yields_empty_new_puts(self):
        # Existing puts cover the gap entirely -> new_puts is empty.
        existing = [{
            "underlying": "SPY", "strike": 475.0,
            "expiry": date(2026, 12, 18), "contracts": 100,  # huge
            "cost_basis_per_share": 12.0, "market_value": 1400.0,
        }]
        rec = build_hedge_basket(
            mode="A", target=0.10,
            holdings=self._holdings(),
            existing_options=existing,
            per_symbol_mcr=self._per_symbol_mcr(),
            spy_holdings=self._spy_holdings(),
            chain_premiums=self._chain_premiums("A", 0.10),
            crash_betas={"SPY": 1.0, "NVDA": 1.5, "SGOV": 0.0},
            today=date(2026, 5, 27),
            spot_prices={"SPY": 500.0, "NVDA": 100.0, "SGOV": 100.0},
        )
        self.assertEqual(rec.new_puts, [])

    def test_worst_case_payoff_is_nan_when_spot_missing(self):
        # Defensive guard: if the worst-scenario spot map is missing a
        # ticker (e.g. upstream NaN cascade left per_scenario_spot empty),
        # the existing-put worst-case payoff must NOT silently fall back
        # to strike × contracts × 100 (which displays as the full
        # "underlying went to $0" notional). NaN is the honest answer.
        # The cascade happens when invert_portfolio_drawdown returns NaN
        # (e.g. crash_betas missing for held tickers) — we simulate by
        # passing crash_betas with all NaN values so spy_drop is NaN.
        existing = [{
            "underlying": "NVDA", "strike": 115.0,
            "expiry": date(2026, 12, 18), "contracts": 40,
            "cost_basis_per_share": 3.0, "market_value": 4000.0,
        }]
        # All-NaN crash_betas -> weighted_beta = NaN -> spy_drop = NaN ->
        # per_scenario_spot is {} for every scenario.
        rec = build_hedge_basket(
            mode="A", target=0.10,
            holdings=self._holdings(),
            existing_options=existing,
            per_symbol_mcr=self._per_symbol_mcr(),
            spy_holdings=self._spy_holdings(),
            chain_premiums=self._chain_premiums("A", 0.10),
            crash_betas={"SPY": float("nan"), "NVDA": float("nan"),
                          "SGOV": float("nan")},
            today=date(2026, 5, 27),
            spot_prices={"SPY": 500.0, "NVDA": 100.0, "SGOV": 100.0},
        )
        import math as _m
        nvda_row = next((p for p in rec.existing_puts if p.ticker == "NVDA"),
                         None)
        self.assertIsNotNone(nvda_row)
        # BEFORE the guard: worst_case_payoff = 115 * 40 * 100 = 460_000.
        # AFTER the guard: NaN — the missing-spot condition surfaces.
        self.assertTrue(_m.isnan(nvda_row.worst_case_payoff),
                         f"Expected NaN, got {nvda_row.worst_case_payoff}")
        # All scenarios should also have NaN unhedged/existing/new/combined
        # PnLs (was silently 0, which then aggregated to cap = 0% and
        # "Already covered — buy nothing" — a gaslighting failure mode).
        for s in rec.scenarios:
            self.assertTrue(_m.isnan(s.implied_spy_drop))
            self.assertTrue(_m.isnan(s.unhedged_pnl))
            self.assertTrue(_m.isnan(s.existing_payoff))
            self.assertTrue(_m.isnan(s.combined_pnl))
            self.assertTrue(_m.isnan(s.combined_pnl_pct))
        # Headline caps should be NaN (not 0.0) so the UI knows to render
        # a "scenarios unavailable" warning instead of "Already covered".
        self.assertTrue(_m.isnan(rec.current_cap_pct))
        self.assertTrue(_m.isnan(rec.combined_cap_pct))
        # Cap-precision note must be empty (the old code printed "within
        # target across all 5 scenarios" with cap_observed = 0.0).
        self.assertEqual(rec.cap_precision_note, "")

    def test_single_no_history_name_does_not_blank_scenarios(self):
        # Regression (the DRAM bug): a held equity name with no crash history
        # — NaN crash beta, e.g. a listing newer than the most recent crash
        # window — must NOT NaN-out the entire scenarios table. It is excluded,
        # the remaining equity renormalized, and the exclusion surfaced in
        # diagnostics. Contrast test_worst_case_payoff_is_nan_when_spot_missing,
        # where EVERY beta is NaN and NaN is the honest answer.
        holdings = pd.DataFrame({
            "ticker":       ["SPY", "NVDA", "DRAM", "SGOV"],
            "quantity":     [100, 100, 100, 1000],
            "market_value": [50_000, 20_000, 5_000, 25_000],
        })
        rec = build_hedge_basket(
            mode="A", target=0.10,
            holdings=holdings,
            existing_options=self._existing_puts(),
            per_symbol_mcr=self._per_symbol_mcr(),
            spy_holdings=self._spy_holdings(),
            chain_premiums=self._chain_premiums("A", 0.10),
            crash_betas={"SPY": 1.0, "NVDA": 1.5, "DRAM": float("nan"),
                          "SGOV": 0.0},
            today=date(2026, 5, 27),
            spot_prices={"SPY": 500.0, "NVDA": 100.0, "DRAM": 30.0,
                          "SGOV": 100.0},
        )
        # Every scenario row is fully finite — the table renders.
        for s in rec.scenarios:
            self.assertTrue(np.isfinite(s.implied_spy_drop), s.portfolio_drawdown)
            self.assertTrue(np.isfinite(s.unhedged_pnl), s.portfolio_drawdown)
            self.assertTrue(np.isfinite(s.combined_pnl), s.portfolio_drawdown)
            self.assertTrue(np.isfinite(s.combined_pnl_pct), s.portfolio_drawdown)
        # Caps computable -> headline is NOT "scenarios unavailable".
        self.assertTrue(np.isfinite(rec.current_cap_pct))
        self.assertTrue(np.isfinite(rec.combined_cap_pct))
        # The excluded name is disclosed for the UI caption, with its
        # pre-renormalization share of the equity slice (5k / 75k equity).
        excluded = dict(rec.diagnostics["scenario_excluded_no_history"])
        self.assertIn("DRAM", excluded)
        self.assertAlmostEqual(excluded["DRAM"], 5_000 / 75_000, places=4)

    def test_excess_no_history_name_does_not_crash_sizing(self):
        # Regression (the second DRAM bug, surfaced after the inversion fix):
        # a no-crash-history name (NaN beta) that is ALSO flagged excess
        # (natural SPY weight ~0 → any MCR is "excess") gets a NaN per-name
        # drop. That NaN reached math.ceil() in size_mode_a_basket →
        # "cannot convert float NaN to integer", which the Options tab showed
        # as "Could not build hedge basket". It must be excluded from sizing
        # (no idio leg) with the rest of the basket built normally.
        holdings = pd.DataFrame({
            "ticker":       ["SPY", "DRAM", "SGOV"],
            "quantity":     [100, 100, 1000],
            "market_value": [50_000, 20_000, 30_000],
        })
        per_symbol_mcr = pd.DataFrame(
            {"pctr_pct": [60.0, 40.0, 0.0], "weight_pct": [50.0, 20.0, 30.0]},
            index=["SPY", "DRAM", "SGOV"])
        spy_holdings = pd.DataFrame({"ticker": ["SPY"], "weight_pct": [100.0]})
        chain = {
            "SPY":  {"strike": 450.0, "premium": 12.0, "expiry": date(2026, 12, 18)},
            "DRAM": {"strike": 27.0,  "premium": 2.0,  "expiry": date(2026, 12, 18)},
        }
        rec = build_hedge_basket(
            mode="A", target=0.10, holdings=holdings, existing_options=[],
            per_symbol_mcr=per_symbol_mcr, spy_holdings=spy_holdings,
            chain_premiums=chain,
            crash_betas={"SPY": 1.0, "DRAM": float("nan"), "SGOV": 0.0},
            today=date(2026, 5, 27),
            spot_prices={"SPY": 500.0, "DRAM": 30.0, "SGOV": 100.0},
        )
        # Builds without raising; scenarios + caps finite.
        for s in rec.scenarios:
            self.assertTrue(np.isfinite(s.implied_spy_drop), s.portfolio_drawdown)
            self.assertTrue(np.isfinite(s.unhedged_pnl), s.portfolio_drawdown)
        self.assertTrue(np.isfinite(rec.current_cap_pct))
        self.assertTrue(np.isfinite(rec.combined_cap_pct))
        # No idiosyncratic leg for the no-history name; it's disclosed instead.
        self.assertNotIn("DRAM", [l.ticker for l in rec.new_puts])
        self.assertIn("DRAM", dict(rec.diagnostics["scenario_excluded_no_history"]))

    def test_duplicate_ticker_rows_get_summed(self):
        # Multi-broker case: SPY held in both Harbor and Alpine surfaces as
        # two rows in the same snapshot. The engine must collapse by
        # ticker (sum market_value) before computing weights / per-name
        # position MV — otherwise weights_in_equity gets a duplicate
        # index that inflates weighted_beta, and dict(zip(...)) silently
        # keeps only the last row in per_name_position_mv.
        holdings = pd.DataFrame({
            "ticker":       ["SPY", "SPY", "NVDA", "SGOV"],
            "quantity":     [60, 40, 100, 1000],
            "market_value": [30_000, 20_000, 20_000, 30_000],
        })
        rec = build_hedge_basket(
            mode="A", target=0.10,
            holdings=holdings,
            existing_options=[],
            per_symbol_mcr=self._per_symbol_mcr(),
            spy_holdings=self._spy_holdings(),
            chain_premiums=self._chain_premiums("A", 0.10),
            crash_betas={"SPY": 1.0, "NVDA": 1.5, "SGOV": 0.0},
            today=date(2026, 5, 27),
            spot_prices={"SPY": 500.0, "NVDA": 100.0, "SGOV": 100.0},
        )
        # Sums are correct via DataFrame.sum() regardless of dedup.
        self.assertAlmostEqual(rec.diagnostics["portfolio_value"], 100_000,
                                 places=2)
        self.assertAlmostEqual(rec.diagnostics["equity_weight"], 0.70,
                                 places=4)
        # Scenario-inverter invariant: unhedged $ P&L at the worst portfolio
        # drawdown should equal portfolio_drawdown × portfolio_value
        # (after beta unwinding). With the duplicate-ticker bug this lands
        # way off (weighted_beta is inflated AND SPY's market_value is
        # truncated to 20k instead of 50k).
        worst_row = rec.scenarios[-1]
        self.assertAlmostEqual(worst_row.unhedged_pnl, -0.25 * 100_000,
                                 delta=10.0)
        # Implied SPY drop at -25% portfolio drawdown:
        #   equity drawdown = -0.25 / 0.70 = -0.3571
        #   weighted_beta = (50/70)*1.0 + (20/70)*1.5 = 1.1429
        #   spy_drop = -0.3571 / 1.1429 = -0.3125
        self.assertAlmostEqual(worst_row.implied_spy_drop, -0.3125, places=3)

    def test_mode_b_uses_20pct_otm_strikes(self):
        rec = build_hedge_basket(
            mode="B", target=0.010,
            holdings=self._holdings(),
            existing_options=[],
            per_symbol_mcr=self._per_symbol_mcr(),
            spy_holdings=self._spy_holdings(),
            chain_premiums=self._chain_premiums("B", 0.10),
            crash_betas={"SPY": 1.0, "NVDA": 1.5, "SGOV": 0.0},
            today=date(2026, 5, 27),
            spot_prices={"SPY": 500.0, "NVDA": 100.0, "SGOV": 100.0},
        )
        self.assertEqual(rec.mode, "B")
        # SPY leg should be at $400 (80% of 500)
        spy = next((l for l in rec.new_puts if l.ticker == "SPY"), None)
        if spy is not None:
            self.assertAlmostEqual(spy.strike, 400.0, places=2)

    def test_mode_b_recommends_full_basket_ignoring_existing_puts(self):
        # Mode B is a standing premium ladder: it recommends the full budget
        # basket regardless of existing coverage (no netting). A huge existing
        # SPY put that would zero out a Mode-A recommendation must NOT change
        # Mode B's basket. (Budget 2%/yr so the fixture affords whole
        # contracts; at 1% the basket is empty either way.)
        big_existing = [{
            "underlying": "SPY", "strike": 480.0,
            "expiry": date(2026, 12, 18), "contracts": 100,
            "cost_basis_per_share": 10.0, "market_value": 50_000.0,
        }]
        common = dict(
            mode="B", target=0.02,
            holdings=self._holdings(),
            per_symbol_mcr=self._per_symbol_mcr(),
            spy_holdings=self._spy_holdings(),
            chain_premiums=self._chain_premiums("B", 0.10),
            crash_betas={"SPY": 1.0, "NVDA": 1.5, "SGOV": 0.0},
            today=date(2026, 5, 27),
            spot_prices={"SPY": 500.0, "NVDA": 100.0, "SGOV": 100.0},
        )
        rec_with = build_hedge_basket(existing_options=big_existing, **common)
        rec_without = build_hedge_basket(existing_options=[], **common)
        # Budget basket is non-empty and identical with or without existing
        # puts — Mode B does not net against existing coverage.
        self.assertTrue(rec_without.new_puts,
                        "expected a non-empty budget basket at 2%/yr")
        self.assertEqual(
            [(leg.ticker, leg.contracts) for leg in rec_with.new_puts],
            [(leg.ticker, leg.contracts) for leg in rec_without.new_puts],
        )


# ---------------------------------------------------------------------------
# Worst-case sign fix + existing-only P&L columns.
#
# The headline caps must report the most-NEGATIVE signed net P&L (a real loss),
# never max(abs(...)) — which mislabeled a convex GAIN in the deepest scenario
# as a "worst-case loss" (the 23.6% bug). And the scenarios table needs an
# existing-only P&L ($ + %) so the existing-puts cap (the 11.7%) is traceable
# to a row.
# ---------------------------------------------------------------------------
from hedge_recommender import (  # noqa: E402
    ScenarioRow, _worst_case_pnl_pct, _loss_cap_pct,
)


def _mk_scenario(**kw) -> "ScenarioRow":
    """ScenarioRow with finite defaults; override per test. Only the fields the
    aggregation helpers read (implied_spy_drop, unhedged_pnl, existing_payoff,
    new_payoff) matter for the helper tests; the rest carry placeholder values."""
    base = dict(
        portfolio_drawdown=-0.10, implied_spy_drop=-0.12,
        unhedged_pnl=-10_000.0, existing_payoff=0.0,
        existing_pnl=-10_000.0, existing_pnl_pct=-0.10,
        new_payoff=0.0, combined_pnl=-10_000.0, combined_pnl_pct=-0.10,
    )
    base.update(kw)
    return ScenarioRow(**base)


class WorstCaseSignTests(unittest.TestCase):
    def test_worst_case_is_most_negative_not_largest_magnitude(self):
        # Over-cover: shallow scenario a small LOSS, deep scenario a big GAIN.
        # Worst case is the LOSS (-3%), not the +23% gain max(abs) grabbed.
        scenarios = [
            _mk_scenario(implied_spy_drop=-0.06, unhedged_pnl=-5_000.0,
                         new_payoff=2_000.0),    # net -3_000 -> -3%
            _mk_scenario(implied_spy_drop=-0.30, unhedged_pnl=-25_000.0,
                         new_payoff=48_000.0),   # net +23_000 -> +23%
        ]
        worst = _worst_case_pnl_pct(scenarios, 100_000.0, include_new=True)
        self.assertAlmostEqual(worst, -0.03, places=6)

    def test_worst_case_all_gains_stays_positive(self):
        scenarios = [
            _mk_scenario(implied_spy_drop=-0.06, unhedged_pnl=-5_000.0,
                         new_payoff=7_000.0),    # +2%
            _mk_scenario(implied_spy_drop=-0.30, unhedged_pnl=-25_000.0,
                         new_payoff=48_000.0),   # +23%
        ]
        worst = _worst_case_pnl_pct(scenarios, 100_000.0, include_new=True)
        self.assertAlmostEqual(worst, 0.02, places=6)

    def test_worst_case_skips_nan_implied_drop(self):
        scenarios = [
            _mk_scenario(implied_spy_drop=float("nan"), unhedged_pnl=-99_000.0),
            _mk_scenario(implied_spy_drop=-0.12, unhedged_pnl=-8_000.0),
        ]
        worst = _worst_case_pnl_pct(scenarios, 100_000.0, include_new=True)
        self.assertAlmostEqual(worst, -0.08, places=6)

    def test_worst_case_nan_when_no_valid_scenarios(self):
        scenarios = [_mk_scenario(implied_spy_drop=float("nan"))]
        self.assertTrue(
            np.isnan(_worst_case_pnl_pct(scenarios, 100_000.0, include_new=True))
        )

    def test_include_new_false_excludes_new_payoff(self):
        scenarios = [
            _mk_scenario(implied_spy_drop=-0.12, unhedged_pnl=-10_000.0,
                         existing_payoff=3_000.0, new_payoff=50_000.0),
        ]
        # existing only: (-10_000 + 3_000)/100_000 = -0.07 (ignores the new 50k)
        self.assertAlmostEqual(
            _worst_case_pnl_pct(scenarios, 100_000.0, include_new=False),
            -0.07, places=6,
        )

    def test_loss_cap_positive_for_a_loss(self):
        self.assertAlmostEqual(_loss_cap_pct(-0.117), 0.117, places=6)

    def test_loss_cap_zero_for_a_gain(self):
        # The core bug: a convex gain must cap at 0, NOT +0.236.
        self.assertEqual(_loss_cap_pct(0.236), 0.0)

    def test_loss_cap_propagates_nan(self):
        self.assertTrue(np.isnan(_loss_cap_pct(float("nan"))))


class CapSignAndExistingPnlIntegrationTests(unittest.TestCase):
    def _holdings(self):
        return pd.DataFrame({
            "ticker":       ["SPY", "SGOV"],
            "quantity":     [100, 1000],
            "market_value": [50_000, 50_000],
        })

    def _mcr(self):
        return pd.DataFrame(
            {"pctr_pct": [100.0, 0.0], "weight_pct": [50.0, 50.0]},
            index=["SPY", "SGOV"])

    def _spy_holdings(self):
        return pd.DataFrame({"ticker": ["SPY"], "weight_pct": [100.0]})

    def test_overcovering_basket_reports_gain_not_loss(self):
        # A huge existing SPY put makes every scenario a net GAIN. The combined
        # cap must be 0 (no loss), and the signed worst-case field must be > 0 —
        # never a positive "cap" that is really a gain (the 23.6% bug).
        existing = [{
            "underlying": "SPY", "strike": 480.0, "expiry": date(2026, 12, 18),
            "contracts": 50, "cost_basis_per_share": 12.0,
            "market_value": 60_000.0,
        }]
        rec = build_hedge_basket(
            mode="A", target=0.10, holdings=self._holdings(),
            existing_options=existing, per_symbol_mcr=self._mcr(),
            spy_holdings=self._spy_holdings(),
            chain_premiums={"SPY": {"strike": 450.0, "premium": 12.0,
                                     "expiry": date(2026, 12, 18)}},
            crash_betas={"SPY": 1.0, "SGOV": 0.0},
            today=date(2026, 5, 27),
            spot_prices={"SPY": 500.0, "SGOV": 100.0},
        )
        self.assertGreater(rec.combined_worst_pnl_pct, 0.0)
        self.assertEqual(rec.combined_cap_pct, 0.0)

    def test_scenario_rows_expose_existing_only_pnl(self):
        # Existing-hedge P&L = unhedged + existing payoff (the result with only
        # the current puts); its % uses portfolio_value.
        existing = [{
            "underlying": "SPY", "strike": 470.0, "expiry": date(2026, 12, 18),
            "contracts": 5, "cost_basis_per_share": 12.0, "market_value": 6_000.0,
        }]
        rec = build_hedge_basket(
            mode="A", target=0.10, holdings=self._holdings(),
            existing_options=existing, per_symbol_mcr=self._mcr(),
            spy_holdings=self._spy_holdings(),
            chain_premiums={"SPY": {"strike": 450.0, "premium": 12.0,
                                     "expiry": date(2026, 12, 18)}},
            crash_betas={"SPY": 1.0, "SGOV": 0.0},
            today=date(2026, 5, 27),
            spot_prices={"SPY": 500.0, "SGOV": 100.0},
        )
        pv = rec.diagnostics["portfolio_value"]
        for s in rec.scenarios:
            self.assertAlmostEqual(s.existing_pnl,
                                   s.unhedged_pnl + s.existing_payoff, places=4)
            self.assertAlmostEqual(s.existing_pnl_pct, s.existing_pnl / pv,
                                   places=6)


if __name__ == "__main__":
    unittest.main()

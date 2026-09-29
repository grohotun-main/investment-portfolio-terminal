"""Unit + integration tests for parsers/hedge_exit_simulator.py.

Builds synthetic SPY history and option-grid frames so each exit rule's
trigger condition is hand-verifiable. The simulator itself runs against
this same fixture in the integration tests.
"""
from __future__ import annotations

import sys
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from parsers.hedge_exit_simulator import (  # noqa: E402
    CONTRACT_MULT,
    EXIT_RULES,
    HedgePolicy,
    Leg,
    SimState,
    build_grid_index,
    compare_runs,
    contracts_for_notional,
    episode_payoffs,
    lookup_close,
    pareto_frontier_mask,
    resolve_from_cache,
    rule_dte_roll,
    rule_empirical_percentile,
    rule_monetize_recovery,
    rule_profit_take,
    simulate_program,
    summarize_run,
)


def _make_leg(expiry=date(2025, 9, 19), strike=550.0, contracts=1,
              premium_paid=10.0, open_date=pd.Timestamp("2025-06-15")) -> Leg:
    return Leg(
        open_date=open_date, ticker="O:SPY250919P00550000",
        underlying="SPY", expiry=expiry, strike=strike,
        contracts=contracts, premium_paid=premium_paid,
    )


def _make_state(today, spot, peak, trough, mv_per_share, contracts=1,
                flatted=False, flatted_at_peak=None) -> SimState:
    return SimState(
        today=pd.Timestamp(today), spot=spot,
        peak_spot=peak, trough_spot=trough,
        leg_mv_per_share=mv_per_share,
        leg_total_mv=mv_per_share * contracts * CONTRACT_MULT,
        flatted=flatted, flatted_at_peak=flatted_at_peak,
    )


# --- Per-rule tests ---------------------------------------------------------

class TestRuleDTERoll(unittest.TestCase):
    def test_holds_when_dte_above_threshold(self):
        leg = _make_leg(expiry=date(2025, 9, 19))
        # 60 DTE > 30 threshold.
        self.assertIsNone(rule_dte_roll(leg, _make_state(
            "2025-07-21", 580.0, 580.0, 580.0, 5.0,
        )))

    def test_fires_at_threshold(self):
        leg = _make_leg(expiry=date(2025, 9, 19))
        # 30 DTE: closes.
        self.assertEqual(rule_dte_roll(leg, _make_state(
            "2025-08-20", 580.0, 580.0, 580.0, 5.0,
        )), "dte_roll")
        # 25 DTE: closes.
        self.assertEqual(rule_dte_roll(leg, _make_state(
            "2025-08-25", 580.0, 580.0, 580.0, 5.0,
        )), "dte_roll")

    def test_expiry_overrides(self):
        leg = _make_leg(expiry=date(2025, 9, 19))
        self.assertEqual(rule_dte_roll(leg, _make_state(
            "2025-09-19", 540.0, 580.0, 540.0, 10.0,
        )), "expiry")

    def test_custom_threshold(self):
        leg = _make_leg(expiry=date(2025, 9, 19))
        # 60 DTE, threshold=45: still open.
        self.assertIsNone(rule_dte_roll(leg, _make_state(
            "2025-07-21", 580.0, 580.0, 580.0, 5.0,
        ), dte_threshold=45))
        # 45 DTE, threshold=45: closes.
        self.assertEqual(rule_dte_roll(leg, _make_state(
            "2025-08-05", 580.0, 580.0, 580.0, 5.0,
        ), dte_threshold=45), "dte_roll")


class TestRuleMonetizeRecovery(unittest.TestCase):
    def test_no_drawdown_no_fire(self):
        leg = _make_leg()
        # peak=trough=580 (no DD yet): doesn't fire.
        self.assertIsNone(rule_monetize_recovery(leg, _make_state(
            "2025-07-15", 580.0, 580.0, 580.0, 5.0,
        )))

    def test_shallow_drawdown_doesnt_engage(self):
        leg = _make_leg()
        # peak=580, trough=575 (0.86% DD, < 2% engagement threshold).
        self.assertIsNone(rule_monetize_recovery(leg, _make_state(
            "2025-07-15", 578.0, 580.0, 575.0, 5.0,
        )))

    def test_recovery_at_50pct_fires(self):
        leg = _make_leg()
        # peak=600, trough=540 (10% DD). recovery=0.5 → trigger at 570.
        self.assertIsNone(rule_monetize_recovery(leg, _make_state(
            "2025-07-15", 565.0, 600.0, 540.0, 30.0,
        )))
        self.assertEqual(rule_monetize_recovery(leg, _make_state(
            "2025-07-15", 570.0, 600.0, 540.0, 30.0,
        )), "monetize")
        self.assertEqual(rule_monetize_recovery(leg, _make_state(
            "2025-07-15", 590.0, 600.0, 540.0, 30.0,
        )), "monetize")

    def test_custom_recovery_frac(self):
        leg = _make_leg()
        # 25% recovery: peak=600, trough=540, trigger=540+0.25*60=555
        self.assertEqual(rule_monetize_recovery(leg, _make_state(
            "2025-07-15", 555.0, 600.0, 540.0, 30.0,
        ), recovery_frac=0.25), "monetize")
        self.assertIsNone(rule_monetize_recovery(leg, _make_state(
            "2025-07-15", 550.0, 600.0, 540.0, 30.0,
        ), recovery_frac=0.25))

    def test_min_hold_days_guard_blocks_immature_leg(self):
        # Leg opened 5 days ago — too young for default min_hold_days=14.
        leg = _make_leg(open_date=pd.Timestamp("2025-07-10"), premium_paid=10.0)
        # Recovery condition is met (spot 570, peak 600, trough 540, recovery=570).
        # But leg only 5 days old → blocked.
        self.assertIsNone(rule_monetize_recovery(leg, _make_state(
            "2025-07-15", 570.0, 600.0, 540.0, 30.0,
        )))
        # Same scenario but leg 20 days old → fires.
        leg_mature = _make_leg(open_date=pd.Timestamp("2025-06-20"), premium_paid=10.0)
        self.assertEqual(rule_monetize_recovery(leg_mature, _make_state(
            "2025-07-15", 570.0, 600.0, 540.0, 30.0,
        )), "monetize")

    def test_min_hold_days_can_be_disabled(self):
        leg = _make_leg(open_date=pd.Timestamp("2025-07-14"), premium_paid=10.0)
        # 1 day old, but min_hold_days=0 → fires (still profitable).
        self.assertEqual(rule_monetize_recovery(leg, _make_state(
            "2025-07-15", 570.0, 600.0, 540.0, 30.0,
        ), min_hold_days=0), "monetize")

    def test_min_profit_guard_blocks_loss_position(self):
        # Mature leg (20 days), recovery condition met, but leg is at a loss.
        # Premium paid was $10, current MV $5 per share → blocked.
        leg = _make_leg(open_date=pd.Timestamp("2025-06-20"), premium_paid=10.0)
        self.assertIsNone(rule_monetize_recovery(leg, _make_state(
            "2025-07-15", 570.0, 600.0, 540.0, mv_per_share=5.0,
        )))
        # Same leg, MV ≥ premium → fires.
        self.assertEqual(rule_monetize_recovery(leg, _make_state(
            "2025-07-15", 570.0, 600.0, 540.0, mv_per_share=10.0,
        )), "monetize")

    def test_min_profit_mult_above_1(self):
        # min_profit_mult=1.5 → require MV >= 1.5 × premium.
        leg = _make_leg(open_date=pd.Timestamp("2025-06-20"), premium_paid=10.0)
        self.assertIsNone(rule_monetize_recovery(leg, _make_state(
            "2025-07-15", 570.0, 600.0, 540.0, mv_per_share=14.0,
        ), min_profit_mult=1.5))
        self.assertEqual(rule_monetize_recovery(leg, _make_state(
            "2025-07-15", 570.0, 600.0, 540.0, mv_per_share=15.0,
        ), min_profit_mult=1.5), "monetize")


class TestRuleProfitTake(unittest.TestCase):
    def test_fires_at_multiple(self):
        leg = _make_leg(premium_paid=10.0)
        # leg_mv 25 = 2.5× → no.
        self.assertIsNone(rule_profit_take(leg, _make_state(
            "2025-07-15", 540.0, 580.0, 540.0, 25.0,
        )))
        # leg_mv 30 = 3.0× → yes.
        self.assertEqual(rule_profit_take(leg, _make_state(
            "2025-07-15", 540.0, 580.0, 540.0, 30.0,
        )), "profit_take")

    def test_custom_mult(self):
        leg = _make_leg(premium_paid=10.0)
        # mult=2.0, leg_mv 20 → fires.
        self.assertEqual(rule_profit_take(leg, _make_state(
            "2025-07-15", 540.0, 580.0, 540.0, 20.0,
        ), mult=2.0), "profit_take")

    def test_zero_premium_safe(self):
        leg = _make_leg(premium_paid=0.0)
        self.assertIsNone(rule_profit_take(leg, _make_state(
            "2025-07-15", 540.0, 580.0, 540.0, 5.0,
        )))


# --- Sizing -----------------------------------------------------------------

class TestContractsForNotional(unittest.TestCase):
    def test_basic(self):
        # 540 strike, 1 contract protects $54,000. Notional 540,000 → 10 contracts.
        self.assertEqual(contracts_for_notional(540_000.0, 540.0), 10)

    def test_at_least_one(self):
        self.assertEqual(contracts_for_notional(100.0, 540.0), 1)

    def test_rounds_to_nearest(self):
        # 540 strike, notional 70_000 → 70_000/54_000 = 1.296 → rounds to 1.
        self.assertEqual(contracts_for_notional(70_000.0, 540.0), 1)
        # 540 strike, notional 100_000 → 100_000/54_000 = 1.85 → rounds to 2.
        self.assertEqual(contracts_for_notional(100_000.0, 540.0), 2)


# --- Cache lookup -----------------------------------------------------------

class TestLookupClose(unittest.TestCase):
    def setUp(self):
        self.grid = pd.DataFrame({
            "contract_ticker": ["O:SPY250919P00550000"] * 5,
            "underlying": ["SPY"] * 5,
            "opt_type": ["put"] * 5,
            "expiry": [pd.Timestamp("2025-09-19")] * 5,
            "strike": [550.0] * 5,
            "date": pd.to_datetime([
                "2025-06-02", "2025-06-03", "2025-06-04",
                "2025-06-05", "2025-06-06",
            ]),
            "close": [10.0, 10.5, 11.0, 10.8, 10.6],
        })
        self.idx = build_grid_index(self.grid)

    def test_exact_match(self):
        c = lookup_close(self.idx, "O:SPY250919P00550000",
                         pd.Timestamp("2025-06-04"))
        self.assertEqual(c, 11.0)

    def test_forward_fill_weekend(self):
        # 2025-06-07 is Saturday; should fall back to Friday's close.
        c = lookup_close(self.idx, "O:SPY250919P00550000",
                         pd.Timestamp("2025-06-07"))
        self.assertEqual(c, 10.6)

    def test_pre_data_returns_none(self):
        c = lookup_close(self.idx, "O:SPY250919P00550000",
                         pd.Timestamp("2025-06-01"))
        self.assertIsNone(c)

    def test_unknown_ticker(self):
        c = lookup_close(self.idx, "O:UNKNOWN", pd.Timestamp("2025-06-04"))
        self.assertIsNone(c)


class TestResolveFromCache(unittest.TestCase):
    def setUp(self):
        # Two contracts available, both with data on 2025-06-15.
        rows = []
        for tk, K, exp in [
            ("O:SPY250815P00540000", 540.0, "2025-08-15"),  # ~60 DTE, 6.9% OTM
            ("O:SPY250919P00550000", 550.0, "2025-09-19"),  # 96 DTE, 5.2% OTM
        ]:
            for d in pd.date_range("2025-06-01", "2025-08-15", freq="D"):
                rows.append({
                    "contract_ticker": tk, "underlying": "SPY",
                    "opt_type": "put", "expiry": pd.Timestamp(exp),
                    "strike": K, "date": d, "close": 5.0,
                })
        self.grid = pd.DataFrame(rows)

    def test_picks_closer_to_target(self):
        # Target 90 DTE, 5% OTM @ spot 580 — the 550-strike at 96 DTE wins.
        pick = resolve_from_cache(self.grid, date(2025, 6, 15), 580.0, 90, 0.05)
        self.assertIsNotNone(pick)
        self.assertEqual(pick["strike"], 550.0)

    def test_no_match_outside_window(self):
        # Target 200 DTE — nothing in cache that far out → None.
        pick = resolve_from_cache(self.grid, date(2025, 6, 15), 580.0, 200, 0.05)
        self.assertIsNone(pick)


# --- Integration: simulate_program -----------------------------------------

def _build_synthetic_world():
    """SPY: linear up Jan-Mar, sharp -10% DD in April, V-shape recovery.
    One SPY put contract: 540 strike, expiring 2025-09-19, fully priced.
    """
    dates = pd.date_range("2025-01-01", "2025-09-30", freq="B")
    n = len(dates)
    spy_closes = []
    # Jan-Mar (~65 biz days): linear from 580 to 600.
    # Apr (~22 biz days): linear drop to 540 (-10% DD from 600).
    # May-end: linear recovery back to 605.
    for d in dates:
        if d < pd.Timestamp("2025-04-01"):
            # Day fraction Jan 2 to Mar 31.
            frac = (d - pd.Timestamp("2025-01-02")).days / max(
                (pd.Timestamp("2025-03-31") - pd.Timestamp("2025-01-02")).days, 1)
            spy_closes.append(580.0 + 20.0 * frac)
        elif d < pd.Timestamp("2025-05-01"):
            frac = (d - pd.Timestamp("2025-04-01")).days / max(
                (pd.Timestamp("2025-04-30") - pd.Timestamp("2025-04-01")).days, 1)
            spy_closes.append(600.0 - 60.0 * frac)
        else:
            frac = (d - pd.Timestamp("2025-05-01")).days / max(
                (pd.Timestamp("2025-09-30") - pd.Timestamp("2025-05-01")).days, 1)
            spy_closes.append(540.0 + 65.0 * min(frac, 1.0))
    spy_history = pd.DataFrame({"date": dates, "close": spy_closes})

    # Build a put grid that prices like a simple BSM-ish model: time value
    # decays, intrinsic appears on the drop. We use a heuristic close
    # series that matches a real put's general shape.
    # Target legs: 90-DTE 5%-OTM rebalancing. We provide contracts for two
    # rolling windows so the simulator can pick them.
    grid_rows = []
    for exp_str, K, open_date in [
        ("2025-04-04", 550.0, "2025-01-02"),  # ~63 DTE leg opened Jan
        ("2025-06-20", 555.0, "2025-03-21"),  # Q2 leg
        ("2025-09-19", 555.0, "2025-06-20"),  # Q3 leg
        ("2025-12-19", 555.0, "2025-09-19"),  # Q4 leg
    ]:
        ticker = f"O:SPY{date.fromisoformat(exp_str).strftime('%y%m%d')}P00{int(K*1000):06d}"
        exp_d = pd.Timestamp(exp_str)
        # Live the contract from (open_date - 30) to expiry day.
        live_start = pd.Timestamp(open_date) - pd.Timedelta(days=30)
        live_dates = pd.date_range(live_start, exp_d, freq="B")
        for d in live_dates:
            # Get spot on this date from spy_history.
            spot_row = spy_history[spy_history["date"] <= d]
            if spot_row.empty:
                continue
            spot = float(spot_row.iloc[-1]["close"])
            dte = (exp_d - d).days
            intrinsic = max(K - spot, 0.0)
            # Time value ~ sqrt(DTE) * 0.5 (rough heuristic for low-vol SPY).
            tv = 0.5 * (dte ** 0.5) if dte > 0 else 0.0
            close = intrinsic + tv
            grid_rows.append({
                "contract_ticker": ticker, "underlying": "SPY",
                "opt_type": "put", "expiry": exp_d,
                "strike": K, "date": d, "close": round(close, 2),
            })
    option_grid = pd.DataFrame(grid_rows)
    return spy_history, option_grid


class TestSimulateProgramIntegration(unittest.TestCase):
    def setUp(self):
        self.spy, self.grid = _build_synthetic_world()
        self.policy = HedgePolicy(
            target_dte=90, target_moneyness=0.05,
            notional_protected=55_000.0,  # 1 contract @ 550 strike
        )

    def test_dte_roll_runs_and_tags_close_reasons(self):
        ledger, legs = simulate_program(
            self.policy, "dte_roll", self.spy, self.grid,
            start=date(2025, 1, 2), end=date(2025, 8, 31),
        )
        self.assertGreater(len(legs), 0)
        # Ledger has one row per SPY trading day.
        self.assertEqual(len(ledger), len(self.spy[
            (self.spy["date"] >= pd.Timestamp("2025-01-02"))
            & (self.spy["date"] <= pd.Timestamp("2025-08-31"))
        ]))
        # Cumulative premium paid is monotonically non-decreasing.
        self.assertTrue((ledger["cum_premium_paid"].diff().dropna() >= -1e-9).all())
        # Closed legs should be tagged 'dte_roll' (or 'expiry' if rule didn't
        # fire before expiry).
        for leg in legs:
            if leg.close_date is not None:
                self.assertIn(leg.close_reason, ("dte_roll", "expiry"))

    def test_monetize_fires_during_april_drop(self):
        # Disable min_hold + min_profit guards: the synthetic price model
        # (sqrt-DTE time value) bleeds too aggressively for the leg to be
        # net-profitable by the time SPY recovers to the 50% level. The
        # guards themselves have their own per-rule unit tests; this
        # integration test verifies the recovery trigger fires at all.
        ledger, legs = simulate_program(
            self.policy, "monetize", self.spy, self.grid,
            start=date(2025, 1, 2), end=date(2025, 8, 31),
            rule_kwargs={"min_hold_days": 0, "min_profit_mult": 0.0},
        )
        reasons = [l.close_reason for l in legs if l.close_reason is not None]
        self.assertIn("monetize", reasons,
                      msg=f"No monetize close fired. Reasons seen: {reasons}")

    def test_summarize_run_basic(self):
        ledger, legs = simulate_program(
            self.policy, "dte_roll", self.spy, self.grid,
            start=date(2025, 1, 2), end=date(2025, 8, 31),
        )
        s = summarize_run(ledger, legs, self.policy)
        self.assertIn("annualized_drag_pct", s)
        self.assertGreater(s["n_trades"], 0)
        # Drag is the net cost ÷ notional ÷ years × 100. With premiums burning
        # and modest April payoff, should be positive (net loss).
        self.assertGreater(s["total_premium_paid"], 0)


class TestEpisodePayoffs(unittest.TestCase):
    def _make_ledger(self, mvs: list[tuple[str, float, float]]) -> pd.DataFrame:
        """mvs is list of (date, sleeve_mv, realized_pnl_to_date)."""
        return pd.DataFrame([
            {"date": pd.Timestamp(d), "sleeve_mv": mv, "realized_pnl_to_date": pnl}
            for (d, mv, pnl) in mvs
        ])

    def test_per_episode_value_change(self):
        led = self._make_ledger([
            ("2025-04-01", 100.0, 0.0),  # peak day
            ("2025-04-15", 250.0, 0.0),  # trough day; sleeve gained 150
            ("2025-05-15", 50.0, 200.0),  # recover; realized 200, MV decayed back
        ])
        eps = pd.DataFrame([{
            "peak_date": pd.Timestamp("2025-04-01"),
            "peak_close": 600.0,
            "trough_date": pd.Timestamp("2025-04-15"),
            "trough_close": 540.0,
            "decline_pct": -10.0,
            "recover_date": pd.Timestamp("2025-05-15"),
            "recovered": True,
            "duration_days": 44,
        }])
        out = episode_payoffs(led, eps)
        self.assertEqual(out["peak_sleeve_value"].iloc[0], 100.0)
        self.assertEqual(out["trough_sleeve_value"].iloc[0], 250.0)
        # Recover sleeve_value = sleeve_mv + realized = 50 + 200 = 250.
        self.assertEqual(out["recover_sleeve_value"].iloc[0], 250.0)
        # Gain peak→trough = 250 − 100 = 150.
        self.assertEqual(out["sleeve_gain_peak_to_trough"].iloc[0], 150.0)

    def test_no_episodes_returns_passthrough(self):
        led = self._make_ledger([("2025-04-01", 100.0, 0.0)])
        out = episode_payoffs(led, pd.DataFrame())
        self.assertTrue(out.empty)

    def test_recover_date_none(self):
        # Episode that never recovered.
        led = self._make_ledger([
            ("2025-04-01", 100.0, 0.0),
            ("2025-04-15", 300.0, 0.0),
        ])
        eps = pd.DataFrame([{
            "peak_date": pd.Timestamp("2025-04-01"),
            "peak_close": 600.0,
            "trough_date": pd.Timestamp("2025-04-15"),
            "trough_close": 540.0,
            "decline_pct": -10.0,
            "recover_date": pd.NaT,
            "recovered": False,
            "duration_days": 14,
        }])
        out = episode_payoffs(led, eps)
        self.assertEqual(out["peak_sleeve_value"].iloc[0], 100.0)
        self.assertEqual(out["trough_sleeve_value"].iloc[0], 300.0)
        self.assertIsNone(out["recover_sleeve_value"].iloc[0])


class TestCompareRuns(unittest.TestCase):
    def setUp(self):
        spy, grid = _build_synthetic_world()
        self.spy = spy
        self.grid = grid
        self.policy = HedgePolicy(
            target_dte=90, target_moneyness=0.05,
            notional_protected=55_000.0,
        )

    def test_compare_table_shape(self):
        runs = {}
        for rule in ("dte_roll", "monetize", "profit_take_3x"):
            ledger, legs = simulate_program(
                self.policy, rule, self.spy, self.grid,
                start=date(2025, 1, 2), end=date(2025, 8, 31),
            )
            runs[rule] = (ledger, legs)

        # Empty episodes frame: compare_runs should still produce one row per rule.
        cmp = compare_runs(runs, pd.DataFrame(), self.policy)
        self.assertEqual(len(cmp), 3)
        self.assertEqual(set(cmp["rule"]), {"dte_roll", "monetize", "profit_take_3x"})
        for col in ("annualized_drag_pct", "n_trades", "total_episode_payoff_pct"):
            self.assertIn(col, cmp.columns)

    def test_pareto_simple_case(self):
        df = pd.DataFrame([
            {"rule": "A", "annualized_drag_pct": 1.0, "total_episode_payoff_pct": 2.0},  # dominated by C
            {"rule": "B", "annualized_drag_pct": 0.5, "total_episode_payoff_pct": 1.5},  # frontier
            {"rule": "C", "annualized_drag_pct": 0.8, "total_episode_payoff_pct": 3.0},  # frontier
            {"rule": "D", "annualized_drag_pct": 2.0, "total_episode_payoff_pct": 5.0},  # frontier
        ])
        mask = pareto_frontier_mask(df)
        self.assertEqual(mask.tolist(), [False, True, True, True])

    def test_pareto_empty(self):
        mask = pareto_frontier_mask(pd.DataFrame())
        self.assertTrue(mask.empty)


class TestRuleEmpiricalPercentile(unittest.TestCase):
    def _leg(self, open_date="2025-01-05", expiry="2025-07-01",
             strike=550.0, premium=10.0):
        return Leg(
            open_date=pd.Timestamp(open_date),
            ticker="O:SPY250701P00550000",
            underlying="SPY",
            expiry=pd.Timestamp(expiry).date(),
            strike=strike,
            contracts=2,
            premium_paid=premium,
        )

    def _state(self, today="2025-06-15", iv_rank=50.0, leg_mv_ps=12.0):
        return SimState(
            today=pd.Timestamp(today),
            spot=540.0, peak_spot=600.0, trough_spot=520.0,
            leg_mv_per_share=leg_mv_ps,
            leg_total_mv=leg_mv_ps * 2 * 100,
            flatted=False, flatted_at_peak=None,
            iv_rank_today=iv_rank,
        )

    def test_fires_above_rhigh(self):
        leg = self._leg()
        state = self._state(iv_rank=85.0)
        reason = rule_empirical_percentile(leg, state, r_high=80.0, r_low=30.0)
        self.assertEqual(reason, "empirical_pct")

    def test_does_not_fire_below_rhigh(self):
        leg = self._leg()
        state = self._state(iv_rank=70.0)
        reason = rule_empirical_percentile(leg, state, r_high=80.0, r_low=30.0)
        self.assertIsNone(reason)

    def test_expiry_safety(self):
        # Even if rank is in the no-fire band, expiry still closes.
        leg = self._leg(expiry="2025-06-15")
        state = self._state(today="2025-06-15", iv_rank=50.0)
        reason = rule_empirical_percentile(leg, state, r_high=80.0, r_low=30.0)
        self.assertEqual(reason, "expiry")

    def test_nan_rank_does_not_fire(self):
        leg = self._leg()
        state = self._state(iv_rank=float("nan"))
        reason = rule_empirical_percentile(leg, state, r_high=80.0, r_low=30.0)
        self.assertIsNone(reason)

    def test_registered_in_exit_rules(self):
        self.assertIn("empirical_pct", EXIT_RULES)


# --- Integration: simulate_program with empirical_pct ----------------------

def _toy_spy(days=90, start_close=600.0):
    dates = pd.bdate_range("2025-01-02", periods=days)
    closes = [start_close - 0.3 * i for i in range(days)]  # gentle drift
    return pd.DataFrame({"date": dates, "close": closes})


def _toy_grid(spy):
    """One synthetic SPY put covering the whole window, slowly appreciating
    as spot drifts down. Strike = start_close * 0.95, expiry = end + 30d."""
    end = spy["date"].iloc[-1]
    expiry = (end + pd.Timedelta(days=30)).date()
    strike = round(spy.iloc[0]["close"] * 0.95 / 5.0) * 5.0
    rows = []
    for _, r in spy.iterrows():
        close = max(strike - r["close"], 0.0) + 5.0  # intrinsic + 5
        rows.append({
            "date": r["date"], "contract_ticker": "O:TOY_PUT",
            "underlying": "SPY", "opt_type": "put",
            "expiry": pd.Timestamp(expiry), "strike": strike, "close": close,
        })
    return pd.DataFrame(rows)


def _toy_rank_series(spy, ranks):
    """ranks: list of floats, same length as spy."""
    return pd.DataFrame({"date": spy["date"], "rank": ranks})


class TestSimulateEmpiricalPct(unittest.TestCase):
    def test_simulate_empirical_pct_closes_at_high_rank_reopens_at_low(self):
        spy = _toy_spy(days=90)
        grid = _toy_grid(spy)
        # First 30 days low rank (open), next 20 high (close + flatted),
        # next 40 low (re-enter).
        ranks = [20.0] * 30 + [85.0] * 20 + [25.0] * 40
        iv_ranks = _toy_rank_series(spy, ranks)
        policy = HedgePolicy(target_dte=30, target_moneyness=0.05,
                             notional_protected=100_000.0)
        ledger, legs = simulate_program(
            policy, "empirical_pct", spy, grid,
            start=spy["date"].iloc[0].date(),
            end=spy["date"].iloc[-1].date(),
            iv_rank_series=iv_ranks,
            rule_kwargs={"r_high": 80.0, "r_low": 30.0},
        )
        self.assertTrue(
            any(l.close_reason == "empirical_pct" for l in legs),
            msg="expected at least one empirical_pct close",
        )
        # Re-entry: at least 2 distinct legs (close + reopen).
        self.assertGreaterEqual(len(legs), 2)


if __name__ == "__main__":
    unittest.main()

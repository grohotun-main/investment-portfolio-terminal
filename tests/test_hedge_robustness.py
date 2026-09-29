"""Tests for parsers/hedge_robustness.py — sweep + walk-forward layers."""
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
from parsers.hedge_robustness import (  # noqa: E402
    pick_optimal_param,
    run_walk_forward,
    summarize_walk_forward,
    sweep_parameter,
    walk_forward_compare_all,
    walk_forward_windows,
)
from tests.test_hedge_exit_simulator import (  # noqa: E402
    _build_synthetic_world,
    _toy_grid,
    _toy_spy,
)


class TestWalkForwardWindows(unittest.TestCase):
    def test_basic(self):
        ws = walk_forward_windows(
            date(2025, 1, 1), date(2025, 12, 31),
            window_days=90, stride_days=30,
        )
        # Windows: (Jan 1, Apr 1), (Jan 31, May 1), (Mar 2, May 31),
        #   (Apr 1, Jun 30), (May 1, Jul 30), (May 31, Aug 29),
        #   (Jun 30, Sep 28), (Jul 30, Oct 28), (Aug 29, Nov 27),
        #   (Sep 28, Dec 27), (Oct 28, Jan 26 — past end, drop)
        # Last window must end ≤ 2025-12-31.
        for w_start, w_end in ws:
            self.assertGreaterEqual(w_end, w_start)
            self.assertLessEqual(w_end, date(2025, 12, 31))
        self.assertGreater(len(ws), 0)

    def test_window_too_long(self):
        ws = walk_forward_windows(
            date(2025, 1, 1), date(2025, 3, 1),
            window_days=365, stride_days=60,
        )
        self.assertEqual(ws, [])


class TestSweepParameter(unittest.TestCase):
    def setUp(self):
        self.spy, self.grid = _build_synthetic_world()
        self.policy = HedgePolicy(
            target_dte=90, target_moneyness=0.05,
            notional_protected=55_000.0,
        )

    def test_dte_roll_sweep(self):
        df = sweep_parameter(
            "dte_roll", "dte_threshold", [15, 30, 45],
            policy=self.policy, spy_history=self.spy, option_grid=self.grid,
            start=date(2025, 1, 2), end=date(2025, 8, 31),
        )
        self.assertEqual(len(df), 3)
        self.assertEqual(list(df["param_value"]), [15, 30, 45])
        for col in ("drag_pct", "sum_payoff_pct", "n_trades"):
            self.assertIn(col, df.columns)
        # As DTE threshold rises, we roll earlier, so n_trades is monotonically
        # non-decreasing (closer thresholds = more rolls in the same window).
        n_trades = df["n_trades"].tolist()
        self.assertTrue(n_trades[0] <= n_trades[1] <= n_trades[2])

    def test_profit_take_sweep(self):
        df = sweep_parameter(
            "profit_take_3x", "mult", [1.5, 3.0, 5.0],
            policy=self.policy, spy_history=self.spy, option_grid=self.grid,
            start=date(2025, 1, 2), end=date(2025, 8, 31),
        )
        self.assertEqual(len(df), 3)


class TestPickOptimalParam(unittest.TestCase):
    def _sweep(self, rows):
        # Build a minimal sweep DataFrame for the picker. Columns match the
        # output of sweep_parameter.
        return pd.DataFrame([
            {"param_value": v, "drag_pct": d, "sum_payoff_pct": p,
             "mean_payoff_pct": p / 5, "payoff_per_dollar_drag": ppd,
             "n_trades": 1}
            for (v, d, p, ppd) in rows
        ])

    def test_picks_max_criterion(self):
        # value=5 has highest pay/$drag → optimum.
        df = self._sweep([
            (2.0, 3.0, 6.0, 2.0),
            (3.0, 2.5, 5.0, 2.0),
            (5.0, 1.7, 9.7, 5.7),  # winner
            (8.0, 1.2, 4.0, 3.3),
        ])
        pick = pick_optimal_param(df)
        self.assertEqual(pick["optimal_value"], 5.0)
        self.assertEqual(pick["n_candidates"], 4)
        self.assertAlmostEqual(pick["optimal_row"]["payoff_per_dollar_drag"], 5.7)

    def test_filters_non_positive_drag(self):
        # value=3 wins on raw pay/$drag (because drag=0 → undefined or
        # negative pay/$drag in practice), but with the positive-drag filter
        # we should pick value=5.
        df = self._sweep([
            (3.0, 0.0, 4.0, 999.0),   # drag=0 — filtered out
            (5.0, 1.0, 5.0, 5.0),     # winner after filter
            (8.0, 2.0, 6.0, 3.0),
        ])
        pick = pick_optimal_param(df)
        self.assertEqual(pick["optimal_value"], 5.0)
        self.assertEqual(pick["n_candidates"], 2)  # only positive-drag rows

    def test_falls_back_when_no_positive_drag(self):
        # All drag ≤ 0 → fall back to all rows; pick max criterion.
        df = self._sweep([
            (2.0, -1.0, 3.0, 1.5),
            (5.0, -2.0, 4.0, 2.0),  # winner among all rows
        ])
        pick = pick_optimal_param(df)
        self.assertEqual(pick["optimal_value"], 5.0)

    def test_empty_sweep_raises(self):
        with self.assertRaises(ValueError):
            pick_optimal_param(pd.DataFrame())

    def test_unknown_criterion_raises(self):
        df = self._sweep([(3.0, 1.0, 5.0, 5.0)])
        with self.assertRaises(ValueError):
            pick_optimal_param(df, criterion="nonexistent_metric")

    def test_custom_criterion(self):
        # Pick by sum_payoff_pct instead of pay/$drag.
        df = self._sweep([
            (3.0, 1.0, 10.0, 10.0),
            (5.0, 1.5, 12.0, 8.0),
            (8.0, 2.0, 15.0, 7.5),
        ])
        pick = pick_optimal_param(df, criterion="sum_payoff_pct")
        self.assertEqual(pick["optimal_value"], 8.0)


class TestWalkForwardRun(unittest.TestCase):
    def setUp(self):
        self.spy, self.grid = _build_synthetic_world()
        self.policy = HedgePolicy(
            target_dte=90, target_moneyness=0.05,
            notional_protected=55_000.0,
        )

    def test_run_walk_forward_basic(self):
        df = run_walk_forward(
            "dte_roll", self.policy, self.spy, self.grid,
            start=date(2025, 1, 2), end=date(2025, 9, 30),
            window_days=120, stride_days=60,
        )
        self.assertGreater(len(df), 0)
        for col in ("window_start", "window_end", "drag_pct", "sum_payoff_pct"):
            self.assertIn(col, df.columns)

    def test_compare_all_returns_all_rules(self):
        df = walk_forward_compare_all(
            self.policy, self.spy, self.grid,
            start=date(2025, 1, 2), end=date(2025, 9, 30),
            window_days=120, stride_days=60,
            rule_kwargs_by_rule={
                "monetize": {"min_hold_days": 0, "min_profit_mult": 0.0},
            },
        )
        self.assertEqual(set(df["rule"]),
                         {"dte_roll", "monetize", "profit_take_3x",
                          "empirical_pct"})

    def test_summarize_walk_forward_shape(self):
        df = walk_forward_compare_all(
            self.policy, self.spy, self.grid,
            start=date(2025, 1, 2), end=date(2025, 9, 30),
            window_days=120, stride_days=60,
            rule_kwargs_by_rule={
                "monetize": {"min_hold_days": 0, "min_profit_mult": 0.0},
            },
        )
        s = summarize_walk_forward(df)
        # 4 rules → 4 rows.
        self.assertEqual(len(s), 4)
        for col in ("drag_median", "drag_p10", "drag_p90",
                    "payoff_median", "payoff_p10", "payoff_p90"):
            self.assertIn(col, s.columns)
        # All p10 ≤ median ≤ p90.
        for _, r in s.iterrows():
            self.assertLessEqual(r["drag_p10"], r["drag_median"] + 1e-9)
            self.assertLessEqual(r["drag_median"], r["drag_p90"] + 1e-9)


class TestWalkForwardCompareAllEmpiricalPct(unittest.TestCase):
    def test_walk_forward_compare_all_runs_empirical_pct(self):
        """With iv_rank_series supplied, walk_forward_compare_all should
        successfully iterate over empirical_pct without errors."""
        spy = _toy_spy(days=120)
        grid = _toy_grid(spy)
        policy = HedgePolicy(target_dte=30, target_moneyness=0.05,
                             notional_protected=100_000.0)
        ranks = pd.DataFrame({
            "date": spy["date"],
            "rank": [50.0] * len(spy),
        })
        out = walk_forward_compare_all(
            policy, spy, grid,
            start=spy["date"].iloc[0].date(),
            end=spy["date"].iloc[-1].date(),
            window_days=60, stride_days=30,
            rule_kwargs_by_rule={
                "empirical_pct": {"r_high": 80.0, "r_low": 30.0},
            },
            iv_rank_series=ranks,
        )
        # empirical_pct must appear in the long-format output.
        self.assertIn("empirical_pct", set(out["rule"]))


if __name__ == "__main__":
    unittest.main()

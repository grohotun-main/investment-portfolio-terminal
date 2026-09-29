"""Tests for parsers/hedge_scenarios.py."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from hedge_scenarios import (  # noqa: E402
    invert_portfolio_drawdown,
    per_name_scenario_drops,
    classify_holdings,
    HoldingClass,
    equity_names_without_crash_history,
)


class ClassifyHoldingsTests(unittest.TestCase):
    def test_sgov_is_cash_equivalent(self):
        self.assertEqual(classify_holdings("SGOV"), HoldingClass.CASH_EQUIVALENT)

    def test_money_market_is_cash_equivalent(self):
        self.assertEqual(classify_holdings("SPAXX"), HoldingClass.CASH_EQUIVALENT)

    def test_treasury_etfs_are_cash_equivalent(self):
        for t in ("BIL", "SHV", "TLH"):
            self.assertEqual(classify_holdings(t), HoldingClass.CASH_EQUIVALENT, t)

    def test_spy_is_equity(self):
        self.assertEqual(classify_holdings("SPY"), HoldingClass.EQUITY)

    def test_single_name_is_equity(self):
        self.assertEqual(classify_holdings("NVDA"), HoldingClass.EQUITY)


class InvertDrawdownTests(unittest.TestCase):
    def test_all_equity_portfolio_passes_drawdown_through(self):
        # 100% equity, weighted beta = 1.0, drawdown -10% → SPY drop -10%
        weights_eq = pd.Series({"SPY": 1.0})
        betas = {"SPY": 1.0}
        spy_drop = invert_portfolio_drawdown(
            target_drawdown=-0.10, equity_weight=1.0,
            weights_in_equity=weights_eq, crash_betas=betas,
        )
        self.assertAlmostEqual(spy_drop, -0.10, places=4)

    def test_half_cash_doubles_implied_spy_drop(self):
        # 50% equity (all in SPY at beta=1), 50% cash, drawdown -10% →
        # equity slice must drop -20% → SPY drop -20%
        weights_eq = pd.Series({"SPY": 1.0})
        betas = {"SPY": 1.0}
        spy_drop = invert_portfolio_drawdown(
            target_drawdown=-0.10, equity_weight=0.5,
            weights_in_equity=weights_eq, crash_betas=betas,
        )
        self.assertAlmostEqual(spy_drop, -0.20, places=4)

    def test_high_beta_dampens_implied_spy_drop(self):
        # 100% equity, all NVDA at beta=2, drawdown -10% →
        # SPY drop = -10% / 2 = -5%
        weights_eq = pd.Series({"NVDA": 1.0})
        betas = {"NVDA": 2.0}
        spy_drop = invert_portfolio_drawdown(
            target_drawdown=-0.10, equity_weight=1.0,
            weights_in_equity=weights_eq, crash_betas=betas,
        )
        self.assertAlmostEqual(spy_drop, -0.05, places=4)

    def test_zero_equity_weight_returns_nan(self):
        # All cash — no equity drawdown can match a non-zero portfolio drawdown.
        spy_drop = invert_portfolio_drawdown(
            target_drawdown=-0.10, equity_weight=0.0,
            weights_in_equity=pd.Series(dtype=float), crash_betas={},
        )
        self.assertTrue(np.isnan(spy_drop))

    def test_no_history_name_excluded_and_remainder_renormalized(self):
        # A name with no crash history (NaN beta — e.g. a listing newer than
        # the most recent crash window) is dropped and the remaining equity
        # weights renormalized, rather than nuking the whole result. Here SPY
        # (beta 1.0) and a no-history name split 50/50; dropping the no-history
        # name renormalizes SPY to weight 1.0 → weighted beta 1.0 → SPY drop
        # -10%. The exclusion is surfaced via equity_names_without_crash_history
        # (caller renders a caption), so this is a disclosed transform, not a
        # hidden one — superseding the earlier strict-propagation contract.
        weights_eq = pd.Series({"SPY": 0.5, "NEWIPO": 0.5})
        betas = {"SPY": 1.0, "NEWIPO": float("nan")}
        spy_drop = invert_portfolio_drawdown(
            target_drawdown=-0.10, equity_weight=1.0,
            weights_in_equity=weights_eq, crash_betas=betas,
        )
        self.assertAlmostEqual(spy_drop, -0.10, places=4)

    def test_renormalization_weights_only_the_kept_names(self):
        # SPY(beta 1.0, w 0.25) + NVDA(beta 2.0, w 0.25) + NEWIPO(nan, w 0.50).
        # Drop NEWIPO; renormalize kept weights to 0.5/0.5 →
        # weighted beta = 0.5*1.0 + 0.5*2.0 = 1.5. drawdown -15%, eq weight 1.0
        # → SPY drop = -0.15 / 1.5 = -0.10.
        weights_eq = pd.Series({"SPY": 0.25, "NVDA": 0.25, "NEWIPO": 0.50})
        betas = {"SPY": 1.0, "NVDA": 2.0, "NEWIPO": float("nan")}
        spy_drop = invert_portfolio_drawdown(
            target_drawdown=-0.15, equity_weight=1.0,
            weights_in_equity=weights_eq, crash_betas=betas,
        )
        self.assertAlmostEqual(spy_drop, -0.10, places=4)

    def test_all_names_lack_history_returns_nan(self):
        # If NO equity name has a usable crash beta, the SPY drop is genuinely
        # undefined — keep returning NaN so the UI shows the failure.
        weights_eq = pd.Series({"AAA": 0.5, "BBB": 0.5})
        betas = {"AAA": float("nan"), "BBB": float("nan")}
        spy_drop = invert_portfolio_drawdown(
            target_drawdown=-0.10, equity_weight=1.0,
            weights_in_equity=weights_eq, crash_betas=betas,
        )
        self.assertTrue(np.isnan(spy_drop))


class EquityNamesWithoutCrashHistoryTests(unittest.TestCase):
    def test_reports_nan_beta_name_with_its_equity_weight(self):
        weights_eq = pd.Series({"SPY": 0.7, "DRAM": 0.2, "NVDA": 0.1})
        betas = {"SPY": 1.0, "DRAM": float("nan"), "NVDA": 1.5}
        out = equity_names_without_crash_history(
            weights_in_equity=weights_eq, crash_betas=betas)
        self.assertEqual(out, [("DRAM", 0.2)])

    def test_reports_names_absent_from_betas(self):
        weights_eq = pd.Series({"SPY": 0.6, "MISSING": 0.4})
        betas = {"SPY": 1.0}  # MISSING has no entry at all
        out = equity_names_without_crash_history(
            weights_in_equity=weights_eq, crash_betas=betas)
        self.assertEqual(out, [("MISSING", 0.4)])

    def test_sorted_by_weight_descending(self):
        weights_eq = pd.Series({"A": 0.1, "B": 0.3, "SPY": 0.6})
        betas = {"SPY": 1.0, "A": float("nan"), "B": float("nan")}
        out = equity_names_without_crash_history(
            weights_in_equity=weights_eq, crash_betas=betas)
        self.assertEqual([t for t, _ in out], ["B", "A"])

    def test_empty_when_every_name_has_history(self):
        weights_eq = pd.Series({"SPY": 0.5, "NVDA": 0.5})
        betas = {"SPY": 1.0, "NVDA": 1.5}
        out = equity_names_without_crash_history(
            weights_in_equity=weights_eq, crash_betas=betas)
        self.assertEqual(out, [])


class PerNameDropsTests(unittest.TestCase):
    def test_scales_by_beta(self):
        out = per_name_scenario_drops(
            spy_drop=-0.20,
            crash_betas={"SPY": 1.0, "NVDA": 1.5, "AAPL": 1.1},
        )
        self.assertAlmostEqual(out["SPY"], -0.20, places=4)
        self.assertAlmostEqual(out["NVDA"], -0.30, places=4)
        self.assertAlmostEqual(out["AAPL"], -0.22, places=4)

    def test_nan_beta_propagates(self):
        out = per_name_scenario_drops(
            spy_drop=-0.20,
            crash_betas={"X": float("nan")},
        )
        self.assertTrue(np.isnan(out["X"]))


if __name__ == "__main__":
    unittest.main()

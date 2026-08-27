"""
Tests for the pure-function math in parsers/compute_twr.py:
  - xirr                   (Newton + bisection IRR solver)
  - modified_dietz_period  (one-period flow-weighted return)
  - link_returns           (chain monthly returns)
  - annualize              (period-return → annualized rate)
  - monthly_navs           (per-account NAV with forward-fill across gaps)

Pipeline-level orchestration (compute_monthly_twr, compute_portfolio_twr,
compute_account_irr, compute_portfolio_irr) requires positions+transactions
fixtures and is left to manual integration testing. The math primitives
covered here are where regressions silently corrupt user-visible numbers.

Run from phase1_build/ with:
    py -m unittest discover tests
"""
import contextlib
import io
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

# Make parsers/ importable.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

# compute_twr.py imports config_local at top-level; the project's
# config_local.py exists alongside app.py (also gitignored). Add the root
# to the path so that import resolves.
sys.path.insert(0, str(ROOT))

import compute_twr as ct  # noqa: E402


# ---------------------------------------------------------------------------
# xirr
# ---------------------------------------------------------------------------
class TestXirr(unittest.TestCase):
    def test_one_year_ten_percent(self) -> None:
        # Deposit $100 today, withdraw $110 in exactly one year ⇒ IRR = 10%.
        dates = [pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")]
        cf = [-100.0, 110.0]
        self.assertAlmostEqual(ct.xirr(cf, dates), 0.10, places=4)

    def test_two_year_ten_percent_compounded(self) -> None:
        # $100 → $121 over 2 years is 10% annualized (1.10² = 1.21).
        # Use 2025-2027 (two non-leap years) so the span is exactly 730 days
        # = 2.0 on xirr's 365-day basis. A leap-day-crossing span returns a
        # rate ~0.0001 lower because the period is slightly longer.
        dates = [pd.Timestamp("2025-01-01"), pd.Timestamp("2027-01-01")]
        cf = [-100.0, 121.0]
        self.assertAlmostEqual(ct.xirr(cf, dates), 0.10, places=4)

    def test_two_year_negative_return(self) -> None:
        # Losing 19% over 2 years (1.0 → 0.81) is -10% annualized.
        dates = [pd.Timestamp("2025-01-01"), pd.Timestamp("2027-01-01")]
        cf = [-100.0, 81.0]
        self.assertAlmostEqual(ct.xirr(cf, dates), -0.10, places=4)

    def test_multiple_contributions(self) -> None:
        # Three equal deposits over a year with a known terminal value.
        # Verified by reconstructing NPV at the returned rate.
        dates = [pd.Timestamp("2025-01-01"),
                 pd.Timestamp("2025-04-01"),
                 pd.Timestamp("2025-07-01"),
                 pd.Timestamp("2026-01-01")]
        cf = [-100.0, -100.0, -100.0, 330.0]
        r = ct.xirr(cf, dates)
        self.assertTrue(np.isfinite(r))
        # NPV at the returned rate must round to ~0.
        d0 = min(dates)
        t = np.array([(d - d0).days / 365.0 for d in dates])
        npv = float(np.sum(np.array(cf) / (1.0 + r) ** t))
        self.assertAlmostEqual(npv, 0.0, places=3)

    def test_too_few_cashflows(self) -> None:
        self.assertTrue(np.isnan(ct.xirr([], [])))
        self.assertTrue(np.isnan(ct.xirr([-100.0], [pd.Timestamp("2025-01-01")])))

    def test_single_sign_returns_nan(self) -> None:
        # All-positive or all-negative cashflows have no IRR.
        dates = [pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")]
        self.assertTrue(np.isnan(ct.xirr([-100.0, -50.0], dates)))
        self.assertTrue(np.isnan(ct.xirr([100.0, 50.0], dates)))

    def test_zero_npv_at_returned_rate(self) -> None:
        # Property test: at the returned rate, NPV ≈ 0 for any solvable case.
        dates = [pd.Timestamp("2020-03-01"),
                 pd.Timestamp("2021-06-15"),
                 pd.Timestamp("2023-11-30"),
                 pd.Timestamp("2026-05-17")]
        cf = [-50_000.0, -25_000.0, 10_000.0, 95_000.0]
        r = ct.xirr(cf, dates)
        self.assertTrue(np.isfinite(r))
        d0 = min(dates)
        t = np.array([(d - d0).days / 365.0 for d in dates])
        npv = float(np.sum(np.array(cf) / (1.0 + r) ** t))
        self.assertAlmostEqual(npv, 0.0, places=2)

    def test_nan_cashflow_returns_nan(self) -> None:
        # A NaN cashflow (an in-kind transfer the parser couldn't price) must
        # yield an undefined IRR, NOT the -0.9999 bisection floor. The
        # same-sign guard uses np.sign, and np.sign(nan) == np.sign(nan) is
        # False, which silently collapsed bisection onto its lower bound.
        dates = [pd.Timestamp("2025-01-01"),
                 pd.Timestamp("2025-06-01"),
                 pd.Timestamp("2026-01-01")]
        cf = [-100.0, float("nan"), 150.0]
        self.assertTrue(np.isnan(ct.xirr(cf, dates)))


# ---------------------------------------------------------------------------
# modified_dietz_period
# ---------------------------------------------------------------------------
class TestModifiedDietz(unittest.TestCase):
    def setUp(self) -> None:
        self.start = pd.Timestamp("2026-01-01")
        self.end   = pd.Timestamp("2026-02-01")  # 31-day period

    def _flows(self, *rows) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=["settlement_date", "amount"]) \
            if rows else pd.DataFrame(columns=["settlement_date", "amount"])

    def test_no_flows_pure_growth(self) -> None:
        # NAV 100 → 110 with no flows ⇒ 10%.
        r = ct.modified_dietz_period(100.0, 110.0, self._flows(),
                                     self.start, self.end)
        self.assertAlmostEqual(r, 0.10, places=6)

    def test_flow_at_start_full_weight(self) -> None:
        # +$50 at period start gets full weight in the denominator.
        # Start NAV 100 + flow 50 at t=0 ⇒ denom 150; end NAV 165.
        # Return = (165 - 100 - 50) / 150 = 15/150 = 0.10
        flows = self._flows((self.start, 50.0))
        r = ct.modified_dietz_period(100.0, 165.0, flows,
                                     self.start, self.end)
        self.assertAlmostEqual(r, 0.10, places=6)

    def test_flow_at_end_minimal_weight(self) -> None:
        # +$50 at period end (t=T) gets ~0 weight in denominator.
        # Start 100, end 160 with $50 dropped at end:
        # numerator = 160 - 100 - 50 = 10
        # denom    ≈ 100 + 50 × 0 = 100
        # return   ≈ 0.10
        flows = self._flows((self.end, 50.0))
        r = ct.modified_dietz_period(100.0, 160.0, flows,
                                     self.start, self.end)
        self.assertAlmostEqual(r, 0.10, places=6)

    def test_flow_midway_half_weight(self) -> None:
        # +$31 dropped exactly halfway through a 31-day period.
        # The Dietz weight for a flow on day t over period T is w=(T-t)/T.
        # Halfway: w = 0.5. Denom = 100 + 31 × 0.5 = 115.5.
        # Pick end NAV so numerator computes cleanly: 100 + 31 + 11.55 = 142.55
        # numerator = 142.55 - 100 - 31 = 11.55, return = 11.55/115.5 = 0.10
        mid = self.start + pd.Timedelta(days=15)  # day 15 of 31, w = 16/31
        # Recompute with the actual mid weight:
        T = (self.end - self.start).days
        t = (mid - self.start).days
        w = (T - t) / T
        denom = 100.0 + 50.0 * w
        end_nav = 100.0 + 50.0 + 0.10 * denom  # arrange for 10% return
        flows = self._flows((mid, 50.0))
        r = ct.modified_dietz_period(100.0, end_nav, flows,
                                     self.start, self.end)
        self.assertAlmostEqual(r, 0.10, places=6)

    def test_nan_v_begin_returns_nan(self) -> None:
        r = ct.modified_dietz_period(np.nan, 100.0, self._flows(),
                                     self.start, self.end)
        self.assertTrue(np.isnan(r))

    def test_zero_v_begin_no_flows_returns_nan(self) -> None:
        # Starting from zero with no flows is undefined.
        r = ct.modified_dietz_period(0.0, 0.0, self._flows(),
                                     self.start, self.end)
        self.assertTrue(np.isnan(r))

    def test_non_positive_period_returns_nan(self) -> None:
        r = ct.modified_dietz_period(100.0, 110.0, self._flows(),
                                     self.start, self.start)
        self.assertTrue(np.isnan(r))

    def test_zero_denom_returns_nan(self) -> None:
        # Start with $100, take out $100 at t=0 → denom = 0.
        flows = self._flows((self.start, -100.0))
        r = ct.modified_dietz_period(100.0, 0.0, flows,
                                     self.start, self.end)
        self.assertTrue(np.isnan(r))


# ---------------------------------------------------------------------------
# link_returns
# ---------------------------------------------------------------------------
class TestLinkReturns(unittest.TestCase):
    def test_empty(self) -> None:
        self.assertTrue(np.isnan(ct.link_returns(pd.Series([], dtype=float))))

    def test_all_nan(self) -> None:
        self.assertTrue(np.isnan(ct.link_returns(pd.Series([np.nan, np.nan]))))

    def test_single_value(self) -> None:
        self.assertAlmostEqual(ct.link_returns(pd.Series([0.05])), 0.05, places=12)

    def test_two_months_compound(self) -> None:
        # 5% then 5% ⇒ (1.05)² − 1 = 0.1025.
        self.assertAlmostEqual(ct.link_returns(pd.Series([0.05, 0.05])),
                               0.1025, places=12)

    def test_skips_nan(self) -> None:
        # NaNs ignored: chaining {0.10, NaN, -0.05} = 1.10 × 0.95 − 1.
        r = ct.link_returns(pd.Series([0.10, np.nan, -0.05]))
        self.assertAlmostEqual(r, 1.10 * 0.95 - 1.0, places=12)


# ---------------------------------------------------------------------------
# annualize
# ---------------------------------------------------------------------------
class TestAnnualize(unittest.TestCase):
    def test_exactly_one_year(self) -> None:
        # 12 months of total 10% should annualize to 10%.
        self.assertAlmostEqual(ct.annualize(0.10, 12), 0.10, places=12)

    def test_two_years_compound(self) -> None:
        # 21% over 24 months ⇒ 10% annualized (1.21^(1/2) − 1).
        self.assertAlmostEqual(ct.annualize(0.21, 24), 0.10, places=6)

    def test_six_months(self) -> None:
        # 10% over 6 months ⇒ (1.10)^2 − 1 = 21% annualized.
        self.assertAlmostEqual(ct.annualize(0.10, 6), 0.21, places=6)

    def test_nan_or_zero_months(self) -> None:
        self.assertTrue(np.isnan(ct.annualize(np.nan, 12)))
        self.assertTrue(np.isnan(ct.annualize(0.10, 0)))
        self.assertTrue(np.isnan(ct.annualize(0.10, -3)))


# ---------------------------------------------------------------------------
# monthly_navs (forward-fill behavior)
# ---------------------------------------------------------------------------
class TestMonthlyNavs(unittest.TestCase):
    def _positions(self, rows) -> pd.DataFrame:
        # rows: list of (account_id, statement_date_str, market_value).
        # Always returns a DataFrame with the canonical schema, even when
        # `rows` is empty — matches what pd.read_csv("positions.csv") would
        # produce against a header-only CSV, which is the realistic
        # "no positions yet" input shape.
        if not rows:
            return pd.DataFrame({
                "account_id":     pd.Series(dtype=str),
                "statement_date": pd.Series(dtype="datetime64[ns]"),
                "market_value":   pd.Series(dtype=float),
            })
        return pd.DataFrame(
            [{"account_id": a,
              "statement_date": pd.Timestamp(d),
              "market_value": float(v)}
             for a, d, v in rows]
        )

    def test_empty(self) -> None:
        out = ct.monthly_navs(self._positions([]))
        self.assertTrue(out.empty)

    def test_single_account_single_month(self) -> None:
        pos = self._positions([("A1", "2026-01-31", 100.0)])
        out = ct.monthly_navs(pos)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["account_id"], "A1")
        self.assertAlmostEqual(out.iloc[0]["nav"], 100.0)

    def test_forward_fills_gap(self) -> None:
        # Account A has Jan + March statements; Feb is missing.
        # monthly_navs should insert a Feb row holding January's $100.
        pos = self._positions([
            ("A", "2026-01-31", 100.0),
            ("A", "2026-03-31", 110.0),
        ])
        out = ct.monthly_navs(pos).sort_values("month").reset_index(drop=True)
        self.assertEqual(len(out), 3)
        months = [str(m) for m in out["month"].tolist()]
        self.assertEqual(months, ["2026-01", "2026-02", "2026-03"])
        # Feb's NAV is forward-filled from January.
        feb = out[out["month"].astype(str) == "2026-02"].iloc[0]
        self.assertAlmostEqual(feb["nav"], 100.0)

    def test_aggregates_multiple_positions_same_month(self) -> None:
        # Two positions in the same account/month should sum.
        pos = self._positions([
            ("A", "2026-01-31",  60.0),
            ("A", "2026-01-31",  40.0),
        ])
        out = ct.monthly_navs(pos)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(out.iloc[0]["nav"], 100.0)

    def test_trailing_gap_fills_lagging_account_to_global_frontier(self) -> None:
        # Account A has Jan-only; account B establishes Feb as the global
        # statement frontier. A lags the frontier by one month, so its last
        # known NAV is carried forward into Feb as a non-real (forward-filled)
        # row — the same trailing carry-forward the holdings-side
        # monthly_normalize already applies. Without this, A silently drops
        # out of the Feb portfolio NAV sum (WSA-1: a phantom loss on the live
        # month). Months BEFORE an account's debut are still left absent, so B
        # gets no Jan row.
        pos = self._positions([
            ("A", "2026-01-31", 100.0),
            ("B", "2026-02-28", 200.0),
        ])
        out = ct.monthly_navs(pos).sort_values(
            ["account_id", "month"]).reset_index(drop=True)
        self.assertEqual(len(out), 3)
        a = out[out["account_id"] == "A"].sort_values("month")
        b = out[out["account_id"] == "B"]
        self.assertEqual([str(m) for m in a["month"]], ["2026-01", "2026-02"])
        self.assertEqual([str(m) for m in b["month"]], ["2026-02"])
        # A's Feb row is a trailing carry-forward: non-real, holding Jan's NAV.
        a_feb = a[a["month"].astype(str) == "2026-02"].iloc[0]
        self.assertFalse(bool(a_feb["is_real_statement"]))
        self.assertAlmostEqual(a_feb["nav"], 100.0)
        # B is not back-filled into Jan (pre-debut months stay absent).
        self.assertTrue(bool(b.iloc[0]["is_real_statement"]))

    def test_multi_month_gap_uses_most_recent_prior(self) -> None:
        # Jan + April; Feb and Mar both get forward-filled from January's
        # NAV (the most recent prior statement, not chained re-fills).
        pos = self._positions([
            ("A", "2026-01-31",  50.0),
            ("A", "2026-04-30", 100.0),
        ])
        out = ct.monthly_navs(pos).sort_values("month").reset_index(drop=True)
        months = [str(m) for m in out["month"]]
        self.assertEqual(months, ["2026-01", "2026-02", "2026-03", "2026-04"])
        self.assertAlmostEqual(
            out[out["month"].astype(str) == "2026-02"].iloc[0]["nav"], 50.0)
        self.assertAlmostEqual(
            out[out["month"].astype(str) == "2026-03"].iloc[0]["nav"], 50.0)


class TestPortfolioTwrFrontierCarry(unittest.TestCase):
    """WSA-1: an account lagging the statement frontier must be carried
    forward into the trailing months at its last known NAV, so the portfolio
    NAV sum stays whole and the live month doesn't book a phantom loss."""

    def _positions(self, rows: list[tuple[str, str, float]]) -> pd.DataFrame:
        return pd.DataFrame(
            [{"account_id": a, "statement_date": pd.Timestamp(d),
              "market_value": float(v)} for a, d, v in rows])

    def _no_transactions(self) -> pd.DataFrame:
        return pd.DataFrame({
            "settlement_date":  pd.Series([], dtype="datetime64[ns]"),
            "account_id":       pd.Series([], dtype="object"),
            "transaction_type": pd.Series([], dtype="object"),
            "amount":           pd.Series([], dtype="float64"),
            "flow_scope":       pd.Series([], dtype="object"),
        })

    def test_lagging_account_carried_no_phantom_loss(self) -> None:
        # A files Jan + Feb (real); B files Jan only and lags the Feb frontier.
        # Feb NAV must include B's carried $500 (-> 1600, not 1100), so the
        # month return reflects only A's real +10% diluted by B's flat carry
        # (+6.667%), NOT a -26.7% phantom loss from B vanishing. The carry is
        # surfaced via n_accounts_filled / filled_accounts (the tripwire).
        pos = self._positions([
            ("A", "2026-01-31", 1000.0),
            ("A", "2026-02-28", 1100.0),
            ("B", "2026-01-31",  500.0),
        ])
        port = ct.compute_portfolio_twr(
            pos, self._no_transactions(), synthetic_onboarding={})
        feb = port[port["month"].astype(str) == "2026-02"].iloc[0]
        self.assertAlmostEqual(feb["nav"], 1600.0)
        self.assertAlmostEqual(feb["return_pct"], (1600.0 - 1500.0) / 1500.0,
                               places=6)
        self.assertEqual(int(feb["n_accounts_filled"]), 1)
        self.assertIn("B", feb["filled_accounts"])


# ---------------------------------------------------------------------------
# Synthetic-onboarding flow invariant — Phase 1A audit
#
# compute_portfolio_twr injects a synthetic inflow on each onboarding account's
# debut month so the rollup return doesn't show as +500% from "magically
# appearing" money. The synthetic amount lands in BOTH the dedicated
# ``synthetic_flow`` column AND the all-flow ``net_external_flow`` column
# (modified-Dietz needs it as a flow). Consumers computing "total real
# deposits ever" must subtract synthetic_flow first, or pull from the
# transactions table directly. This test locks in the invariant so a future
# refactor that accidentally drops the disclosure can't ship silently.
# ---------------------------------------------------------------------------
class TestPortfolioTwrSyntheticFlowInvariant(unittest.TestCase):
    def _positions(self, rows: list[tuple[str, str, float, str]]) -> pd.DataFrame:
        return pd.DataFrame([
            {"account_id": a, "statement_date": pd.Timestamp(d),
             "market_value": v, "broker": b}
            for a, d, v, b in rows
        ])

    def _transactions(self, rows: list[dict]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame({
                "settlement_date":  pd.Series([], dtype="datetime64[ns]"),
                "account_id":       pd.Series([], dtype="object"),
                "broker":           pd.Series([], dtype="object"),
                "transaction_type": pd.Series([], dtype="object"),
                "amount":           pd.Series([], dtype="float64"),
                "flow_scope":       pd.Series([], dtype="object"),
            })
        df = pd.DataFrame(rows)
        df["settlement_date"] = pd.to_datetime(df["settlement_date"])
        return df

    def test_net_external_flow_minus_synthetic_recovers_real_flows(self) -> None:
        # REG-1 has a $1000 real deposit on 2026-02-15. SYNTH-1 debuts 2026-01
        # at $500 NAV (no real wire — predates archive). Real external flow
        # total: $1000. Synthetic total: $500. Invariant:
        # sum(net_external_flow) - sum(synthetic_flow) == sum(real flows).
        pos = self._positions([
            ("REG-1",   "2026-01-31",  1000.0, "fidelity"),
            ("REG-1",   "2026-02-28",  2050.0, "fidelity"),
            ("REG-1",   "2026-03-31",  2050.0, "fidelity"),
            ("SYNTH-1", "2026-01-31",   500.0, "fidelity"),
            ("SYNTH-1", "2026-02-28",   500.0, "fidelity"),
            ("SYNTH-1", "2026-03-31",   500.0, "fidelity"),
        ])
        txn = self._transactions([
            {"settlement_date": "2026-02-15", "account_id": "REG-1",
             "broker": "fidelity", "transaction_type": "transfer_in",
             "amount": 1000.0, "flow_scope": "external"},
        ])
        out = ct.compute_portfolio_twr(
            pos, txn, synthetic_onboarding={"SYNTH-1": "2026-01"},
        )
        total_net_external = float(out["net_external_flow"].sum())
        total_synthetic = float(out["synthetic_flow"].sum())
        real_recovered = total_net_external - total_synthetic
        self.assertAlmostEqual(
            real_recovered, 1000.0,
            msg="Invariant broken: subtracting synthetic_flow from "
                "net_external_flow no longer recovers the real external "
                "flow sum. A consumer summing net_external_flow as 'total "
                "real deposits' would now overstate by the synthetic amount."
        )
        self.assertAlmostEqual(total_synthetic, 500.0)


# ---------------------------------------------------------------------------
# IRR-since-cutoff — start_date parameter on compute_account_irr /
# compute_portfolio_irr. Truncates the cashflow window and injects the
# pre-cutoff NAV as a synthetic deposit at start_date so the algorithm
# doesn't treat held capital as a free gift.
# ---------------------------------------------------------------------------
class TestIrrSinceCutoff(unittest.TestCase):
    def _positions(self, rows: list[tuple[str, str, float, str]]) -> pd.DataFrame:
        return pd.DataFrame([
            {"account_id": a, "statement_date": pd.Timestamp(d),
             "market_value": v, "broker": b}
            for a, d, v, b in rows
        ])

    def _transactions(self, rows: list[dict]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame({
                "settlement_date":  pd.Series([], dtype="datetime64[ns]"),
                "account_id":       pd.Series([], dtype="object"),
                "broker":           pd.Series([], dtype="object"),
                "transaction_type": pd.Series([], dtype="object"),
                "amount":           pd.Series([], dtype="float64"),
                "flow_scope":       pd.Series([], dtype="object"),
            })
        df = pd.DataFrame(rows)
        df["settlement_date"] = pd.to_datetime(df["settlement_date"])
        return df

    def test_account_irr_default_matches_unbounded(self) -> None:
        # start_date=None must produce byte-identical results to omitting it.
        pos = self._positions([
            ("AAA-11111", "2024-01-31", 1000.0, "fidelity"),
            ("AAA-11111", "2025-01-31", 1100.0, "fidelity"),
        ])
        txn = self._transactions([
            {"settlement_date": "2024-01-31", "account_id": "AAA-11111",
             "broker": "fidelity", "transaction_type": "transfer_in",
             "amount": 1000.0, "flow_scope": "external"},
        ])
        a = ct.compute_account_irr(pos, txn, synthetic_onboarding={})
        b = ct.compute_account_irr(pos, txn, synthetic_onboarding={},
                                   start_date=None)
        self.assertAlmostEqual(float(a["irr"].iloc[0]),
                               float(b["irr"].iloc[0]), places=8)

    def test_account_irr_cutoff_injects_nav(self) -> None:
        # Account grew 100 → 150 in 2024, then 150 → 165 in 2025 (no flows).
        # Full-history IRR ≈ +28%; since-2025-01-01 IRR ≈ +10% (165/150 - 1).
        # Without the NAV-at-cutoff injection, xirr would see only the
        # terminal $165 with no negative cashflow → degenerate.
        pos = self._positions([
            ("AAA-11111", "2024-01-31",  100.0, "fidelity"),
            ("AAA-11111", "2024-12-31",  150.0, "fidelity"),
            ("AAA-11111", "2025-12-31",  165.0, "fidelity"),
        ])
        txn = self._transactions([
            {"settlement_date": "2024-01-31", "account_id": "AAA-11111",
             "broker": "fidelity", "transaction_type": "transfer_in",
             "amount": 100.0, "flow_scope": "external"},
        ])
        full = ct.compute_account_irr(pos, txn, synthetic_onboarding={})
        since = ct.compute_account_irr(
            pos, txn, synthetic_onboarding={},
            start_date=pd.Timestamp("2025-01-01"),
        )
        self.assertGreater(float(full["irr"].iloc[0]), 0.20)
        self.assertAlmostEqual(float(since["irr"].iloc[0]), 0.10, places=3)

    def test_account_irr_cutoff_drops_pre_window_flows(self) -> None:
        # A real $100 deposit in 2024 then $50 withdrawal in 2024 should be
        # invisible when cutoff is 2025-01-01. The pre-cutoff NAV (after the
        # withdrawal) is what's injected on the cutoff date.
        pos = self._positions([
            ("AAA-11111", "2024-01-31",  100.0, "fidelity"),
            ("AAA-11111", "2024-06-30",   55.0, "fidelity"),  # post-withdrawal
            ("AAA-11111", "2024-12-31",   60.0, "fidelity"),
            ("AAA-11111", "2025-12-31",   66.0, "fidelity"),  # +10% in 2025
        ])
        txn = self._transactions([
            {"settlement_date": "2024-01-31", "account_id": "AAA-11111",
             "broker": "fidelity", "transaction_type": "transfer_in",
             "amount": 100.0, "flow_scope": "external"},
            {"settlement_date": "2024-06-15", "account_id": "AAA-11111",
             "broker": "fidelity", "transaction_type": "transfer_out",
             "amount": -50.0, "flow_scope": "external"},
        ])
        since = ct.compute_account_irr(
            pos, txn, synthetic_onboarding={},
            start_date=pd.Timestamp("2025-01-01"),
        )
        # n_cashflows should be exactly 2: injected NAV-at-cutoff + terminal.
        self.assertEqual(int(since["n_cashflows"].iloc[0]), 2)
        self.assertAlmostEqual(float(since["irr"].iloc[0]), 0.10, places=3)

    def test_account_irr_cutoff_account_opens_after(self) -> None:
        # Account doesn't exist until June 2025, well after a 2025-01-01 cutoff.
        # Should fall back to flow-based IRR — no NAV injection (nothing held
        # at cutoff), the real deposit + terminal NAV solve cleanly.
        pos = self._positions([
            ("BBB-22222", "2025-06-30", 1000.0, "fidelity"),
            ("BBB-22222", "2026-06-30", 1100.0, "fidelity"),
        ])
        txn = self._transactions([
            {"settlement_date": "2025-06-15", "account_id": "BBB-22222",
             "broker": "fidelity", "transaction_type": "transfer_in",
             "amount": 1000.0, "flow_scope": "external"},
        ])
        since = ct.compute_account_irr(
            pos, txn, synthetic_onboarding={},
            start_date=pd.Timestamp("2025-01-01"),
        )
        self.assertEqual(int(since["n_cashflows"].iloc[0]), 2)
        self.assertAlmostEqual(float(since["irr"].iloc[0]), 0.10, places=2)

    def test_account_irr_cutoff_account_closed_before(self) -> None:
        # Account's last statement is before the cutoff → no row in output.
        pos = self._positions([
            ("CCC-33333", "2024-01-31", 1000.0, "fidelity"),
            ("CCC-33333", "2024-06-30", 1050.0, "fidelity"),
        ])
        txn = self._transactions([
            {"settlement_date": "2024-01-31", "account_id": "CCC-33333",
             "broker": "fidelity", "transaction_type": "transfer_in",
             "amount": 1000.0, "flow_scope": "external"},
        ])
        since = ct.compute_account_irr(
            pos, txn, synthetic_onboarding={},
            start_date=pd.Timestamp("2025-01-01"),
        )
        self.assertTrue(since.empty)

    def test_account_irr_cutoff_synthetic_onboarding_post_cutoff(self) -> None:
        # Synthetic-onboarding still applies when the synthetic debut is INSIDE
        # the truncated window (account materialized post-cutoff with money
        # that predates the tracked transaction archive). Without this, the
        # account has no negative cashflow at all and xirr is undefined.
        pos = self._positions([
            ("SYN-1", "2022-10-31", 100.0, "fidelity"),
            ("SYN-1", "2023-10-31", 110.0, "fidelity"),  # +10% in a year
        ])
        txn = self._transactions([])  # no real transactions
        since = ct.compute_account_irr(
            pos, txn, synthetic_onboarding={"SYN-1": "2022-10"},
            start_date=pd.Timestamp("2022-01-01"),
        )
        # Two cashflows: synthetic -$100 at 2022-10-31, terminal +$110 at 2023-10-31.
        self.assertEqual(int(since["n_cashflows"].iloc[0]), 2)
        self.assertAlmostEqual(float(since["irr"].iloc[0]), 0.10, places=3)

    def test_account_irr_cutoff_synthetic_suppressed_when_pre_cutoff_nav(self) -> None:
        # When the account existed pre-cutoff (nav_at_cutoff > 0), synthetic
        # onboarding must NOT also fire — nav_at_cutoff already captures the
        # pre-cutoff value the synthetic was modeling. Double-counting would
        # halve the IRR.
        pos = self._positions([
            ("SYN-2", "2020-01-31", 100.0, "fidelity"),
            ("SYN-2", "2024-12-31", 150.0, "fidelity"),
            ("SYN-2", "2025-12-31", 165.0, "fidelity"),  # +10% in 2025
        ])
        txn = self._transactions([])
        since = ct.compute_account_irr(
            pos, txn, synthetic_onboarding={"SYN-2": "2020-01"},
            start_date=pd.Timestamp("2025-01-01"),
        )
        # Two cashflows: injected nav_at_cutoff $150 at 2025-01-01, terminal
        # $165 at 2025-12-31. NO additional synthetic at 2020-01.
        self.assertEqual(int(since["n_cashflows"].iloc[0]), 2)
        self.assertAlmostEqual(float(since["irr"].iloc[0]), 0.10, places=3)


# ---------------------------------------------------------------------------
# NaN-amount flow rows (in-kind journals the parser couldn't price) must be
# skipped everywhere a flow's dollar amount feeds the math: the IRR cashflow
# series (poisoned xirr into the -0.9999 floor) AND modified-Dietz TWR (a NaN
# weighted-flow blanked the month's return, silently dropping it from the
# linked cumulative). Regression for the 2026-06-07 historical-Fidelity
# re-parse: 3 unpriced JPM in-kind journals floored the portfolio + two JPM
# accounts' IRR to -99.99% and dropped Dec-2025 from the portfolio TWR.
# ---------------------------------------------------------------------------
class TestIrrNanAmountFlows(unittest.TestCase):
    def _positions(self, rows: list[tuple[str, str, float, str]]) -> pd.DataFrame:
        return pd.DataFrame([
            {"account_id": a, "statement_date": pd.Timestamp(d),
             "market_value": v, "broker": b}
            for a, d, v, b in rows
        ])

    def _transactions(self, rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        df["settlement_date"] = pd.to_datetime(df["settlement_date"])
        return df

    def test_account_irr_skips_nan_amount_flow(self) -> None:
        # $1000 deposit 2025-01-31, grows to $1100 by 2026-01-31 (+10%), plus
        # an in-kind journal with no parseable amount (amount=NaN). The NaN row
        # must be skipped: a finite +10% IRR over exactly 2 cashflows, NOT the
        # -0.9999 floor over 3.
        pos = self._positions([
            ("AAA-11111", "2025-01-31", 1000.0, "fidelity"),
            ("AAA-11111", "2026-01-31", 1100.0, "fidelity"),
        ])
        txn = self._transactions([
            {"settlement_date": "2025-01-31", "account_id": "AAA-11111",
             "broker": "fidelity", "transaction_type": "transfer_in",
             "amount": 1000.0, "flow_scope": "external"},
            {"settlement_date": "2025-06-30", "account_id": "AAA-11111",
             "broker": "fidelity", "transaction_type": "transfer_in",
             "amount": float("nan"), "flow_scope": "external"},
        ])
        out = ct.compute_account_irr(pos, txn, synthetic_onboarding={})
        irr = float(out["irr"].iloc[0])
        self.assertTrue(np.isfinite(irr))
        self.assertAlmostEqual(irr, 0.10, places=3)
        self.assertEqual(int(out["n_cashflows"].iloc[0]), 2)

    def test_portfolio_irr_skips_nan_amount_flow(self) -> None:
        # The same unpriced in-kind journal must not floor the portfolio IRR.
        pos = self._positions([
            ("AAA-11111", "2025-01-31", 1000.0, "fidelity"),
            ("AAA-11111", "2026-01-31", 1100.0, "fidelity"),
        ])
        txn = self._transactions([
            {"settlement_date": "2025-01-31", "account_id": "AAA-11111",
             "broker": "fidelity", "transaction_type": "transfer_in",
             "amount": 1000.0, "flow_scope": "external"},
            {"settlement_date": "2025-06-30", "account_id": "AAA-11111",
             "broker": "fidelity", "transaction_type": "transfer_out",
             "amount": float("nan"), "flow_scope": "external"},
        ])
        out = ct.compute_portfolio_irr(pos, txn, synthetic_onboarding={})
        self.assertTrue(np.isfinite(out["irr"]))
        self.assertAlmostEqual(out["irr"], 0.10, places=3)

    def test_monthly_twr_skips_nan_amount_flow(self) -> None:
        # The same unpriced in-kind journal also poisons TWR. modified_dietz's
        # weighted-flow loop does `weighted_flows += amount * w`, and NaN * w
        # = NaN, so the month's return goes NaN and is silently dropped from
        # the linked cumulative (.sum() on the flow total skips NaN, but the
        # weighted loop does not). Dropping the NaN flow keeps the month:
        # $1000 -> $1100 with no usable flow = +10%.
        pos = self._positions([
            ("AAA-11111", "2025-11-30", 1000.0, "fidelity"),
            ("AAA-11111", "2025-12-31", 1100.0, "fidelity"),
        ])
        txn = self._transactions([
            {"settlement_date": "2025-12-15", "account_id": "AAA-11111",
             "broker": "fidelity", "transaction_type": "transfer_in",
             "amount": float("nan"), "flow_scope": "external"},
        ])
        twr = ct.compute_monthly_twr(pos, txn)
        dec = twr[twr["month"].astype(str) == "2025-12"]
        self.assertEqual(len(dec), 1)
        r = float(dec.iloc[0]["return_pct"])
        self.assertTrue(np.isfinite(r))
        self.assertAlmostEqual(r, 0.10, places=3)


class TestIrrSanityGate(unittest.TestCase):
    """The ingest-time IRR sanity gate (compute_twr).

    PR #147's NaN-amount in-kind flows floored the portfolio + two JPM accounts'
    IRR at xirr's -0.9999 bisection bound (shown as -99.99%), and it went
    unnoticed because nothing validated the IRR step the way reconcile_holdings
    (PR #129) validates NAV before writing. This gate is that missing check: a
    floor-pinned IRR (the fake-return corruption signature) blocks the ingest
    loud, while a non-finite IRR (an honest "n/a") and a legitimately large
    negative return are surfaced as advisories rather than blocked. Mirrors
    reconcile_holdings' banded `classify`.
    """

    # ---- classify_irr: the per-account band -------------------------------
    def test_floor_pinned_irr_is_error(self) -> None:
        # Exactly at, just within tolerance of, and below xirr's -0.9999 floor.
        self.assertEqual(ct.classify_irr(-0.9999), "error")
        self.assertEqual(ct.classify_irr(-0.9999 + 1e-9), "error")
        self.assertEqual(ct.classify_irr(-0.99999), "error")

    def test_non_finite_irr_is_watch_not_error(self) -> None:
        # A NaN/inf IRR is *undefined* (shown as "n/a"), not the silent fake
        # return the floor is. Real accounts legitimately produce it (a
        # single-cashflow new account, e.g. 100-000005 / Z10-000009 in the live
        # data), so it must be surfaced, never block ingest.
        for bad in (float("nan"), float("inf"), float("-inf")):
            self.assertEqual(ct.classify_irr(bad), "watch")

    def test_legitimately_large_negative_return_is_not_error(self) -> None:
        # The crux: a real -60%/yr loss must NOT be mistaken for the floor.
        self.assertEqual(ct.classify_irr(-0.60), "ok")

    def test_normal_returns_are_ok(self) -> None:
        for good in (0.2703, 0.0, -0.10, 1.5):
            self.assertEqual(ct.classify_irr(good), "ok")

    def test_extreme_but_finite_irr_is_watch(self) -> None:
        # Surfaced (advisory), never blocks — between "ok" and the "error" floor.
        self.assertEqual(ct.classify_irr(-0.95), "watch")   # huge loss, unfloored
        self.assertEqual(ct.classify_irr(50.0), "watch")    # implausibly high

    # ---- check_irr_sanity: band each row of the computed IRR table ---------
    def test_check_irr_sanity_flags_floored_rows(self) -> None:
        # The exact #147 signature: PORTFOLIO + one account floored, rest sane.
        irr_df = pd.DataFrame([
            {"account_id": "AAA-11111", "irr": 0.27},
            {"account_id": "BBB-22222", "irr": -0.9999},
            {"account_id": "PORTFOLIO", "irr": -0.9999},
        ])
        bands = {r.account_id: r.band for r in ct.check_irr_sanity(irr_df)}
        self.assertEqual(bands["AAA-11111"], "ok")
        self.assertEqual(bands["BBB-22222"], "error")
        self.assertEqual(bands["PORTFOLIO"], "error")

    def test_check_irr_sanity_all_ok_when_clean(self) -> None:
        irr_df = pd.DataFrame([
            {"account_id": "AAA-11111", "irr": 0.27},
            {"account_id": "PORTFOLIO", "irr": 0.19},
        ])
        self.assertTrue(all(r.band == "ok"
                            for r in ct.check_irr_sanity(irr_df)))

    # ---- run_irr_gate: the block decision ---------------------------------
    def test_gate_blocks_on_error_band(self) -> None:
        irr_df = pd.DataFrame([
            {"account_id": "AAA-11111", "irr": 0.27},
            {"account_id": "PORTFOLIO", "irr": -0.9999},
        ])
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(ct.run_irr_gate(irr_df), 3)

    def test_gate_force_overrides_block(self) -> None:
        irr_df = pd.DataFrame([
            {"account_id": "PORTFOLIO", "irr": -0.9999},
        ])
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(ct.run_irr_gate(irr_df, force=True), 0)

    def test_gate_passes_when_clean(self) -> None:
        irr_df = pd.DataFrame([
            {"account_id": "AAA-11111", "irr": 0.27},
            {"account_id": "PORTFOLIO", "irr": 0.19},
        ])
        with contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(ct.run_irr_gate(irr_df), 0)

    # ---- end-to-end: a NaN-amount in-kind flow -> undefined IRR, surfaced ---
    def test_nan_amount_flow_yields_undefined_irr_surfaced_as_watch(self) -> None:
        # A lone unpriced in-kind journal (amount=NaN) is the PR #147 input.
        # Post-fix it is dropped upstream, leaving only the terminal NAV -> a
        # degenerate single-cashflow series -> xirr returns NaN (undefined),
        # shown as "n/a" on the dashboard. The gate surfaces that as "watch" --
        # NOT the blocking "error" floor, and NOT a fake -99.99%.
        pos = self._positions([
            ("BBB-22222", "2025-01-31", 1000.0, "jpm"),
            ("BBB-22222", "2026-01-31", 1100.0, "jpm"),
        ])
        txn = self._transactions([
            {"settlement_date": "2025-06-30", "account_id": "BBB-22222",
             "broker": "jpm", "transaction_type": "transfer_in",
             "amount": float("nan"), "flow_scope": "external"},
        ])
        irr_df = ct.compute_account_irr(pos, txn, synthetic_onboarding={})
        self.assertFalse(np.isfinite(float(irr_df["irr"].iloc[0])))
        rows = ct.check_irr_sanity(irr_df)
        self.assertEqual(rows[0].account_id, "BBB-22222")
        self.assertEqual(rows[0].band, "watch")

    # ---- main(): writes when clean, blocks + preserves prior CSV on floor ---
    def test_main_writes_csv_when_irr_is_clean(self) -> None:
        # Happy path: finite per-account + portfolio IRR -> gate passes ->
        # main writes irr_per_account.csv and returns 0.
        with TemporaryDirectory() as td:
            td = Path(td)
            pos_csv, txn_csv = td / "positions.csv", td / "transactions.csv"
            irr_csv = td / "irr_per_account.csv"
            self._positions([
                ("ZZZ-99999", "2025-01-31", 1000.0, "jpm"),
                ("ZZZ-99999", "2026-01-31", 1100.0, "jpm"),
            ]).to_csv(pos_csv, index=False)
            self._transactions([
                {"settlement_date": "2025-01-31", "account_id": "ZZZ-99999",
                 "broker": "jpm", "transaction_type": "transfer_in",
                 "amount": 1000.0, "flow_scope": "external"},
            ]).to_csv(txn_csv, index=False)
            with patch.object(ct, "POSITIONS_CSV", pos_csv), \
                    patch.object(ct, "TRANSACTIONS_CSV", txn_csv), \
                    patch.object(ct, "DATA_DIR", td), \
                    patch.object(ct, "FIDELITY_COVERAGE_CSV", td / "none.csv"), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()):
                code = ct.main([])
            self.assertEqual(code, 0)
            written = pd.read_csv(irr_csv)
            self.assertIn("ZZZ-99999", set(written["account_id"].astype(str)))

    def test_main_blocks_and_preserves_prior_csv_on_floored_irr(self) -> None:
        # Simulate the PR #147 corruption reaching the write step: a floored
        # portfolio IRR (post-fix the engine can't produce it naturally, so
        # inject it at the computation boundary). The gate must abort (return 3)
        # and leave the prior-good irr_per_account.csv untouched.
        with TemporaryDirectory() as td:
            td = Path(td)
            pos_csv, txn_csv = td / "positions.csv", td / "transactions.csv"
            irr_csv = td / "irr_per_account.csv"
            self._positions([
                ("ZZZ-99999", "2025-01-31", 1000.0, "jpm"),
                ("ZZZ-99999", "2026-01-31", 1100.0, "jpm"),
            ]).to_csv(pos_csv, index=False)
            self._transactions([
                {"settlement_date": "2025-01-31", "account_id": "ZZZ-99999",
                 "broker": "jpm", "transaction_type": "transfer_in",
                 "amount": 1000.0, "flow_scope": "external"},
            ]).to_csv(txn_csv, index=False)
            sentinel = "account_id,irr\nPRIOR-GOOD,0.19\n"
            irr_csv.write_text(sentinel)
            floored = {"irr": -0.9999, "n_cashflows": 5, "terminal_nav": 1100.0,
                       "start_date": pd.Timestamp("2025-01-31"),
                       "end_date": pd.Timestamp("2026-01-31"),
                       "total_deposits": 1000.0, "total_withdrawals": 0.0}
            with patch.object(ct, "POSITIONS_CSV", pos_csv), \
                    patch.object(ct, "TRANSACTIONS_CSV", txn_csv), \
                    patch.object(ct, "DATA_DIR", td), \
                    patch.object(ct, "FIDELITY_COVERAGE_CSV", td / "none.csv"), \
                    patch.object(ct, "compute_portfolio_irr", return_value=floored), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    contextlib.redirect_stderr(io.StringIO()) as err:
                code = ct.main([])
            self.assertEqual(code, 3)                        # ingest aborts
            self.assertEqual(irr_csv.read_text(), sentinel)  # prior CSV kept
            self.assertIn("BLOCKED", err.getvalue())

    def _positions(self, rows: list[tuple[str, str, float, str]]) -> pd.DataFrame:
        return pd.DataFrame([
            {"account_id": a, "statement_date": pd.Timestamp(d),
             "market_value": v, "broker": b}
            for a, d, v, b in rows
        ])

    def _transactions(self, rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        df["settlement_date"] = pd.to_datetime(df["settlement_date"])
        return df


class TestScopedPortfolioIrr(unittest.TestCase):
    """compute_portfolio_irr(scoped=True): a lone internal-transfer leg (its
    pair_id partner absent from the frame) is a real flow from the scope's
    POV; a pair complete within the frame still washes; default path
    unchanged. Spec: docs/superpowers/specs/2026-08-07-broker-scoped-irr-
    design.md §Engine."""

    def _positions(self, rows: list[tuple[str, str, float, str]]) -> pd.DataFrame:
        return pd.DataFrame([
            {"account_id": a, "statement_date": pd.Timestamp(d),
             "market_value": v, "broker": b}
            for a, d, v, b in rows
        ])

    def _transactions(self, rows: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        df["settlement_date"] = pd.to_datetime(df["settlement_date"])
        return df

    def _one_account_frames(self, internal_pair_id):
        # AAA deposits 1000 on 2025-01-31, sends 500 OUT via an internal
        # transfer on 2025-06-30 whose partner account is NOT in the frame
        # (broker-narrowed scope), and still ends 2026-01-31 at 1100.
        pos = self._positions([
            ("AAA-11111", "2025-01-31", 1000.0, "fidelity"),
            ("AAA-11111", "2026-01-31", 1100.0, "fidelity"),
        ])
        txn = self._transactions([
            {"settlement_date": "2025-01-31", "account_id": "AAA-11111",
             "broker": "fidelity", "transaction_type": "transfer_in",
             "amount": 1000.0, "flow_scope": "external", "pair_id": None},
            {"settlement_date": "2025-06-30", "account_id": "AAA-11111",
             "broker": "fidelity", "transaction_type": "transfer_out",
             "amount": -500.0, "flow_scope": "internal",
             "pair_id": internal_pair_id},
        ])
        return pos, txn

    def test_lone_internal_leg_counts_as_flow(self):
        pos, txn = self._one_account_frames("pairX")
        base = ct.compute_portfolio_irr(pos, txn, synthetic_onboarding={})
        scoped = ct.compute_portfolio_irr(pos, txn, synthetic_onboarding={},
                                          scoped=True)
        # default: external only -> 2 cashflows, +10%.
        self.assertEqual(base["n_cashflows"], 2)
        self.assertAlmostEqual(base["irr"], 0.10, places=3)
        # scoped: the lone internal leg is a withdrawal -> 3 cashflows,
        # strictly higher money-weighted return.
        self.assertEqual(scoped["n_cashflows"], 3)
        self.assertTrue(np.isfinite(scoped["irr"]))
        self.assertGreater(scoped["irr"], base["irr"])

    def test_null_pair_id_internal_leg_included(self):
        pos, txn = self._one_account_frames(None)
        scoped = ct.compute_portfolio_irr(pos, txn, synthetic_onboarding={},
                                          scoped=True)
        self.assertEqual(scoped["n_cashflows"], 3)

    def test_intra_frame_pair_still_washes(self):
        # Both legs of pair p1 are inside the frame (AAA -> BBB, 500 on
        # 2025-06-30): scoped must equal the default external-only result.
        pos = self._positions([
            ("AAA-11111", "2025-01-31", 1000.0, "fidelity"),
            ("AAA-11111", "2026-01-31", 600.0, "fidelity"),
            ("BBB-22222", "2025-06-30", 500.0, "fidelity"),
            ("BBB-22222", "2026-01-31", 550.0, "fidelity"),
        ])
        txn = self._transactions([
            {"settlement_date": "2025-01-31", "account_id": "AAA-11111",
             "broker": "fidelity", "transaction_type": "transfer_in",
             "amount": 1000.0, "flow_scope": "external", "pair_id": None},
            {"settlement_date": "2025-06-30", "account_id": "AAA-11111",
             "broker": "fidelity", "transaction_type": "transfer_out",
             "amount": -500.0, "flow_scope": "internal", "pair_id": "p1"},
            {"settlement_date": "2025-06-30", "account_id": "BBB-22222",
             "broker": "fidelity", "transaction_type": "transfer_in",
             "amount": 500.0, "flow_scope": "internal", "pair_id": "p1"},
        ])
        base = ct.compute_portfolio_irr(pos, txn, synthetic_onboarding={})
        scoped = ct.compute_portfolio_irr(pos, txn, synthetic_onboarding={},
                                          scoped=True)
        self.assertEqual(scoped["n_cashflows"], base["n_cashflows"])
        self.assertAlmostEqual(scoped["irr"], base["irr"], places=12)
        # 1000 -> 1150 over 12 months = +15%.
        self.assertAlmostEqual(scoped["irr"], 0.15, places=3)

    def test_scoped_composes_with_start_date(self):
        # Cutoff after the deposit, before the internal leg: the pre-cutoff
        # NAV seeds the series and the lone leg still counts.
        pos, txn = self._one_account_frames("pairX")
        out = ct.compute_portfolio_irr(
            pos, txn, synthetic_onboarding={},
            start_date=pd.Timestamp("2025-03-31"), scoped=True)
        # cashflows: -nav_at_cutoff, +500 internal-out, +1100 terminal.
        self.assertEqual(out["n_cashflows"], 3)
        self.assertTrue(np.isfinite(out["irr"]))

    def test_scoped_noop_without_flow_scope_column(self):
        # Legacy frames (no flow_scope column) fall back to FLOW_TYPES —
        # scoped must change nothing.
        pos, txn = self._one_account_frames("pairX")
        txn2 = txn.drop(columns=["flow_scope", "pair_id"])
        base = ct.compute_portfolio_irr(pos, txn2, synthetic_onboarding={})
        scoped = ct.compute_portfolio_irr(pos, txn2, synthetic_onboarding={},
                                          scoped=True)
        self.assertEqual(scoped["n_cashflows"], base["n_cashflows"])

    def test_colliding_pair_id_boundary_legs_counted(self):
        # _make_pair_id hashes date|amount|sorted(accounts) with no
        # occurrence counter: two same-day same-|amount| transfers OUT of
        # AAA to two different (both out-of-frame) counterparties collide
        # onto ONE pair_id ("dup"). The old count==1 rule saw counts["dup"]
        # == 2 and wrongly called that "paired" (both dropped); the correct
        # read is a GROUP whose in-scope legs net to -1000 (not 0) — pure
        # boundary-crossing money, so BOTH legs must count.
        pos = self._positions([
            ("AAA-11111", "2025-01-31", 1000.0, "fidelity"),
            ("AAA-11111", "2026-01-31", 1100.0, "fidelity"),
        ])
        txn = self._transactions([
            {"settlement_date": "2025-01-31", "account_id": "AAA-11111",
             "broker": "fidelity", "transaction_type": "transfer_in",
             "amount": 1000.0, "flow_scope": "external", "pair_id": None},
            {"settlement_date": "2025-06-30", "account_id": "AAA-11111",
             "broker": "fidelity", "transaction_type": "transfer_out",
             "amount": -500.0, "flow_scope": "internal", "pair_id": "dup"},
            {"settlement_date": "2025-06-30", "account_id": "AAA-11111",
             "broker": "fidelity", "transaction_type": "transfer_out",
             "amount": -500.0, "flow_scope": "internal", "pair_id": "dup"},
        ])
        base = ct.compute_portfolio_irr(pos, txn, synthetic_onboarding={})
        scoped = ct.compute_portfolio_irr(pos, txn, synthetic_onboarding={},
                                          scoped=True)
        self.assertEqual(scoped["n_cashflows"], 4)
        self.assertEqual(scoped["total_withdrawals"], 1000.0)
        self.assertTrue(np.isfinite(scoped["irr"]))
        self.assertGreater(scoped["irr"], base["irr"])

    def test_colliding_pair_id_balanced_group_still_washes(self):
        # Same hash collision ("dup" shared by two same-day same-|amount|
        # pairs), but this time BOTH complete pairs are inside the frame
        # (AAA<->BBB twice). The group nets to exactly $0, so it must wash
        # in full — scoped must equal the whole-book default, identically
        # to a single intra-frame pair. This holds under the OLD count==1
        # rule too (counts["dup"] == 4, never "lone") — a same-before-and-
        # after case, unlike the boundary-leg test above.
        pos = self._positions([
            ("AAA-11111", "2025-01-31", 1000.0, "fidelity"),
            ("AAA-11111", "2026-01-31", 600.0, "fidelity"),
            ("BBB-22222", "2025-06-30", 1000.0, "fidelity"),
            ("BBB-22222", "2026-01-31", 1100.0, "fidelity"),
        ])
        txn = self._transactions([
            {"settlement_date": "2025-01-31", "account_id": "AAA-11111",
             "broker": "fidelity", "transaction_type": "transfer_in",
             "amount": 1000.0, "flow_scope": "external", "pair_id": None},
            {"settlement_date": "2025-06-30", "account_id": "AAA-11111",
             "broker": "fidelity", "transaction_type": "transfer_out",
             "amount": -500.0, "flow_scope": "internal", "pair_id": "dup"},
            {"settlement_date": "2025-06-30", "account_id": "BBB-22222",
             "broker": "fidelity", "transaction_type": "transfer_in",
             "amount": 500.0, "flow_scope": "internal", "pair_id": "dup"},
            {"settlement_date": "2025-06-30", "account_id": "AAA-11111",
             "broker": "fidelity", "transaction_type": "transfer_out",
             "amount": -500.0, "flow_scope": "internal", "pair_id": "dup"},
            {"settlement_date": "2025-06-30", "account_id": "BBB-22222",
             "broker": "fidelity", "transaction_type": "transfer_in",
             "amount": 500.0, "flow_scope": "internal", "pair_id": "dup"},
        ])
        base = ct.compute_portfolio_irr(pos, txn, synthetic_onboarding={})
        scoped = ct.compute_portfolio_irr(pos, txn, synthetic_onboarding={},
                                          scoped=True)
        self.assertEqual(scoped["n_cashflows"], base["n_cashflows"])
        self.assertAlmostEqual(scoped["irr"], base["irr"], places=12)


if __name__ == "__main__":
    unittest.main()

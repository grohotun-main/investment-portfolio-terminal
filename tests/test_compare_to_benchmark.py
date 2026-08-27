"""Tests for parsers/compare_to_benchmark.py.

The May 2026 audit's #1 finding (real SPY spread is -5.69 pp/yr, not the
displayed -1.05) traced back to a stale comparison_spy.csv. The three
CLI-side pure-shaped helpers in this module — `benchmark_value_lookup`,
`compute_twr_comparison`, `compute_irr_comparison` — had no direct unit
tests. Pin the window-filter rules and the SPY-counterfactual math here.

Phase 1B follow-up: the dashboard-side `build_twr_comparison` and
`build_irr_comparison` (formerly inline in app.py, moved here so they're
testable without importing Streamlit) get the same treatment. The two
pairs share the no-partial-period overlap rule but have different output
shapes — the build_* variants emit the wealth / drawdown columns the
Performance-vs-Benchmark tab plots and the win-rate counts.
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from compare_to_benchmark import (  # noqa: E402
    benchmark_value_lookup,
    build_irr_comparison,
    build_twr_comparison,
    compute_irr_comparison,
    compute_twr_comparison,
)


def _tr_lookup(start: str, end: str, daily_return: float = 0.0) -> pd.Series:
    """Build a tr_lookup Series spanning [start, end] with a constant daily
    return. Used to give the IRR comparison a stable benchmark price path."""
    idx = pd.date_range(start, end, freq="D")
    values = 100.0 * (1.0 + daily_return) ** np.arange(len(idx))
    return pd.Series(values, index=idx, name="tr_value")


class TestBenchmarkValueLookup(unittest.TestCase):
    def test_forward_fills_weekend(self) -> None:
        # Friday 2026-01-02 closes at 100; next trading day Monday 2026-01-05
        # at 101. Lookup on Saturday/Sunday must resolve to Friday's 100.
        tr = pd.DataFrame([
            {"date": pd.Timestamp("2026-01-02"), "tr_value": 100.0},
            {"date": pd.Timestamp("2026-01-05"), "tr_value": 101.0},
        ])
        s = benchmark_value_lookup(tr)
        self.assertAlmostEqual(s.loc[pd.Timestamp("2026-01-03")], 100.0)
        self.assertAlmostEqual(s.loc[pd.Timestamp("2026-01-04")], 100.0)
        self.assertAlmostEqual(s.loc[pd.Timestamp("2026-01-05")], 101.0)

    def test_returns_calendar_day_index(self) -> None:
        # Sparse trading-day input -> dense calendar-day output.
        tr = pd.DataFrame([
            {"date": pd.Timestamp("2026-01-02"), "tr_value": 100.0},
            {"date": pd.Timestamp("2026-01-05"), "tr_value": 101.0},
        ])
        s = benchmark_value_lookup(tr)
        self.assertEqual(list(s.index), list(pd.date_range("2026-01-02", "2026-01-05")))

    def test_unsorted_input_sorted_in_output(self) -> None:
        tr = pd.DataFrame([
            {"date": pd.Timestamp("2026-01-05"), "tr_value": 101.0},
            {"date": pd.Timestamp("2026-01-02"), "tr_value": 100.0},
        ])
        s = benchmark_value_lookup(tr)
        self.assertTrue(s.index.is_monotonic_increasing)
        self.assertAlmostEqual(s.iloc[0], 100.0)


def _port_row(month: str, prev: str, end: str, prev_nav: float, nav: float,
              ret: float) -> dict:
    return {
        "month": month,
        "prev_stmt_date": pd.Timestamp(prev),
        "statement_date": pd.Timestamp(end),
        "prev_nav": prev_nav,
        "nav": nav,
        "return_pct": ret,
    }


class TestComputeTwrComparison(unittest.TestCase):
    def test_excludes_period_with_prev_before_bench_start(self) -> None:
        # bench window starts 2026-01-02. Period whose prev_stmt_date is
        # 2025-12-31 must be dropped (partial period would be unfair).
        port = pd.DataFrame([
            _port_row("2026-01", "2025-12-31", "2026-01-31", 1000.0, 1050.0, 0.05),
            _port_row("2026-02", "2026-01-31", "2026-02-28", 1050.0, 1100.0, 0.04762),
        ])
        tr = _tr_lookup("2026-01-02", "2026-02-28")
        comp, _ = compute_twr_comparison("SPY", port, tr)
        self.assertEqual(len(comp), 1)
        self.assertEqual(comp.iloc[0]["month"], "2026-02")

    def test_excludes_period_with_end_after_bench_end(self) -> None:
        port = pd.DataFrame([
            _port_row("2026-01", "2025-12-31", "2026-01-31", 1000.0, 1050.0, 0.05),
            _port_row("2026-02", "2026-01-31", "2026-02-28", 1050.0, 1100.0, 0.04762),
        ])
        tr = _tr_lookup("2025-12-15", "2026-02-15")  # ends before Feb close
        comp, _ = compute_twr_comparison("SPY", port, tr)
        self.assertEqual(len(comp), 1)
        self.assertEqual(comp.iloc[0]["month"], "2026-01")

    def test_spread_is_port_minus_bench(self) -> None:
        # Construct a TR series whose Jan close / Dec close = 1.05 exactly,
        # so bench_return = 0.05. Port returned 0.05 -> spread = 0.0.
        # Then port returned 0.10 in Feb but bench is flat -> spread = +0.10.
        port = pd.DataFrame([
            _port_row("2026-01", "2025-12-31", "2026-01-31", 1000.0, 1050.0, 0.05),
            _port_row("2026-02", "2026-01-31", "2026-02-28", 1050.0, 1155.0, 0.10),
        ])
        # TR explicitly: 100 at Dec-31, 105 at Jan-31, 105 at Feb-28.
        tr_idx = pd.date_range("2025-12-15", "2026-03-15", freq="D")
        tr_vals = pd.Series(105.0, index=tr_idx)
        tr_vals.loc[:pd.Timestamp("2026-01-30")] = 100.0
        comp, _ = compute_twr_comparison("SPY", port, tr_vals)
        self.assertAlmostEqual(comp.iloc[0]["spread"], 0.0, places=6)
        self.assertAlmostEqual(comp.iloc[1]["spread"], 0.10, places=6)

    def test_summary_none_when_no_overlap(self) -> None:
        # Port window is entirely outside bench window.
        port = pd.DataFrame([
            _port_row("2026-01", "2025-12-31", "2026-01-31", 1000.0, 1050.0, 0.05),
        ])
        tr = _tr_lookup("2027-01-01", "2027-12-31")
        comp, summary = compute_twr_comparison("SPY", port, tr)
        self.assertTrue(comp.empty)
        self.assertIsNone(summary)

    def test_annualized_uses_12_over_n_months_formula(self) -> None:
        # 12 months of 1% port return each compounds to (1.01)^12 - 1 cum,
        # which annualized = (1+cum)^(12/12) - 1 = cum itself.
        rows = []
        for i in range(12):
            prev = pd.Timestamp("2026-01-01") + pd.DateOffset(months=i)
            end = prev + pd.DateOffset(months=1)
            rows.append(_port_row(end.strftime("%Y-%m"),
                                  prev.strftime("%Y-%m-%d"),
                                  end.strftime("%Y-%m-%d"),
                                  1000.0, 1010.0, 0.01))
        port = pd.DataFrame(rows)
        tr = _tr_lookup("2025-12-15", "2027-03-15", daily_return=0.0)  # flat bench
        _, summary = compute_twr_comparison("SPY", port, tr)
        expected_cum = (1.01) ** 12 - 1.0
        self.assertAlmostEqual(summary["port_twr_cum"], expected_cum, places=6)
        self.assertAlmostEqual(summary["port_twr_ann"], expected_cum, places=6)
        self.assertEqual(summary["n_months"], 12)


class TestComputeIrrComparison(unittest.TestCase):
    def _basic_inputs(self) -> tuple:
        port = pd.DataFrame([
            _port_row("2026-01", "2025-12-31", "2026-01-31", 1000.0, 1010.0, 0.01),
            _port_row("2026-02", "2026-01-31", "2026-02-28", 1010.0, 1020.0, 0.0099),
        ])
        positions = pd.DataFrame([
            {"account_id": "TEST-1", "statement_date": pd.Timestamp("2025-12-31"),
             "market_value": 1000.0},
            {"account_id": "TEST-1", "statement_date": pd.Timestamp("2026-01-31"),
             "market_value": 1010.0},
            {"account_id": "TEST-1", "statement_date": pd.Timestamp("2026-02-28"),
             "market_value": 1020.0},
        ])
        transactions = pd.DataFrame(columns=[
            "settlement_date", "amount", "flow_scope",
        ])
        return port, positions, transactions

    def test_returns_none_when_no_eligible_periods(self) -> None:
        port, positions, transactions = self._basic_inputs()
        tr = _tr_lookup("2027-01-01", "2027-12-31")  # disjoint
        result = compute_irr_comparison(port, tr, transactions, positions, {})
        self.assertIsNone(result)

    def test_counts_real_and_synthetic_flows_separately(self) -> None:
        # One external txn in window + synthetic onboarding at Jan-31 inside
        # the window. n_real_flows = 1, n_synth_flows = 1.
        port, positions, _ = self._basic_inputs()
        # Add the synthetic-onboarding account row to positions.
        positions = pd.concat([positions, pd.DataFrame([
            {"account_id": "SYNTH-1",
             "statement_date": pd.Timestamp("2026-01-31"),
             "market_value": 500.0},
        ])], ignore_index=True)
        transactions = pd.DataFrame([
            {"settlement_date": pd.Timestamp("2026-02-10"),
             "amount": 250.0, "flow_scope": "external"},
        ])
        tr = _tr_lookup("2025-12-15", "2026-03-15", daily_return=0.0001)
        result = compute_irr_comparison(port, tr, transactions, positions,
                                        {"SYNTH-1": "2026-01"})
        self.assertIsNotNone(result)
        self.assertEqual(result["n_real_flows"], 1)
        self.assertEqual(result["n_synth_flows"], 1)

    def test_spy_counterfactual_replaces_each_flow_with_spy_buy(self) -> None:
        # Flat benchmark, zero real flows, no synthetic flows: the SPY
        # counterfactual terminal NAV equals the start NAV. Spread between
        # portfolio NAV (which grew via returns) and SPY counterfactual is
        # therefore exactly the portfolio's cumulative return.
        port, positions, transactions = self._basic_inputs()
        tr = _tr_lookup("2025-12-15", "2026-03-15", daily_return=0.0)  # flat
        result = compute_irr_comparison(port, tr, transactions, positions, {})
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["spy_terminal_nav"], 1000.0, places=4)
        self.assertAlmostEqual(result["window_end_nav"], 1020.0)


class TestBuildTwrComparison(unittest.TestCase):
    """Dashboard-side TWR comparison — wealth, drawdown, and win-rate
    invariants that the Performance-vs-Benchmark tab depends on."""

    def test_returns_none_when_port_empty(self) -> None:
        tr = _tr_lookup("2026-01-01", "2026-12-31")
        self.assertIsNone(build_twr_comparison(pd.DataFrame(), tr))

    def test_returns_none_when_tr_empty(self) -> None:
        port = pd.DataFrame([
            _port_row("2026-01", "2025-12-31", "2026-01-31",
                      1000.0, 1050.0, 0.05),
        ])
        self.assertIsNone(build_twr_comparison(port, pd.Series(dtype=float)))

    def test_returns_none_when_no_overlap(self) -> None:
        port = pd.DataFrame([
            _port_row("2026-01", "2025-12-31", "2026-01-31",
                      1000.0, 1050.0, 0.05),
        ])
        tr = _tr_lookup("2027-01-01", "2027-12-31")
        self.assertIsNone(build_twr_comparison(port, tr))

    def test_filters_nan_return_pct_rows(self) -> None:
        # Mirrors compute_twr's i=0-NaN convention — the debut row carries
        # NaN return_pct because no prior NAV to chain from. That row must
        # be excluded; only fully-valid months land in the comparison.
        port = pd.DataFrame([
            _port_row("2026-01", "2025-12-31", "2026-01-31",
                      1000.0, 1050.0, float("nan")),
            _port_row("2026-02", "2026-01-31", "2026-02-28",
                      1050.0, 1100.0, 0.0476),
        ])
        tr = _tr_lookup("2025-12-15", "2026-03-15")
        result = build_twr_comparison(port, tr)
        self.assertIsNotNone(result)
        self.assertEqual(len(result["comp"]), 1)
        self.assertEqual(result["comp"].iloc[0]["month"], "2026-02")

    def test_excludes_period_with_prev_before_bench_start(self) -> None:
        port = pd.DataFrame([
            _port_row("2026-01", "2025-12-31", "2026-01-31",
                      1000.0, 1050.0, 0.05),
            _port_row("2026-02", "2026-01-31", "2026-02-28",
                      1050.0, 1100.0, 0.0476),
        ])
        tr = _tr_lookup("2026-01-02", "2026-12-31")  # starts AFTER 2025-12-31
        result = build_twr_comparison(port, tr)
        self.assertEqual(len(result["comp"]), 1)
        self.assertEqual(result["comp"].iloc[0]["month"], "2026-02")

    def test_excludes_period_with_end_after_bench_end(self) -> None:
        port = pd.DataFrame([
            _port_row("2026-01", "2025-12-31", "2026-01-31",
                      1000.0, 1050.0, 0.05),
            _port_row("2026-02", "2026-01-31", "2026-02-28",
                      1050.0, 1100.0, 0.0476),
        ])
        tr = _tr_lookup("2025-12-15", "2026-02-15")  # ends before Feb close
        result = build_twr_comparison(port, tr)
        self.assertEqual(len(result["comp"]), 1)
        self.assertEqual(result["comp"].iloc[0]["month"], "2026-01")

    def test_wealth_columns_compound_from_base_amount(self) -> None:
        # Base = $100K. Three months at +10%, -10%, +5% → wealth at each
        # statement_date should be 110000, 99000, 103950.
        port = pd.DataFrame([
            _port_row("2026-01", "2025-12-31", "2026-01-31",
                      1000.0, 1100.0, 0.10),
            _port_row("2026-02", "2026-01-31", "2026-02-28",
                      1100.0, 990.0, -0.10),
            _port_row("2026-03", "2026-02-28", "2026-03-31",
                      990.0, 1039.5, 0.05),
        ])
        tr = _tr_lookup("2025-12-15", "2026-04-15", daily_return=0.0)
        result = build_twr_comparison(port, tr, base_amount=100_000.0)
        comp = result["comp"]
        self.assertAlmostEqual(comp.iloc[0]["port_wealth"], 110_000.0, places=2)
        self.assertAlmostEqual(comp.iloc[1]["port_wealth"], 99_000.0, places=2)
        self.assertAlmostEqual(comp.iloc[2]["port_wealth"], 103_950.0, places=2)
        # summary["port_wealth_final"] mirrors the last row.
        self.assertAlmostEqual(
            result["summary"]["port_wealth_final"], 103_950.0, places=2)

    def test_drawdown_recovers_to_zero_when_new_high_reached(self) -> None:
        # Up 20% → down 10% → up 20%. Wealth = 120, 108, 129.6. Peak-to-
        # trough at month 2 is 108/120-1 = -10%. By month 3 (129.6) the
        # peak is reset to 129.6, so drawdown returns to 0%.
        port = pd.DataFrame([
            _port_row("2026-01", "2025-12-31", "2026-01-31",
                      1000.0, 1200.0, 0.20),
            _port_row("2026-02", "2026-01-31", "2026-02-28",
                      1200.0, 1080.0, -0.10),
            _port_row("2026-03", "2026-02-28", "2026-03-31",
                      1080.0, 1296.0, 0.20),
        ])
        tr = _tr_lookup("2025-12-15", "2026-04-15", daily_return=0.0)
        result = build_twr_comparison(port, tr)
        dd = result["comp"]["port_dd_pct"]
        self.assertAlmostEqual(dd.iloc[0], 0.0, places=4)
        self.assertAlmostEqual(dd.iloc[1], -10.0, places=4)
        self.assertAlmostEqual(dd.iloc[2], 0.0, places=4)
        # summary["port_max_dd"] is the minimum (most-negative) of these.
        self.assertAlmostEqual(
            result["summary"]["port_max_dd"], -10.0, places=4)

    def test_max_dd_date_is_actual_trough_date(self) -> None:
        # Lock the date-of-max-dd, not just the value — display uses both.
        port = pd.DataFrame([
            _port_row("2026-01", "2025-12-31", "2026-01-31",
                      1000.0, 1100.0, 0.10),
            _port_row("2026-02", "2026-01-31", "2026-02-28",
                      1100.0, 990.0, -0.10),
            _port_row("2026-03", "2026-02-28", "2026-03-31",
                      990.0, 1039.5, 0.05),
        ])
        tr = _tr_lookup("2025-12-15", "2026-04-15", daily_return=0.0)
        result = build_twr_comparison(port, tr)
        self.assertEqual(
            result["summary"]["port_max_dd_date"],
            pd.Timestamp("2026-02-28"),
        )

    def test_win_loss_tie_counts(self) -> None:
        # Construct port returns and a flat bench. Spreads = port_return.
        # +1%, -1%, 0%, +0.5% → win=2, loss=1, tie=1.
        port = pd.DataFrame([
            _port_row("2026-01", "2025-12-31", "2026-01-31",
                      1000.0, 1010.0, 0.01),
            _port_row("2026-02", "2026-01-31", "2026-02-28",
                      1010.0, 999.9, -0.01),
            _port_row("2026-03", "2026-02-28", "2026-03-31",
                      999.9, 999.9, 0.0),
            _port_row("2026-04", "2026-03-31", "2026-04-30",
                      999.9, 1004.9, 0.005),
        ])
        tr = _tr_lookup("2025-12-15", "2026-05-15", daily_return=0.0)
        s = build_twr_comparison(port, tr)["summary"]
        self.assertEqual(s["win_months"], 2)
        self.assertEqual(s["loss_months"], 1)
        self.assertEqual(s["tie_months"], 1)

    def test_sorts_port_by_statement_date(self) -> None:
        # Unsorted port input — output rows must be in date order so the
        # wealth.cumprod and drawdown.cummax are chained correctly.
        port = pd.DataFrame([
            _port_row("2026-03", "2026-02-28", "2026-03-31",
                      1100.0, 1155.0, 0.05),
            _port_row("2026-01", "2025-12-31", "2026-01-31",
                      1000.0, 1050.0, 0.05),
            _port_row("2026-02", "2026-01-31", "2026-02-28",
                      1050.0, 1100.0, 0.0476),
        ])
        tr = _tr_lookup("2025-12-15", "2026-04-15", daily_return=0.0)
        comp = build_twr_comparison(port, tr)["comp"]
        self.assertEqual(list(comp["month"]), ["2026-01", "2026-02", "2026-03"])


class TestBuildIrrComparison(unittest.TestCase):
    """Dashboard-side IRR comparison — windowed cashflow + SPY
    counterfactual. The `allowed_accounts` filter (introduced in PR #49
    for the Holdings filter wiring) and the explicit synthetic-onboarding
    parameter (moved out of an app.py global in this PR) are the two
    invariants most likely to drift in a future refactor."""

    def _basic_inputs(self) -> tuple:
        port = pd.DataFrame([
            _port_row("2026-01", "2025-12-31", "2026-01-31",
                      1000.0, 1010.0, 0.01),
            _port_row("2026-02", "2026-01-31", "2026-02-28",
                      1010.0, 1020.0, 0.0099),
        ])
        positions = pd.DataFrame([
            {"account_id": "TEST-1",
             "statement_date": pd.Timestamp("2025-12-31"),
             "market_value": 1000.0},
            {"account_id": "TEST-1",
             "statement_date": pd.Timestamp("2026-01-31"),
             "market_value": 1010.0},
            {"account_id": "TEST-1",
             "statement_date": pd.Timestamp("2026-02-28"),
             "market_value": 1020.0},
        ])
        transactions = pd.DataFrame(columns=[
            "settlement_date", "amount", "flow_scope", "account_id",
        ])
        return port, positions, transactions

    def test_returns_none_when_port_empty(self) -> None:
        tr = _tr_lookup("2026-01-01", "2026-12-31")
        result = build_irr_comparison(
            pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), tr, {})
        self.assertIsNone(result)

    def test_returns_none_when_tr_empty(self) -> None:
        port, positions, transactions = self._basic_inputs()
        result = build_irr_comparison(
            port, transactions, positions, pd.Series(dtype=float), {})
        self.assertIsNone(result)

    def test_returns_none_when_no_eligible_periods(self) -> None:
        port, positions, transactions = self._basic_inputs()
        tr = _tr_lookup("2027-01-01", "2027-12-31")  # disjoint
        result = build_irr_comparison(port, transactions, positions, tr, {})
        self.assertIsNone(result)

    def test_flat_bench_no_flows_terminal_equals_start(self) -> None:
        # SPY counterfactual: with flat TR + zero real flows + zero
        # synth flows, the only cashflow is -window_start_nav at t=0 and
        # +spy_terminal at t=end. spy_terminal = shares × tr_end =
        # (start_nav / tr_start) × tr_start = start_nav. So irr_bench = 0.
        port, positions, transactions = self._basic_inputs()
        tr = _tr_lookup("2025-12-15", "2026-03-15", daily_return=0.0)
        result = build_irr_comparison(
            port, transactions, positions, tr, {})
        self.assertIsNotNone(result)
        self.assertAlmostEqual(
            result["spy_terminal_nav"], 1000.0, places=4)
        self.assertAlmostEqual(result["irr_bench"], 0.0, places=4)

    def test_synth_only_included_when_account_in_window(self) -> None:
        # Synth maps "LATE-1" to month 2027-06 — well outside the test
        # window. n_synth_flows must be 0.
        port, positions, transactions = self._basic_inputs()
        positions = pd.concat([positions, pd.DataFrame([
            {"account_id": "LATE-1",
             "statement_date": pd.Timestamp("2027-06-30"),
             "market_value": 500.0},
        ])], ignore_index=True)
        tr = _tr_lookup("2025-12-15", "2026-03-15", daily_return=0.0)
        result = build_irr_comparison(
            port, transactions, positions, tr,
            synthetic_onboarding={"LATE-1": "2027-06"})
        self.assertEqual(result["n_synth_flows"], 0)

    def test_synth_flow_counted_when_in_window(self) -> None:
        # SYNTH-1 debuts inside the matching window.
        port, positions, transactions = self._basic_inputs()
        positions = pd.concat([positions, pd.DataFrame([
            {"account_id": "SYNTH-1",
             "statement_date": pd.Timestamp("2026-01-31"),
             "market_value": 500.0},
        ])], ignore_index=True)
        tr = _tr_lookup("2025-12-15", "2026-03-15", daily_return=0.0001)
        result = build_irr_comparison(
            port, transactions, positions, tr,
            synthetic_onboarding={"SYNTH-1": "2026-01"})
        self.assertEqual(result["n_synth_flows"], 1)

    def test_allowed_accounts_filters_synth(self) -> None:
        # Two synthetic-onboarding entries; only one account is allowed.
        # The disallowed account's synth flow must be dropped.
        port, positions, transactions = self._basic_inputs()
        positions = pd.concat([positions, pd.DataFrame([
            {"account_id": "SYNTH-A",
             "statement_date": pd.Timestamp("2026-01-31"),
             "market_value": 500.0},
            {"account_id": "SYNTH-B",
             "statement_date": pd.Timestamp("2026-01-31"),
             "market_value": 700.0},
        ])], ignore_index=True)
        tr = _tr_lookup("2025-12-15", "2026-03-15", daily_return=0.0)
        result = build_irr_comparison(
            port, transactions, positions, tr,
            synthetic_onboarding={"SYNTH-A": "2026-01",
                                  "SYNTH-B": "2026-01"},
            allowed_accounts={"TEST-1", "SYNTH-A"},
        )
        self.assertEqual(result["n_synth_flows"], 1)

    def test_allowed_accounts_filters_real_txn(self) -> None:
        # Two real txns in window; only TEST-1's is included when
        # allowed_accounts restricts to TEST-1.
        port, positions, _ = self._basic_inputs()
        transactions = pd.DataFrame([
            {"settlement_date": pd.Timestamp("2026-02-10"),
             "amount": 200.0, "flow_scope": "external",
             "account_id": "TEST-1"},
            {"settlement_date": pd.Timestamp("2026-02-15"),
             "amount": 300.0, "flow_scope": "external",
             "account_id": "OTHER-2"},
        ])
        tr = _tr_lookup("2025-12-15", "2026-03-15", daily_return=0.0)
        result = build_irr_comparison(
            port, transactions, positions, tr, {},
            allowed_accounts={"TEST-1"},
        )
        self.assertEqual(result["n_real_flows"], 1)

    def test_total_deposits_includes_window_start_nav(self) -> None:
        # Phase 1B audit noted: total_deposits in the summary aggregates
        # window_start_nav + external deposits + synth flows. Pin this so
        # a future refactor doesn't split the field without updating the
        # tile prose.
        port, positions, _ = self._basic_inputs()
        transactions = pd.DataFrame([
            {"settlement_date": pd.Timestamp("2026-02-10"),
             "amount": 250.0, "flow_scope": "external",
             "account_id": "TEST-1"},
        ])
        tr = _tr_lookup("2025-12-15", "2026-03-15", daily_return=0.0)
        result = build_irr_comparison(
            port, transactions, positions, tr, {})
        # window_start_nav (1000) + one external deposit (250) = 1250.
        self.assertAlmostEqual(result["total_deposits"], 1250.0, places=2)


if __name__ == "__main__":
    unittest.main()

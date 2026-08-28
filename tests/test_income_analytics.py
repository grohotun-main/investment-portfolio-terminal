"""Unit tests for parsers/income_analytics.py."""
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from parsers.income_analytics import (
    income_timeseries,
    latest_ex_date_through,
    load_div_history,
    forward_income,
    trailing_income,
)


def _tx(rows):
    """Build a transactions frame from (date, type, amount) or
    (date, type, amount, account_id) tuples."""
    recs = []
    for r in rows:
        recs.append({
            "settlement_date": r[0],
            "transaction_type": r[1],
            "amount": r[2],
            "account_id": r[3] if len(r) > 3 else "ACC-1",
        })
    df = pd.DataFrame(recs)
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    return df


def _tx_st(rows):
    """Build a transactions frame from
    (settlement_date, trade_date, type, amount) tuples. Either date may be
    None (-> NaT) to exercise the settlement->trade_date coalescing."""
    df = pd.DataFrame([{
        "settlement_date": r[0],
        "trade_date": r[1],
        "transaction_type": r[2],
        "amount": r[3],
        "account_id": "ACC-1",
    } for r in rows])
    df["settlement_date"] = pd.to_datetime(df["settlement_date"],
                                           errors="coerce")
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    return df


def _tx_sec(rows):
    """Build a transactions frame from
    (date, type, amount, symbol, cusip, description) tuples — the columns
    the distribution gate reads."""
    df = pd.DataFrame([{
        "settlement_date": r[0], "transaction_type": r[1], "amount": r[2],
        "symbol": r[3], "cusip": r[4], "description": r[5],
        "account_id": "ACC-1",
    } for r in rows])
    df["settlement_date"] = pd.to_datetime(df["settlement_date"])
    return df


class TestDistributionsAsIncome(unittest.TestCase):
    """Distributions S2 (spec 2026-08-22): a `principal_pmt` payout on held
    shares is yield (NEOS-style return-of-capital distributions; VISN's $10)
    and counts as dividends; cash-in-lieu and bond principal do not."""

    def test_security_principal_pmt_counts_as_dividends(self) -> None:
        ts = income_timeseries(_tx_sec([
            ("2026-04-27", "principal_pmt", 27500.0, "VISN", None,
             "VISTANCE NETWORKS INC 04/27 RT 10.000 PRINCIPAL"),
            ("2026-04-10", "dividend", 100.0, "SPY", None, "DIVIDEND"),
        ]))
        self.assertAlmostEqual(float(ts.iloc[0]["dividends"]), 27600.0, places=6)
        self.assertAlmostEqual(float(ts.iloc[0]["net"]), 27600.0, places=6)

    def test_cash_in_lieu_and_bond_principal_excluded(self) -> None:
        ts = income_timeseries(_tx_sec([
            ("2026-04-10", "dividend", 100.0, "SPY", None, "DIVIDEND"),
            ("2026-04-12", "principal_pmt", 41.20, "EA", None,
             "ELECTRONIC ARTS INC CASH IN LIEU OF FRACTIONAL SHARES"),
            ("2026-04-15", "principal_pmt", 5000.0, None, "912828X88",
             "UNITED STATES TREAS NOTE PRINCIPAL PMT"),
            ("2026-04-16", "principal_pmt", 3.0, "", None, "CIL MERGER"),
        ]))
        self.assertAlmostEqual(float(ts.iloc[0]["dividends"]), 100.0, places=6)

    def test_frame_without_security_columns_keeps_legacy_exclusion(self) -> None:
        # The minimal (date, type, amount) frame has no symbol to judge by —
        # principal_pmt stays out, exactly as before.
        ts = income_timeseries(_tx([
            ("2026-01-05", "dividend", 100.0),
            ("2026-01-06", "principal_pmt", 50.0),
        ]))
        self.assertAlmostEqual(float(ts.iloc[0]["net"]), 100.0, places=6)

    def test_trailing_income_counts_security_distributions(self) -> None:
        val = trailing_income(_tx_sec([
            ("2026-04-27", "principal_pmt", 27500.0, "VISN", None,
             "VISTANCE NETWORKS INC 04/27 RT 10.000 PRINCIPAL"),
            ("2026-04-27", "reinvestment", -100.0, "SPY", None, "REINVESTMENT"),
        ]), date(2026, 6, 30))
        self.assertAlmostEqual(val, 27500.0, places=6)


class TestIncomeTimeseries(unittest.TestCase):
    def test_components_split_and_net(self) -> None:
        ts = income_timeseries(_tx([
            ("2026-01-05", "dividend", 100.0),
            ("2026-01-20", "interest", 10.0),
            ("2026-01-25", "withholding", -15.0),
        ]))
        self.assertEqual(len(ts), 1)
        row = ts.iloc[0]
        self.assertAlmostEqual(row["dividends"], 100.0, places=6)
        self.assertAlmostEqual(row["interest"], 10.0, places=6)
        self.assertAlmostEqual(row["withholding"], -15.0, places=6)
        self.assertAlmostEqual(row["net"], 95.0, places=6)

    def test_excludes_reinvestment_and_principal_pmt(self) -> None:
        ts = income_timeseries(_tx([
            ("2026-01-05", "dividend", 100.0),
            ("2026-01-05", "reinvestment", -100.0),
            ("2026-01-06", "principal_pmt", 50.0),
            ("2026-01-07", "buy", -500.0),
        ]))
        self.assertAlmostEqual(ts.iloc[0]["net"], 100.0, places=6)

    def test_groups_by_calendar_month(self) -> None:
        ts = income_timeseries(_tx([
            ("2026-01-31", "dividend", 1.0),
            ("2026-02-01", "dividend", 2.0),
        ]))
        self.assertEqual(len(ts), 2)
        self.assertEqual(ts.index[0], pd.Timestamp("2026-01-01"))
        self.assertEqual(ts.index[1], pd.Timestamp("2026-02-01"))

    def test_zero_amount_rows_are_harmless(self) -> None:
        ts = income_timeseries(_tx([
            ("2026-01-05", "dividend", 100.0),
            ("2026-01-06", "dividend", 0.0),
        ]))
        self.assertAlmostEqual(ts.iloc[0]["dividends"], 100.0, places=6)

    def test_missing_component_column_is_zero(self) -> None:
        ts = income_timeseries(_tx([("2026-01-05", "dividend", 100.0)]))
        self.assertAlmostEqual(ts.iloc[0]["interest"], 0.0, places=6)
        self.assertAlmostEqual(ts.iloc[0]["withholding"], 0.0, places=6)

    def test_by_account_split(self) -> None:
        ts = income_timeseries(_tx([
            ("2026-01-05", "dividend", 100.0, "A"),
            ("2026-01-06", "dividend", 40.0, "B"),
        ]), by="account_id")
        self.assertEqual(len(ts), 2)
        self.assertAlmostEqual(ts.loc[(pd.Timestamp("2026-01-01"), "A"),
                                      "dividends"], 100.0, places=6)
        self.assertAlmostEqual(ts.loc[(pd.Timestamp("2026-01-01"), "B"),
                                      "net"], 40.0, places=6)

    def test_empty_input_returns_empty_frame_with_columns(self) -> None:
        ts = income_timeseries(_tx([("2026-01-07", "buy", -500.0)]))
        self.assertEqual(len(ts), 0)
        for col in ("dividends", "interest", "withholding", "net"):
            self.assertIn(col, ts.columns)

    def test_nat_settlement_falls_back_to_trade_date(self) -> None:
        # WSD-1: a real dividend whose settlement_date is blank/unparseable
        # (interim CSV / some Harbor income rows) must still be counted via its
        # trade_date instead of being silently dropped.
        ts = income_timeseries(_tx_st([
            (None, "2026-01-15", "dividend", 250.00),
            ("2026-01-20", "2026-01-18", "interest", 10.0),
        ]))
        self.assertEqual(len(ts), 1)
        self.assertAlmostEqual(
            ts.loc[pd.Timestamp("2026-01-01"), "dividends"], 250.00, places=6)
        self.assertAlmostEqual(
            ts.loc[pd.Timestamp("2026-01-01"), "interest"], 10.0, places=6)

    def test_settlement_date_wins_when_both_present(self) -> None:
        # Coalescing only fills the gap; a present settlement_date still
        # decides the month (settlement Feb beats trade Jan).
        ts = income_timeseries(_tx_st([
            ("2026-02-02", "2026-01-30", "dividend", 50.0),
        ]))
        self.assertEqual(ts.index[0], pd.Timestamp("2026-02-01"))

    def test_row_with_no_parseable_date_is_dropped(self) -> None:
        # Both dates NaT -> genuinely undateable -> still dropped, as before.
        ts = income_timeseries(_tx_st([
            (None, None, "dividend", 99.0),
            ("2026-01-05", "2026-01-04", "dividend", 100.0),
        ]))
        self.assertEqual(len(ts), 1)
        self.assertAlmostEqual(ts.iloc[0]["dividends"], 100.0, places=6)

    def test_frame_without_trade_date_column_unchanged(self) -> None:
        # Legacy callers pass frames with no trade_date column at all; the
        # coalescing must degrade to settlement-only without erroring.
        ts = income_timeseries(_tx([("2026-01-05", "dividend", 100.0)]))
        self.assertAlmostEqual(ts.iloc[0]["dividends"], 100.0, places=6)


class TestLoadDivHistory(unittest.TestCase):
    def test_loads_parses_and_uppercases_ticker(self) -> None:
        with TemporaryDirectory() as td:
            p = Path(td) / "dividends_spy.csv"
            p.write_text(
                "cash_amount,ex_dividend_date,ticker\n"
                "1.5,2026-03-20,SPY\n"
                "1.6,2025-12-19,SPY\n", encoding="utf-8")
            hist = load_div_history(Path(td))
        self.assertIn("SPY", hist)
        df = hist["SPY"]
        self.assertEqual(len(df), 2)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(
            df["ex_dividend_date"]))
        self.assertAlmostEqual(float(df["cash_amount"].sum()), 3.1,
                               places=6)

    def test_header_only_file_is_known_nonpayer(self) -> None:
        with TemporaryDirectory() as td:
            (Path(td) / "dividends_aaa.csv").write_text(
                "cash_amount,ex_dividend_date,ticker\n", encoding="utf-8")
            hist = load_div_history(Path(td))
        self.assertIn("AAA", hist)
        self.assertTrue(hist["AAA"].empty)

    def test_mixed_garbage_rows_are_dropped(self) -> None:
        with TemporaryDirectory() as td:
            (Path(td) / "dividends_mix.csv").write_text(
                "cash_amount,ex_dividend_date,ticker\n"
                "1.5,2026-03-20,MIX\n"
                ",2026-06-20,MIX\n"          # null amount -> dropped
                "2.0,not-a-date,MIX\n",      # bad date -> dropped
                encoding="utf-8")
            hist = load_div_history(Path(td))
        self.assertEqual(len(hist["MIX"]), 1)
        self.assertAlmostEqual(float(hist["MIX"]["cash_amount"].iloc[0]),
                               1.5, places=6)

    def test_zero_byte_file_is_skipped_not_crash(self) -> None:
        with TemporaryDirectory() as td:
            (Path(td) / "dividends_bad.csv").write_text("", encoding="utf-8")
            hist = load_div_history(Path(td))
        self.assertNotIn("BAD", hist)
        self.assertEqual(hist, {})

    def test_missing_dir_returns_empty(self) -> None:
        hist = load_div_history(Path("Z:/does/not/exist-income-test"))
        self.assertEqual(hist, {})


_EMPTY_HIST = pd.DataFrame(columns=["ex_dividend_date", "cash_amount"])


def _positions(rows):
    """(symbol, asset_class, qty, mv, cost) tuples -> positions frame."""
    if not rows:
        return pd.DataFrame(columns=["symbol", "asset_class", "quantity",
                                     "market_value", "cost_basis"])
    return pd.DataFrame([{
        "symbol": r[0], "asset_class": r[1], "quantity": r[2],
        "market_value": r[3], "cost_basis": r[4],
    } for r in rows])


def _hist(pairs):
    """(ex_date, cash) pairs -> dividend-history frame."""
    df = pd.DataFrame(list(pairs),
                      columns=["ex_dividend_date", "cash_amount"])
    df["ex_dividend_date"] = pd.to_datetime(df["ex_dividend_date"])
    return df


def _hist_t(rows):
    """(ex_date, cash, frequency, dividend_type) tuples -> history frame
    carrying the Polygon special-dividend markers."""
    df = pd.DataFrame(list(rows), columns=[
        "ex_dividend_date", "cash_amount", "frequency", "dividend_type"])
    df["ex_dividend_date"] = pd.to_datetime(df["ex_dividend_date"])
    return df


def _prow(**kw):
    """One positions row as a dict; defaults cover the required columns so
    tests only spell out what they assert on."""
    base = {"symbol": "", "asset_class": "equity_stock", "quantity": 0.0,
            "market_value": 0.0, "cost_basis": 0.0}
    base.update(kw)
    return base


class TestForwardIncome(unittest.TestCase):
    ASOF = date(2026, 6, 10)

    def test_t12m_projection_yield_and_yoc(self) -> None:
        pos = _positions([("SPY", "equity_etf", 100, 60000.0, 50000.0)])
        hist = {"SPY": _hist([("2025-09-19", 1.5), ("2025-12-19", 1.5),
                              ("2026-03-20", 1.5)])}
        df, roll = forward_income(pos, hist, self.ASOF)
        row = df.iloc[0]
        self.assertAlmostEqual(row["t12m_per_share"], 4.5, places=6)
        self.assertAlmostEqual(row["projected"], 450.0, places=6)
        self.assertAlmostEqual(row["yield_mv"], 450.0 / 60000.0, places=9)
        self.assertAlmostEqual(row["yield_cost"], 450.0 / 50000.0, places=9)
        self.assertAlmostEqual(roll["projected_12m"], 450.0, places=6)

    def test_window_edges_inclusive_asof_exclusive_start(self) -> None:
        pos = _positions([("XYZ", "equity_stock", 1, 100.0, 100.0)])
        hist = {"XYZ": _hist([
            ("2026-06-10", 1.0),   # exactly asof -> IN
            ("2025-06-10", 10.0),  # exactly asof-12mo -> OUT (strict >)
            ("2025-06-11", 0.5),   # just inside -> IN
        ])}
        df, _ = forward_income(pos, hist, self.ASOF)
        self.assertAlmostEqual(df.iloc[0]["t12m_per_share"], 1.5, places=6)

    def test_multi_account_rows_summed_before_math(self) -> None:
        pos = _positions([("SPY", "equity_etf", 100, 60000.0, 50000.0),
                          ("SPY", "equity_etf", 50, 30000.0, 30000.0)])
        hist = {"SPY": _hist([("2026-03-20", 2.0)])}
        df, _ = forward_income(pos, hist, self.ASOF)
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df.iloc[0]["projected"], 300.0, places=6)
        self.assertAlmostEqual(df.iloc[0]["yield_mv"],
                               300.0 / 90000.0, places=9)

    def test_nonpayer_covered_vs_missing_uncovered(self) -> None:
        pos = _positions([("AAA", "equity_stock", 10, 1000.0, 800.0),
                          ("BBB", "equity_stock", 10, 2000.0, 1500.0)])
        hist = {"AAA": _EMPTY_HIST.copy()}
        df, roll = forward_income(pos, hist, self.ASOF)
        a = df[df["symbol"] == "AAA"].iloc[0]
        b = df[df["symbol"] == "BBB"].iloc[0]
        self.assertTrue(bool(a["covered"]))
        self.assertAlmostEqual(a["projected"], 0.0, places=6)
        self.assertFalse(bool(b["covered"]))
        # covered MV (1000) includes the non-payer; BBB's 2000 is not covered
        self.assertAlmostEqual(roll["covered_mv"], 1000.0, places=6)

    def test_cash_excluded_unknown_bond_symbol_uncovered(self) -> None:
        # cash never enters; a symbol-bearing bond with no history file and
        # no coupon info shows up uncovered (visible via coverage %).
        pos = _positions([("SPY", "equity_etf", 100, 60000.0, 50000.0),
                          ("", "cash", 0, 10000.0, 10000.0),
                          ("BOND1", "fixed_income", 10, 30000.0, 30000.0)])
        hist = {"SPY": _hist([("2026-03-20", 1.0)])}
        df, roll = forward_income(pos, hist, self.ASOF)
        self.assertEqual(set(df["symbol"]), {"SPY", "BOND1"})
        self.assertFalse(bool(df[df["symbol"] == "BOND1"]
                              .iloc[0]["covered"]))
        self.assertAlmostEqual(roll["nav"], 100000.0, places=6)
        self.assertAlmostEqual(roll["coverage_pct_nav"], 0.6, places=9)

    def test_yoc_nan_when_cost_basis_zero(self) -> None:
        pos = _positions([("SPY", "equity_etf", 100, 60000.0, 0.0)])
        hist = {"SPY": _hist([("2026-03-20", 1.0)])}
        df, _ = forward_income(pos, hist, self.ASOF)
        self.assertTrue(np.isnan(df.iloc[0]["yield_cost"]))

    def test_yield_on_covered_mv_denominator_includes_nonpayers(self) -> None:
        pos = _positions([("SPY", "equity_etf", 100, 60000.0, 50000.0),
                          ("AAA", "equity_stock", 10, 40000.0, 30000.0)])
        hist = {"SPY": _hist([("2026-03-20", 6.0)]),
                "AAA": _EMPTY_HIST.copy()}
        _, roll = forward_income(pos, hist, self.ASOF)
        self.assertAlmostEqual(roll["projected_12m"], 600.0, places=6)
        self.assertAlmostEqual(roll["yield_on_covered_mv"],
                               600.0 / 100000.0, places=9)

    def test_empty_history_dict_zero_coverage(self) -> None:
        pos = _positions([("SPY", "equity_etf", 100, 60000.0, 50000.0)])
        df, roll = forward_income(pos, {}, self.ASOF)
        self.assertFalse(bool(df.iloc[0]["covered"]))
        self.assertAlmostEqual(roll["coverage_pct_nav"], 0.0, places=9)
        self.assertAlmostEqual(roll["projected_12m"], 0.0, places=6)
        self.assertTrue(np.isnan(roll["yield_on_covered_cost"]))

    def test_bonds_only_book_real_nav_zero_coverage(self) -> None:
        # No history file, no description / est_annual_income columns at
        # all: the bond row stays uncovered and nothing crashes.
        pos = _positions([("BOND1", "fixed_income", 10, 30000.0, 30000.0)])
        df, roll = forward_income(pos, {}, self.ASOF)
        self.assertEqual(len(df), 1)
        self.assertFalse(bool(df.iloc[0]["covered"]))
        self.assertAlmostEqual(roll["nav"], 30000.0, places=6)
        self.assertAlmostEqual(roll["coverage_pct_nav"], 0.0, places=9)
        self.assertAlmostEqual(roll["projected_12m"], 0.0, places=6)
        self.assertTrue(np.isnan(roll["yield_on_covered_cost"]))

    def test_rollup_yoc_ignores_zero_cost_holdings(self) -> None:
        pos = _positions([("SPY", "equity_etf", 100, 60000.0, 50000.0),
                          ("ZRO", "equity_stock", 10, 40000.0, 0.0)])
        hist = {"SPY": _hist([("2026-03-20", 5.0)]),
                "ZRO": _hist([("2026-03-20", 10.0)])}
        df, roll = forward_income(pos, hist, self.ASOF)
        # projected_12m counts BOTH (500 + 100)
        self.assertAlmostEqual(roll["projected_12m"], 600.0, places=6)
        # YoC = cost-known only: 500 / 50000, NOT 600 / 50000
        self.assertAlmostEqual(roll["yield_on_covered_cost"],
                               500.0 / 50000.0, places=9)

    def test_empty_positions(self) -> None:
        df, roll = forward_income(_positions([]), {}, self.ASOF)
        self.assertEqual(len(df), 0)
        self.assertTrue(np.isnan(roll["coverage_pct_nav"]))


class TestForwardIncomeSpecials(unittest.TestCase):
    """One-time / special distributions must not be annualized forward."""
    ASOF = date(2026, 6, 10)

    def test_frequency_zero_excluded_from_t12m(self) -> None:
        # The VISN case: Polygon marks the one-time payout frequency=0
        # but dividend_type stays "CD".
        pos = _positions([("VISN", "equity_stock", 2750, 34320.0, 0.0)])
        hist = {"VISN": _hist_t([("2026-04-28", 10.0, 0, "CD"),
                                 ("2026-03-02", 0.05, 4, "CD")])}
        df, roll = forward_income(pos, hist, self.ASOF)
        self.assertAlmostEqual(df.iloc[0]["t12m_per_share"], 0.05, places=6)
        self.assertAlmostEqual(roll["projected_12m"], 0.05 * 2750, places=6)

    def test_dividend_type_sc_excluded_from_t12m(self) -> None:
        # The CME case: annual special is frequency=1 but typed "SC".
        pos = _positions([("CME", "equity_stock", 10, 2700.0, 2000.0)])
        hist = {"CME": _hist_t([("2026-03-10", 6.15, 1, "SC"),
                                ("2026-02-01", 1.25, 4, "CD")])}
        df, _ = forward_income(pos, hist, self.ASOF)
        self.assertAlmostEqual(df.iloc[0]["t12m_per_share"], 1.25, places=6)

    def test_missing_marker_columns_all_rows_kept(self) -> None:
        # Histories without frequency/dividend_type columns (legacy CSVs,
        # spliced frames) keep every in-window row.
        pos = _positions([("SPY", "equity_etf", 100, 60000.0, 50000.0)])
        hist = {"SPY": _hist([("2026-03-20", 1.5), ("2025-12-19", 1.5)])}
        df, _ = forward_income(pos, hist, self.ASOF)
        self.assertAlmostEqual(df.iloc[0]["t12m_per_share"], 3.0, places=6)

    def test_nan_markers_treated_as_regular(self) -> None:
        pos = _positions([("XYZ", "equity_stock", 1, 100.0, 100.0)])
        hist = {"XYZ": _hist_t([("2026-03-20", 1.0, np.nan, np.nan)])}
        df, _ = forward_income(pos, hist, self.ASOF)
        self.assertAlmostEqual(df.iloc[0]["t12m_per_share"], 1.0, places=6)


class TestForwardIncomeEligibility(unittest.TestCase):
    """Dividend-channel candidacy is symbol-driven: broker/display classes
    are unreliable (Harbor files SGOV under fixed income; the app reclasses
    TLH-account rows and gold ETFs). Only options and cash are structural
    non-candidates."""
    ASOF = date(2026, 6, 10)

    def test_fixed_income_class_rows_join_symbol_aggregation(self) -> None:
        # SGOV held at two brokers, one row classed fixed_income: both
        # quantities must be projected.
        pos = _positions([
            ("SGOV", "equity_stock", 1000, 100000.0, 100000.0),
            ("SGOV", "fixed_income", 3980, 400000.0, 0.0)])
        hist = {"SGOV": _hist([("2026-03-02", 0.30), ("2026-04-01", 0.30)])}
        df, roll = forward_income(pos, hist, self.ASOF)
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df.iloc[0]["quantity"], 4980.0, places=6)
        self.assertAlmostEqual(df.iloc[0]["projected"], 0.60 * 4980,
                               places=6)
        self.assertAlmostEqual(roll["covered_mv"], 500000.0, places=6)

    def test_display_only_classes_are_candidates(self) -> None:
        # tax_loss_harvesting and gold are app-side display classes; their
        # payers (or confirmed non-payers) still belong in the model.
        pos = _positions([
            ("DIVR", "tax_loss_harvesting", 10, 1000.0, 800.0),
            ("GLD", "gold", 5, 1500.0, 0.0)])
        hist = {"DIVR": _hist([("2026-03-20", 2.0)]),
                "GLD": _EMPTY_HIST.copy()}
        df, roll = forward_income(pos, hist, self.ASOF)
        d = df[df["symbol"] == "DIVR"].iloc[0]
        g = df[df["symbol"] == "GLD"].iloc[0]
        self.assertAlmostEqual(d["projected"], 20.0, places=6)
        self.assertTrue(bool(g["covered"]))
        self.assertAlmostEqual(g["projected"], 0.0, places=6)
        self.assertAlmostEqual(roll["covered_mv"], 2500.0, places=6)

    def test_option_legs_never_eligible_despite_underlying_symbol(self) -> None:
        # A SPY put carries symbol "SPY" — its contracts must not be
        # multiplied by SPY's per-share dividends.
        pos = _positions([("SPY", "equity_etf", 100, 60000.0, 50000.0),
                          ("SPY", "option_put", 5, 7500.0, 8000.0)])
        hist = {"SPY": _hist([("2026-03-20", 1.0)])}
        df, roll = forward_income(pos, hist, self.ASOF)
        self.assertEqual(len(df), 1)
        self.assertAlmostEqual(df.iloc[0]["quantity"], 100.0, places=6)
        self.assertAlmostEqual(df.iloc[0]["projected"], 100.0, places=6)
        self.assertAlmostEqual(roll["covered_mv"], 60000.0, places=6)

    def test_cash_rows_never_eligible(self) -> None:
        # Sweep/money-market tickers (QJERQ, SPAXX) stay out even if a
        # history file somehow exists for the symbol.
        pos = _positions([("SPY", "equity_etf", 100, 60000.0, 50000.0),
                          ("QJERQ", "cash", 2291, 2291.0, 0.0)])
        hist = {"SPY": _hist([("2026-03-20", 1.0)]),
                "QJERQ": _hist([("2026-03-20", 1.0)])}
        df, roll = forward_income(pos, hist, self.ASOF)
        self.assertNotIn("QJERQ", set(df["symbol"]))
        self.assertAlmostEqual(roll["covered_mv"], 60000.0, places=6)

    def test_partial_cost_symbol_yoc_prorated_to_known_lots(self) -> None:
        # SGOV-style: one lot with cost, one without (Harbor statements omit
        # it). YoC must price only the cost-known lot's income — not the
        # whole symbol's income over the known lot's cost (19% mirage).
        pos = _positions([("SGOV", "equity_stock", 100, 10000.0, 10000.0),
                          ("SGOV", "fixed_income", 300, 30000.0, 0.0)])
        hist = {"SGOV": _hist([("2026-03-02", 1.0)])}
        df, roll = forward_income(pos, hist, self.ASOF)
        row = df.iloc[0]
        # projected covers all 400 shares; YoC covers the 100 with cost
        self.assertAlmostEqual(row["projected"], 400.0, places=6)
        self.assertAlmostEqual(row["yield_cost"], 100.0 / 10000.0,
                               places=9)
        self.assertAlmostEqual(roll["yield_on_covered_cost"],
                               100.0 / 10000.0, places=9)

    def test_nan_symbol_rows_are_not_a_nan_ticker(self) -> None:
        # Bare-CUSIP rows have symbol=NaN; astype(str) would turn that
        # into a "NAN" ticker without the fillna guard.
        pos = pd.DataFrame([{
            "symbol": np.nan, "asset_class": "fixed_income",
            "quantity": 18000.0, "market_value": 17821.0,
            "cost_basis": 17705.0}])
        df, roll = forward_income(pos, {}, self.ASOF)
        self.assertNotIn("NAN", set(df["symbol"].astype(str).str.upper()))
        self.assertAlmostEqual(roll["nav"], 17821.0, places=6)


_UST_DESC = ("UNITED STATES TREASURY NOTE DATED DATE 07/31/2022 "
             "07/31/2029 1.25000% JJ 31")


class TestForwardIncomeCouponChannel(unittest.TestCase):
    """Fixed-income rows without dividend-history coverage project coupon
    income: statement est_annual_income when present, else face x coupon
    parsed from the description."""
    ASOF = date(2026, 6, 10)

    def test_treasury_rung_face_times_coupon(self) -> None:
        pos = pd.DataFrame([_prow(
            symbol="", asset_class="fixed_income", quantity=18000.0,
            market_value=17820.90, cost_basis=17705.16,
            description=_UST_DESC, cusip="91282CZZ9")])
        df, roll = forward_income(pos, {}, self.ASOF)
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["symbol"], "UST 1.25% 07/2029")
        self.assertEqual(row["channel"], "coupon")
        self.assertTrue(bool(row["covered"]))
        self.assertAlmostEqual(row["projected"], 225.0, places=6)
        self.assertTrue(np.isnan(row["t12m_per_share"]))
        self.assertAlmostEqual(row["yield_mv"], 225.0 / 17820.90, places=9)
        self.assertAlmostEqual(roll["projected_12m"], 225.0, places=6)
        self.assertAlmostEqual(roll["covered_mv"], 17820.90, places=6)

    def test_label_uses_maturity_not_dated_date(self) -> None:
        # Description carries both the dated date (2022) and maturity
        # (2029); the label must show the later one.
        pos = pd.DataFrame([_prow(
            symbol="", asset_class="fixed_income", quantity=1000.0,
            market_value=990.0, cost_basis=980.0, description=_UST_DESC)])
        df, _ = forward_income(pos, {}, self.ASOF)
        self.assertIn("07/2029", df.iloc[0]["symbol"])
        self.assertNotIn("07/2022", df.iloc[0]["symbol"])

    def test_est_annual_income_preferred_over_parse(self) -> None:
        pos = pd.DataFrame([_prow(
            symbol="BBB", asset_class="fixed_income", quantity=300.0,
            market_value=15150.0, cost_basis=15000.0,
            description="SYNTHETIC BOND FUND 9.99% NOMINAL",
            est_annual_income=600.0)])
        df, roll = forward_income(pos, {}, self.ASOF)
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["symbol"], "BBB")
        self.assertEqual(row["channel"], "coupon")
        self.assertAlmostEqual(row["projected"], 600.0, places=6)
        self.assertAlmostEqual(roll["projected_12m"], 600.0, places=6)

    def test_covered_symbol_takes_dividend_channel_not_coupon(self) -> None:
        # SGOV-at-Harbor: fixed_income class but the symbol has a history
        # file -> dividend channel only, no duplicate coupon row.
        pos = pd.DataFrame([_prow(
            symbol="SGOV", asset_class="fixed_income", quantity=3980.0,
            market_value=400000.0, cost_basis=0.0,
            description="ISHARES 1.23% LOOKALIKE NOTE TRAP")])
        hist = {"SGOV": _hist([("2026-03-02", 0.30)])}
        df, _ = forward_income(pos, hist, self.ASOF)
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["channel"], "dividend")
        self.assertAlmostEqual(row["projected"], 0.30 * 3980, places=6)

    def test_label_falls_back_to_cusip_for_non_treasury(self) -> None:
        pos = pd.DataFrame([_prow(
            symbol="", asset_class="fixed_income", quantity=10000.0,
            market_value=9900.0, cost_basis=9800.0,
            description="FHLB AGENCY NOTE 5.5% SERIES K",
            cusip="3130AXYZ1")])
        df, _ = forward_income(pos, {}, self.ASOF)
        self.assertEqual(df.iloc[0]["symbol"], "BOND 3130AXYZ1")
        self.assertAlmostEqual(df.iloc[0]["projected"], 550.0, places=6)

    def test_rollup_blends_dividend_and_coupon_channels(self) -> None:
        pos = pd.DataFrame([
            _prow(symbol="SPY", asset_class="equity_etf", quantity=100.0,
                  market_value=60000.0, cost_basis=50000.0),
            _prow(symbol="", asset_class="fixed_income", quantity=18000.0,
                  market_value=17820.90, cost_basis=17705.16,
                  description=_UST_DESC)])
        hist = {"SPY": _hist([("2026-03-20", 1.0)])}
        df, roll = forward_income(pos, hist, self.ASOF)
        self.assertEqual(len(df), 2)
        self.assertAlmostEqual(roll["projected_12m"], 100.0 + 225.0,
                               places=6)
        self.assertAlmostEqual(roll["covered_mv"], 60000.0 + 17820.90,
                               places=6)
        self.assertAlmostEqual(
            roll["yield_on_covered_cost"],
            (100.0 + 225.0) / (50000.0 + 17705.16), places=9)

    def test_dividend_rows_carry_channel_dividend(self) -> None:
        pos = _positions([("SPY", "equity_etf", 100, 60000.0, 50000.0)])
        hist = {"SPY": _hist([("2026-03-20", 1.0)])}
        df, _ = forward_income(pos, hist, self.ASOF)
        self.assertEqual(df.iloc[0]["channel"], "dividend")


class TestForwardIncomeSleeves(unittest.TestCase):
    """Account-level display rollup: accounts managed as one strategy
    (the Treasury ladder, the direct-index TLH book) collapse into a
    single labeled row each. Display-only — the rollup KPIs are always
    computed over the full book."""
    ASOF = date(2026, 6, 10)

    SLEEVES = {"ACC-LAD": "Treasury Ladder", "ACC-TLH": "Tax Loss Harvesting"}

    def _book(self):
        return pd.DataFrame([
            _prow(symbol="SPY", asset_class="equity_etf", quantity=100.0,
                  market_value=60000.0, cost_basis=50000.0,
                  account_id="ACC-CORE"),
            _prow(symbol="AAA", asset_class="equity_stock", quantity=10.0,
                  market_value=1000.0, cost_basis=800.0,
                  account_id="ACC-TLH"),
            _prow(symbol="BBB", asset_class="equity_stock", quantity=20.0,
                  market_value=2000.0, cost_basis=1500.0,
                  account_id="ACC-TLH"),
            _prow(symbol="", asset_class="fixed_income", quantity=18000.0,
                  market_value=17820.90, cost_basis=17705.16,
                  description=_UST_DESC, cusip="91282CZZ9",
                  account_id="ACC-LAD"),
        ])

    def _hist(self):
        return {"SPY": _hist([("2026-03-20", 1.0)]),
                "AAA": _hist([("2026-02-10", 2.0)]),
                "BBB": _hist([("2026-02-10", 3.0)])}

    def test_sleeve_accounts_collapse_to_single_rows(self) -> None:
        df, _ = forward_income(self._book(), self._hist(), self.ASOF,
                               sleeves=self.SLEEVES)
        self.assertEqual(set(df["symbol"]),
                         {"SPY", "Treasury Ladder", "Tax Loss Harvesting"})
        tlh = df[df["symbol"] == "Tax Loss Harvesting"].iloc[0]
        # AAA: 2.0 x 10 = 20; BBB: 3.0 x 20 = 60
        self.assertAlmostEqual(tlh["projected"], 80.0, places=6)
        self.assertAlmostEqual(tlh["market_value"], 3000.0, places=6)
        self.assertAlmostEqual(tlh["cost_basis"], 2300.0, places=6)
        self.assertAlmostEqual(tlh["yield_mv"], 80.0 / 3000.0, places=9)
        self.assertAlmostEqual(tlh["yield_cost"], 80.0 / 2300.0, places=9)
        self.assertTrue(bool(tlh["covered"]))
        self.assertTrue(np.isnan(tlh["t12m_per_share"]))
        self.assertTrue(np.isnan(tlh["quantity"]))
        self.assertEqual(tlh["channel"], "sleeve")
        lad = df[df["symbol"] == "Treasury Ladder"].iloc[0]
        self.assertAlmostEqual(lad["projected"], 225.0, places=6)
        self.assertEqual(lad["channel"], "sleeve")

    def test_projected_total_matches_unsleeved_run(self) -> None:
        plain, _ = forward_income(self._book(), self._hist(), self.ASOF)
        sleeved, _ = forward_income(self._book(), self._hist(), self.ASOF,
                                    sleeves=self.SLEEVES)
        self.assertAlmostEqual(plain["projected"].fillna(0).sum(),
                               sleeved["projected"].fillna(0).sum(),
                               places=6)

    def test_same_symbol_inside_and_outside_sleeve(self) -> None:
        # AAA held in BOTH the TLH account and a core account: the core
        # lot keeps its own AAA row priced on its own quantity; the TLH
        # lot's income lands inside the sleeve.
        book = self._book()
        book = pd.concat([book, pd.DataFrame([_prow(
            symbol="AAA", asset_class="equity_stock", quantity=5.0,
            market_value=500.0, cost_basis=400.0,
            account_id="ACC-CORE")])], ignore_index=True)
        df, _ = forward_income(book, self._hist(), self.ASOF,
                               sleeves=self.SLEEVES)
        aaa = df[df["symbol"] == "AAA"].iloc[0]
        self.assertAlmostEqual(aaa["projected"], 2.0 * 5, places=6)
        tlh = df[df["symbol"] == "Tax Loss Harvesting"].iloc[0]
        self.assertAlmostEqual(tlh["projected"], 80.0, places=6)

    def test_rollup_is_full_book_regardless_of_sleeving(self) -> None:
        _, plain_roll = forward_income(self._book(), self._hist(), self.ASOF)
        _, sleeve_roll = forward_income(self._book(), self._hist(), self.ASOF,
                                        sleeves=self.SLEEVES)
        for k in plain_roll:
            with self.subTest(key=k):
                a, b = plain_roll[k], sleeve_roll[k]
                if isinstance(a, float) and np.isnan(a):
                    self.assertTrue(np.isnan(b))
                else:
                    self.assertAlmostEqual(a, b, places=9)

    def test_none_and_empty_sleeves_are_passthrough(self) -> None:
        plain, _ = forward_income(self._book(), self._hist(), self.ASOF)
        for sleeves in (None, {}):
            out, _ = forward_income(self._book(), self._hist(), self.ASOF,
                                    sleeves=sleeves)
            pd.testing.assert_frame_equal(out, plain)

    def test_sleeve_yield_cost_prices_costed_lots_only(self) -> None:
        # BBB is a carry-forward lot with cost coerced to 0: its income
        # must drop out of BOTH sides of the sleeve yield-on-cost (the
        # rollup YoC lesson) while still counting in projected / MV.
        book = pd.DataFrame([
            _prow(symbol="AAA", asset_class="equity_stock", quantity=10.0,
                  market_value=1000.0, cost_basis=800.0,
                  account_id="ACC-TLH"),
            _prow(symbol="BBB", asset_class="equity_stock", quantity=20.0,
                  market_value=2000.0, cost_basis=0.0,
                  account_id="ACC-TLH"),
        ])
        df, _ = forward_income(book, self._hist(), self.ASOF,
                               sleeves=self.SLEEVES)
        tlh = df.iloc[0]
        self.assertAlmostEqual(tlh["projected"], 80.0, places=6)
        self.assertAlmostEqual(tlh["yield_cost"], 20.0 / 800.0, places=9)
        self.assertAlmostEqual(tlh["yield_mv"], 80.0 / 3000.0, places=9)

    def test_uncovered_holding_in_sleeve_projects_covered_only(self) -> None:
        book = pd.DataFrame([
            _prow(symbol="AAA", asset_class="equity_stock", quantity=10.0,
                  market_value=1000.0, cost_basis=800.0,
                  account_id="ACC-TLH"),
            _prow(symbol="ZZZ", asset_class="equity_stock", quantity=10.0,
                  market_value=5000.0, cost_basis=4000.0,
                  account_id="ACC-TLH"),   # no history file -> uncovered
        ])
        df, roll = forward_income(book, self._hist(), self.ASOF,
                                  sleeves=self.SLEEVES)
        tlh = df[df["symbol"] == "Tax Loss Harvesting"].iloc[0]
        self.assertAlmostEqual(tlh["projected"], 20.0, places=6)
        # The KPI rollup still reports true coverage: ZZZ's MV uncovered.
        self.assertAlmostEqual(roll["covered_mv"], 1000.0, places=6)

    def test_sorted_by_projected_desc(self) -> None:
        df, _ = forward_income(self._book(), self._hist(), self.ASOF,
                               sleeves=self.SLEEVES)
        proj = df["projected"].fillna(-1.0)
        self.assertTrue(proj.is_monotonic_decreasing)

    def test_sleeve_account_with_no_positions_adds_no_row(self) -> None:
        book = self._book()
        book = book[book["account_id"] != "ACC-LAD"]
        df, _ = forward_income(book, self._hist(), self.ASOF,
                               sleeves=self.SLEEVES)
        self.assertNotIn("Treasury Ladder", set(df["symbol"]))

    def test_sleeves_without_account_id_column_raises(self) -> None:
        pos = _positions([("SPY", "equity_etf", 100, 60000.0, 50000.0)])
        with self.assertRaises(ValueError):
            forward_income(pos, self._hist(), self.ASOF,
                           sleeves=self.SLEEVES)


class TestTrailingIncome(unittest.TestCase):
    """trailing_income: net actual income over a rolling day-window on the
    coalesced income date (the true TTM, replacing the month-bucket window
    that dropped the boundary month and leaned on a partial current month —
    WSG-5)."""
    ASOF = pd.Timestamp("2026-06-15")

    def test_sums_net_over_window(self) -> None:
        val = trailing_income(_tx([
            ("2026-06-01", "dividend", 100.0),
            ("2026-05-01", "interest", 10.0),
            ("2026-05-01", "withholding", -5.0),
        ]), self.ASOF)
        self.assertAlmostEqual(val, 105.0, places=6)

    def test_window_start_exclusive_asof_inclusive(self) -> None:
        val = trailing_income(_tx([
            ("2026-06-15", "dividend", 1.0),    # exactly asof -> IN
            ("2025-06-15", "dividend", 10.0),   # exactly asof-365d -> OUT
            ("2025-06-16", "dividend", 0.5),    # just inside -> IN
        ]), self.ASOF)
        self.assertAlmostEqual(val, 1.5, places=6)

    def test_boundary_month_income_not_dropped(self) -> None:
        # WSG-5 regression: the old month-bucket window dropped the prior
        # boundary month wholesale. 2025-06-20 is inside the trailing 365
        # days from 2026-06-15 and must count.
        val = trailing_income(_tx([("2025-06-20", "dividend", 42.0)]),
                              self.ASOF)
        self.assertAlmostEqual(val, 42.0, places=6)

    def test_excludes_reinvestment_and_buys(self) -> None:
        val = trailing_income(_tx([
            ("2026-06-01", "dividend", 100.0),
            ("2026-06-01", "reinvestment", -100.0),
            ("2026-06-02", "buy", -500.0),
        ]), self.ASOF)
        self.assertAlmostEqual(val, 100.0, places=6)

    def test_nat_settlement_counted_via_trade_date(self) -> None:
        # WSD-1 coalescing flows into the trailing window too.
        val = trailing_income(_tx_st([
            (None, "2026-06-01", "dividend", 250.00),
        ]), self.ASOF)
        self.assertAlmostEqual(val, 250.00, places=6)

    def test_empty_returns_zero(self) -> None:
        self.assertEqual(
            trailing_income(_tx([("2026-01-07", "buy", -500.0)]), self.ASOF),
            0.0)

    def test_accepts_python_date(self) -> None:
        # app passes a normalized Timestamp; tests/other callers may pass a
        # datetime.date — both must work.
        val = trailing_income(_tx([("2026-06-01", "dividend", 7.0)]),
                              date(2026, 6, 15))
        self.assertAlmostEqual(val, 7.0, places=6)


class TestLatestExDateThrough(unittest.TestCase):
    """WSD-5: Polygon publishes declared FUTURE ex-dates, so a plain
    max(ex_dividend_date) can claim 'Dividend history through <date>' for a
    date that has not happened yet. latest_ex_date_through clamps to as-of."""

    def _hist(self, dates):
        return pd.DataFrame({"ex_dividend_date": pd.to_datetime(pd.Series(dates, dtype="object"))})

    def test_future_ex_dates_are_excluded(self):
        asof = pd.Timestamp("2026-06-18")
        div_hist = {"SGOV": self._hist(["2026-05-01", "2026-07-15"])}  # July = future
        self.assertEqual(latest_ex_date_through(div_hist, asof),
                         pd.Timestamp("2026-05-01"))

    def test_latest_past_date_across_tickers(self):
        asof = pd.Timestamp("2026-06-18")
        div_hist = {
            "AAA": self._hist(["2026-03-01", "2026-06-10"]),
            "BBB": self._hist(["2026-06-15"]),
        }
        self.assertEqual(latest_ex_date_through(div_hist, asof),
                         pd.Timestamp("2026-06-15"))

    def test_all_future_or_empty_returns_none(self):
        asof = pd.Timestamp("2026-06-18")
        self.assertIsNone(
            latest_ex_date_through({"AAA": self._hist(["2026-09-01"])}, asof))
        self.assertIsNone(
            latest_ex_date_through({"AAA": self._hist([])}, asof))
        self.assertIsNone(latest_ex_date_through({}, asof))


if __name__ == "__main__":
    unittest.main()

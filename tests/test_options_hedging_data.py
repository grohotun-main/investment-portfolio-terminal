"""
Tests for parsers/options_hedging_data.py — pure data-shaping helpers
that feed the Options Hedging tab. Lock down the 15-bug ledger from
commit cfd373c (Phase 2 reliability cleanup).

Run from phase1_build/ with:
    py -m unittest discover tests
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

import options_hedging_data as ohd  # noqa: E402


class TestAsOfHoldings(unittest.TestCase):
    def test_as_of_holdings_slices_to_latest(self) -> None:
        positions = pd.DataFrame([
            {"statement_date": pd.Timestamp("2026-03-31"),
             "symbol": "SPY", "market_value": 100_000.0,
             "asset_class": "equity_etf"},
            {"statement_date": pd.Timestamp("2026-04-30"),
             "symbol": "SPY", "market_value": 110_000.0,
             "asset_class": "equity_etf"},
            {"statement_date": pd.Timestamp("2026-04-30"),
             "symbol": "GLD", "market_value": 50_000.0,
             "asset_class": "other"},
        ])
        out = ohd.as_of_holdings(
            positions, as_of=pd.Timestamp("2026-04-30"),
        )
        self.assertEqual(len(out), 2)
        # Earlier snapshot dropped
        self.assertNotIn(100_000.0, out["market_value"].tolist())
        # symbol renamed to ticker
        self.assertIn("ticker", out.columns)
        # NaN-symbol rows would have empty-string ticker after normalization
        # (covered separately in Treasury CUSIP test)
        self.assertTrue(
            (out["ticker"].astype(str).str.isupper() |
             out["ticker"].astype(str).str.len().eq(0)).all()
        )


    def test_dual_date_month_keeps_both_brokers(self) -> None:
        # WSF-2: Harbor (last-biz-day) and Alpine (month-end) land on different
        # dates in the SAME month. Resolving by month must keep both legs.
        positions = pd.DataFrame([
            {"statement_date": pd.Timestamp("2026-04-29"),  # Harbor
             "symbol": "SPY", "market_value": 100_000.0,
             "asset_class": "equity_etf"},
            {"statement_date": pd.Timestamp("2026-04-30"),  # Alpine
             "symbol": "GLD", "market_value": 50_000.0,
             "asset_class": "other"},
        ])
        out = ohd.as_of_holdings(positions, as_of=pd.Timestamp("2026-04-30"))
        self.assertEqual(len(out), 2, "month-resolution must keep the 04-29 leg")
        self.assertEqual(out["market_value"].sum(), 150_000.0)


class TestClassifyHolding(unittest.TestCase):
    def _row(self, ticker: str, asset_class: str = "equity_etf",
             description: str = "") -> pd.Series:
        return pd.Series({
            "ticker": ticker,
            "asset_class": asset_class,
            "description": description,
            "market_value": 1.0,
        })

    def test_classify_gld_as_equity_when_asset_class_other(self) -> None:
        # GLD was being dropped by the equity_classes whitelist because
        # upstream sometimes tags it asset_class="other". $288k bug.
        self.assertEqual(
            ohd.classify_holding(self._row("GLD", asset_class="other")),
            "equity",
        )

    def test_classify_sgov_as_cash_even_when_tagged_fixed_income(self) -> None:
        # Known-cash-ticker rule overrides asset_class so SGOV stays cash.
        self.assertEqual(
            ohd.classify_holding(
                self._row("SGOV", asset_class="fixed_income")
            ),
            "cash",
        )

    def test_classify_treasury_cusip_as_cash(self) -> None:
        # Treasury bonds have NaN symbol (CUSIP only) — they survive
        # as_of_holdings as empty-string ticker. Must classify via
        # asset_class=fixed_income, not get dropped.
        self.assertEqual(
            ohd.classify_holding(
                self._row("", asset_class="fixed_income",
                          description="UST 4.25% 2027 CUSIP 91282CFB1")
            ),
            "cash",
        )

    def test_classify_option_via_description_regex(self) -> None:
        # synthesize_interim_positions sometimes tags freshly-opened
        # options as asset_class="other"; symbol/description still mark
        # them as PUT/CALL. Belt-and-suspenders detection.
        self.assertEqual(
            ohd.classify_holding(
                self._row("SPY", asset_class="other",
                          description="PUT SPY 11/20/26 640")
            ),
            "option",
        )

    def test_classify_option_via_asset_class(self) -> None:
        self.assertEqual(
            ohd.classify_holding(
                self._row("NVDA", asset_class="option_put",
                          description="")
            ),
            "option",
        )

    def test_classify_plain_equity_defaults_to_equity(self) -> None:
        self.assertEqual(
            ohd.classify_holding(self._row("AAPL", asset_class="equity_stock")),
            "equity",
        )


class TestAggregateByTicker(unittest.TestCase):
    def test_aggregate_by_ticker_collapses_multi_broker(self) -> None:
        holdings = pd.DataFrame([
            {"ticker": "SPY", "market_value": 60_000.0, "broker": "harbor"},
            {"ticker": "SPY", "market_value": 40_000.0, "broker": "alpine"},
            {"ticker": "AAPL", "market_value": 25_000.0, "broker": "harbor"},
        ])
        out = ohd.aggregate_by_ticker(holdings)
        spy_mv = float(out.loc[out["ticker"] == "SPY", "market_value"].iloc[0])
        aapl_mv = float(out.loc[out["ticker"] == "AAPL", "market_value"].iloc[0])
        self.assertAlmostEqual(spy_mv, 100_000.0, places=2)
        self.assertAlmostEqual(aapl_mv, 25_000.0, places=2)
        # SPY collapsed to a single row — not two
        self.assertEqual(len(out[out["ticker"] == "SPY"]), 1)


class TestFilterToPriced(unittest.TestCase):
    def test_filter_to_priced_excludes_unpriced_and_reports_coverage(self) -> None:
        equity = pd.DataFrame([
            {"ticker": "SPY",  "market_value": 100_000.0},
            {"ticker": "AAPL", "market_value":  25_000.0},
            {"ticker": "XYZQQ", "market_value": 15_000.0},  # unpriced
        ])
        priced = {"SPY", "AAPL", "MSFT"}
        out, stats = ohd.filter_to_priced(equity, priced)
        self.assertEqual(set(out["ticker"]), {"SPY", "AAPL"})
        self.assertAlmostEqual(stats["equity_mv_total"], 140_000.0)
        self.assertAlmostEqual(stats["equity_mv_priced"], 125_000.0)
        self.assertAlmostEqual(stats["coverage_pct"], 125_000.0/140_000.0)
        self.assertEqual(stats["n_priced_tickers"], 2)
        self.assertEqual(stats["n_unpriced"], 1)

    def test_filter_to_priced_empty_input_returns_zero_coverage(self) -> None:
        empty = pd.DataFrame(columns=["ticker", "market_value"])
        out, stats = ohd.filter_to_priced(empty, {"SPY"})
        self.assertTrue(out.empty)
        self.assertEqual(stats["equity_mv_total"], 0.0)
        self.assertEqual(stats["equity_mv_priced"], 0.0)
        self.assertEqual(stats["coverage_pct"], 0.0)
        self.assertEqual(stats["n_priced_tickers"], 0)
        self.assertEqual(stats["n_unpriced"], 0)


class TestBuildHoldingsForEngine(unittest.TestCase):
    def test_appends_synthetic_sgov_when_cash_positive(self) -> None:
        equity_priced = pd.DataFrame([
            {"ticker": "SPY",  "market_value": 100_000.0},
            {"ticker": "AAPL", "market_value":  25_000.0},
        ])
        out = ohd.build_holdings_for_engine(equity_priced, cash_total=50_000.0)
        self.assertIn("SGOV", out["ticker"].tolist())
        sgov_mv = float(out.loc[out["ticker"] == "SGOV", "market_value"].iloc[0])
        self.assertAlmostEqual(sgov_mv, 50_000.0)
        # Equity rows preserved verbatim
        self.assertEqual(len(out), 3)

    def test_no_sgov_row_when_cash_zero(self) -> None:
        equity_priced = pd.DataFrame([
            {"ticker": "SPY", "market_value": 100_000.0},
        ])
        out = ohd.build_holdings_for_engine(equity_priced, cash_total=0.0)
        self.assertNotIn("SGOV", out["ticker"].tolist())
        self.assertEqual(len(out), 1)

    def test_no_sgov_row_when_cash_negative(self) -> None:
        # Defensive: negative cash (margin balance) shouldn't conjure SGOV.
        equity_priced = pd.DataFrame([
            {"ticker": "SPY", "market_value": 100_000.0},
        ])
        out = ohd.build_holdings_for_engine(equity_priced, cash_total=-1_000.0)
        self.assertNotIn("SGOV", out["ticker"].tolist())


class TestBuildExistingOptionsRows(unittest.TestCase):
    def test_keeps_broker_lots_separate(self) -> None:
        # Two NVDA $115 puts, same strike + expiry, different brokers.
        # Must come out as TWO entries — not collapsed by key.
        opt_tbl = pd.DataFrame([
            {"underlying": "NVDA", "opt_type": "put",
             "strike": 115.0, "expiry": pd.Timestamp("2026-12-19").date(),
             "quantity": -3, "cost_basis_per_share": 4.50,
             "market_value": 1200.0, "broker": "harbor"},
            {"underlying": "NVDA", "opt_type": "put",
             "strike": 115.0, "expiry": pd.Timestamp("2026-12-19").date(),
             "quantity": -5, "cost_basis_per_share": 4.20,
             "market_value": 2050.0, "broker": "alpine"},
            # A call should be ignored
            {"underlying": "SPY", "opt_type": "call",
             "strike": 600.0, "expiry": pd.Timestamp("2026-12-19").date(),
             "quantity": 1, "cost_basis_per_share": 10.0,
             "market_value": 1100.0, "broker": "harbor"},
        ])
        out = ohd.build_existing_options_rows(opt_tbl)
        self.assertEqual(len(out), 2)
        self.assertEqual({r["underlying"] for r in out}, {"NVDA"})
        # Contracts converted to positive int regardless of short/long sign
        self.assertEqual(sorted(r["contracts"] for r in out), [3, 5])
        # Required keys present
        for r in out:
            self.assertEqual(
                set(r.keys()),
                {"underlying", "strike", "expiry", "contracts",
                 "cost_basis_per_share", "market_value"},
            )

    def test_empty_opt_tbl_returns_empty_list(self) -> None:
        self.assertEqual(
            ohd.build_existing_options_rows(pd.DataFrame()), [],
        )


class TestOrchestrator(unittest.TestCase):
    def _full_book_fixture(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """6-position portfolio touching every bug class:
        - SPY in both brokers (multi-broker dedup, bug #7)
        - GLD with asset_class='other' (bug #2)
        - SGOV with asset_class='fixed_income' (bug #3)
        - Treasury CUSIP, NaN symbol, asset_class='fixed_income' (bug #4)
        - SPY $640 put with asset_class='other' (bug #5)
        - XYZQQ unpriced microcap (bug #14)
        """
        as_of = pd.Timestamp("2026-04-30")
        positions = pd.DataFrame([
            # Multi-broker SPY
            {"statement_date": as_of, "symbol": "SPY",
             "market_value": 60_000.0, "asset_class": "equity_etf",
             "broker": "harbor", "description": "SPDR S&P 500 ETF"},
            {"statement_date": as_of, "symbol": "SPY",
             "market_value": 40_000.0, "asset_class": "equity_etf",
             "broker": "alpine", "description": "SPDR S&P 500 ETF"},
            # GLD tagged 'other'
            {"statement_date": as_of, "symbol": "GLD",
             "market_value": 30_000.0, "asset_class": "other",
             "broker": "harbor", "description": "SPDR Gold Shares"},
            # SGOV tagged fixed_income
            {"statement_date": as_of, "symbol": "SGOV",
             "market_value": 50_000.0, "asset_class": "fixed_income",
             "broker": "alpine", "description": "iShares 0-3 Month T-Bill"},
            # Treasury CUSIP, NaN symbol
            {"statement_date": as_of, "symbol": None,
             "market_value": 20_000.0, "asset_class": "fixed_income",
             "broker": "harbor",
             "description": "UST 4.25% 2027 CUSIP 91282CFB1"},
            # SPY put tagged 'other' via description (bug #5 detection
            # path: classify_holding finds the PUT via description regex
            # even though asset_class is misleading).
            {"statement_date": as_of, "symbol": "SPY 11/26 PUT 640",
             "market_value": 1_500.0, "asset_class": "other",
             "broker": "alpine",
             "description": "PUT SPY 11/20/26 640"},
            # Unpriced microcap
            {"statement_date": as_of, "symbol": "XYZQQ",
             "market_value": 5_000.0, "asset_class": "equity_stock",
             "broker": "harbor", "description": "Tiny Microcap Inc"},
            # Earlier snapshot — should be dropped
            {"statement_date": pd.Timestamp("2026-03-31"),
             "symbol": "SPY", "market_value": 999_999.0,
             "asset_class": "equity_etf", "broker": "harbor",
             "description": "stale"},
        ])
        opt_tbl = pd.DataFrame([
            {"underlying": "SPY", "opt_type": "put",
             "strike": 640.0, "expiry": pd.Timestamp("2026-11-20").date(),
             "quantity": -1, "cost_basis_per_share": 15.0,
             "market_value": 1_500.0, "broker": "alpine"},
        ])
        return positions, opt_tbl

    def test_orchestrator_full_book_reconciles(self) -> None:
        positions, opt_tbl = self._full_book_fixture()
        priced = {"SPY", "GLD", "AAPL"}  # XYZQQ deliberately missing
        inputs = ohd.build_options_hedging_inputs(
            positions=positions, opt_tbl=opt_tbl,
            priced_tickers=priced,
            as_of=pd.Timestamp("2026-04-30"),
        )
        comp = inputs.composition_breakdown
        # portfolio_value = SPY 60 + 40 + GLD 30 + SGOV 50 + Treasury 20
        #                 + SPY put 1.5 + XYZQQ 5 = 206.5k
        self.assertAlmostEqual(comp["portfolio_value"], 206_500.0, places=2)
        # equity = SPY 100 + GLD 30 + XYZQQ 5 = 135k
        self.assertAlmostEqual(comp["equity_mv"], 135_000.0, places=2)
        # cash = SGOV 50 + Treasury 20 = 70k
        self.assertAlmostEqual(comp["cash_mv"], 70_000.0, places=2)
        # options = SPY put 1.5k
        self.assertAlmostEqual(comp["options_mv"], 1_500.0, places=2)
        # holdings_for_engine: priced equity (SPY collapsed + GLD) + SGOV row
        # XYZQQ excluded (unpriced)
        eng = inputs.holdings_for_engine
        self.assertEqual(set(eng["ticker"]), {"SPY", "GLD", "SGOV"})
        spy_mv = float(eng.loc[eng["ticker"] == "SPY", "market_value"].iloc[0])
        self.assertAlmostEqual(spy_mv, 100_000.0)
        sgov_mv = float(eng.loc[eng["ticker"] == "SGOV", "market_value"].iloc[0])
        self.assertAlmostEqual(sgov_mv, 70_000.0)
        # existing_options: 1 SPY put
        self.assertEqual(len(inputs.existing_options), 1)
        self.assertEqual(inputs.existing_options[0]["underlying"], "SPY")
        # coverage_stats reports the unpriced gap
        self.assertEqual(inputs.coverage_stats["n_unpriced"], 1)
        self.assertLess(inputs.coverage_stats["coverage_pct"], 1.0)

    def test_orchestrator_exposes_equity_priced_tickers(self) -> None:
        # The equity universe the recommender hedges = priced equity names
        # only. SPY collapses across brokers; GLD survives; the unpriced
        # XYZQQ microcap is excluded.
        positions, opt_tbl = self._full_book_fixture()
        inputs = ohd.build_options_hedging_inputs(
            positions=positions, opt_tbl=opt_tbl,
            priced_tickers={"SPY", "GLD", "AAPL"},  # XYZQQ missing
            as_of=pd.Timestamp("2026-04-30"),
        )
        self.assertEqual(set(inputs.equity_priced_tickers), {"SPY", "GLD"})

    def test_equity_priced_tickers_excludes_synthetic_sgov(self) -> None:
        # The synthetic SGOV cash sentinel lives in holdings_for_engine but
        # must NOT leak into the equity universe. This field replaces the
        # brittle `holdings_for_engine["ticker"] != "SGOV"` wrapper filter.
        positions, opt_tbl = self._full_book_fixture()
        inputs = ohd.build_options_hedging_inputs(
            positions=positions, opt_tbl=opt_tbl,
            priced_tickers={"SPY", "GLD", "AAPL"},
            as_of=pd.Timestamp("2026-04-30"),
        )
        self.assertIn("SGOV", inputs.holdings_for_engine["ticker"].tolist())
        self.assertNotIn("SGOV", inputs.equity_priced_tickers)

    def test_orchestrator_no_cash_no_sgov_row(self) -> None:
        as_of = pd.Timestamp("2026-04-30")
        positions = pd.DataFrame([
            {"statement_date": as_of, "symbol": "SPY",
             "market_value": 100_000.0, "asset_class": "equity_etf",
             "broker": "harbor", "description": ""},
            {"statement_date": as_of, "symbol": "AAPL",
             "market_value": 25_000.0, "asset_class": "equity_stock",
             "broker": "harbor", "description": ""},
        ])
        inputs = ohd.build_options_hedging_inputs(
            positions=positions, opt_tbl=pd.DataFrame(),
            priced_tickers={"SPY", "AAPL"}, as_of=as_of,
        )
        self.assertNotIn("SGOV", inputs.holdings_for_engine["ticker"].tolist())
        self.assertAlmostEqual(inputs.composition_breakdown["cash_mv"], 0.0)

    def test_orchestrator_empty_opt_tbl_returns_empty_existing_options(self) -> None:
        as_of = pd.Timestamp("2026-04-30")
        positions = pd.DataFrame([
            {"statement_date": as_of, "symbol": "SPY",
             "market_value": 100_000.0, "asset_class": "equity_etf",
             "broker": "harbor", "description": ""},
        ])
        inputs = ohd.build_options_hedging_inputs(
            positions=positions, opt_tbl=pd.DataFrame(),
            priced_tickers={"SPY"}, as_of=as_of,
        )
        self.assertEqual(inputs.existing_options, [])


if __name__ == "__main__":
    unittest.main()

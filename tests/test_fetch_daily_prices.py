"""Tests for parsers/fetch_daily_prices.py symbol-collection helpers.

The three `collect_*_symbols` functions decide which tickers `fetch_daily`
will pull from Polygon. Coverage matters here because:

  * `collect_equity_symbols` decides whether a previously-held but now-sold
    position keeps its history in `daily_prices.csv` — needed by any
    risk-tab analytic that re-includes the holding period.
  * `collect_prior_symbols` powers TICKER_HISTORY rename-splicing — without
    fetching the prior ticker, a renamed position shows up with truncated
    history (artificially low vol, near-zero correlation).

`fetch_daily_history` is HTTP-bound and not covered here.
"""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))
sys.path.insert(0, str(ROOT))  # for config_local

import config_local  # noqa: E402
import fetch_daily_prices as fdp  # noqa: E402
from fetch_daily_prices import (  # noqa: E402
    collect_equity_symbols,
    collect_prior_symbols,
)


def _write_positions(td: Path, rows: list[dict]) -> Path:
    csv = td / "positions.csv"
    pd.DataFrame(rows).to_csv(csv, index=False)
    return csv


def _pos(date: str, symbol, asset_class: str = "equity_stock",
         account_id: str = "TEST-1") -> dict:
    return {"statement_date": date, "account_id": account_id,
            "symbol": symbol, "asset_class": asset_class,
            "market_value": 1000.0}


class TestCollectEquitySymbols(unittest.TestCase):
    def test_returns_all_equity_class_symbols_across_dates(self) -> None:
        with TemporaryDirectory() as td:
            csv = _write_positions(Path(td), [
                _pos("2026-01-31", "AAPL"),
                _pos("2026-02-28", "MSFT"),
                _pos("2026-02-28", "AAPL"),
            ])
            with patch.object(fdp, "POSITIONS_CSV", csv):
                syms = collect_equity_symbols()
        self.assertEqual(syms, ["AAPL", "MSFT"])

    def test_latest_only_restricts_to_max_statement_date(self) -> None:
        # AAPL held 2026-01 then sold; MSFT bought 2026-02. latest_only=True
        # excludes AAPL because it isn't in the latest snapshot.
        with TemporaryDirectory() as td:
            csv = _write_positions(Path(td), [
                _pos("2026-01-31", "AAPL"),
                _pos("2026-02-28", "MSFT"),
            ])
            with patch.object(fdp, "POSITIONS_CSV", csv):
                syms = collect_equity_symbols(latest_only=True)
        self.assertEqual(syms, ["MSFT"])

    def test_non_equity_class_excluded(self) -> None:
        # Options, cash, and other non-equity asset_class rows must not
        # leak into the fetch universe.
        with TemporaryDirectory() as td:
            csv = _write_positions(Path(td), [
                _pos("2026-02-28", "SPY", asset_class="equity_etf"),
                _pos("2026-02-28", "FCASH", asset_class="cash"),
                _pos("2026-02-28", "SPY240321P450",
                     asset_class="option_put"),
            ])
            with patch.object(fdp, "POSITIONS_CSV", csv):
                syms = collect_equity_symbols()
        self.assertEqual(syms, ["SPY"])

    def test_nan_or_empty_symbols_excluded(self) -> None:
        # Bare-CUSIP bond rows have symbol=NaN; the fetcher must skip them
        # rather than passing NaN to Polygon.
        with TemporaryDirectory() as td:
            csv = _write_positions(Path(td), [
                _pos("2026-02-28", "SPY"),
                _pos("2026-02-28", None),       # bond CUSIP-only row
                _pos("2026-02-28", "   "),      # whitespace only
            ])
            with patch.object(fdp, "POSITIONS_CSV", csv):
                syms = collect_equity_symbols()
        self.assertEqual(syms, ["SPY"])


class TestCollectPriorSymbols(unittest.TestCase):
    def test_returns_priors_from_ticker_history(self) -> None:
        history = {
            "VISN": [{"prior_symbol": "COMM", "effective_date": "2026-01-14"}],
            "META": [{"prior_symbol": "FB",   "effective_date": "2022-06-09"}],
        }
        with patch.object(config_local, "TICKER_HISTORY", history, create=True):
            priors = collect_prior_symbols()
        self.assertEqual(priors, ["COMM", "FB"])

    def test_multi_segment_chain_all_priors_returned(self) -> None:
        # CCC was BBB before 2022-01-03 and AAA before 2020-01-02.
        # Both prior symbols must appear in the fetch universe.
        history = {
            "CCC": [
                {"prior_symbol": "BBB", "effective_date": "2022-01-03"},
                {"prior_symbol": "AAA", "effective_date": "2020-01-02"},
            ],
        }
        with patch.object(config_local, "TICKER_HISTORY", history, create=True):
            priors = collect_prior_symbols()
        self.assertEqual(priors, ["AAA", "BBB"])

    def test_empty_history_returns_empty_list(self) -> None:
        with patch.object(config_local, "TICKER_HISTORY", {}, create=True):
            self.assertEqual(collect_prior_symbols(), [])

    def test_blank_prior_symbol_skipped(self) -> None:
        # Defensive: if the user typoes an entry with a blank prior_symbol,
        # the function should silently drop it rather than emit an empty
        # ticker to Polygon.
        history = {"XYZ": [{"prior_symbol": "", "effective_date": "2025-01-01"}]}
        with patch.object(config_local, "TICKER_HISTORY", history, create=True):
            self.assertEqual(collect_prior_symbols(), [])


if __name__ == "__main__":
    unittest.main()

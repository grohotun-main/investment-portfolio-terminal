"""Tests for the pure classifiers in parsers/fetch_holding_prices.py.

`fetch_latest_close` is HTTP-bound and not covered here (would need requests
mocking; low payoff vs effort for a small Polygon wrapper). The two pure
classifiers below decide which symbols feed into the MTM pipeline and how
they're labelled — the bucketing is what determines whether a price-fetch
attempt happens.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from fetch_holding_prices import _classify, _norm_symbol  # noqa: E402


class TestNormSymbol(unittest.TestCase):
    def test_string_passes_through(self) -> None:
        self.assertEqual(_norm_symbol("SPY"), "SPY")

    def test_whitespace_stripped(self) -> None:
        self.assertEqual(_norm_symbol("  VTI  "), "VTI")

    def test_empty_string_returns_none(self) -> None:
        self.assertIsNone(_norm_symbol(""))
        self.assertIsNone(_norm_symbol("   "))

    def test_non_string_returns_none(self) -> None:
        # Bare-CUSIP bond rows arrive with symbol=NaN; the classifier must
        # not crash and must surface this as None for the caller to skip.
        self.assertIsNone(_norm_symbol(float("nan")))
        self.assertIsNone(_norm_symbol(None))


class TestClassify(unittest.TestCase):
    def test_cash_class(self) -> None:
        self.assertEqual(_classify("cash", "FCASH"), "cash")

    def test_option_class_starts_with_option(self) -> None:
        # asset_class is "option_call" / "option_put" downstream; the
        # classifier uses startswith() so both flavors map to "option".
        self.assertEqual(_classify("option_call", "SPY240321P450"), "option")
        self.assertEqual(_classify("option_put", "VIX260618P20"), "option")

    def test_fixed_income_with_ticker_treated_as_equity(self) -> None:
        # Bond ETFs (SGOV, TLT, SCHO, ...) are fetchable like equities.
        # Bare-CUSIP Treasuries arrive with symbol=NaN and are filtered
        # before reaching the classifier.
        self.assertEqual(_classify("fixed_income", "SGOV"), "equity")

    def test_plain_equity(self) -> None:
        self.assertEqual(_classify("equity", "AAPL"), "equity")

    def test_case_insensitive_class(self) -> None:
        # Source PDFs occasionally normalize asset_class with different
        # capitalization. The classifier lowercases first.
        self.assertEqual(_classify("CASH", "FCASH"), "cash")
        self.assertEqual(_classify("Option_Put", "SPY240321P450"), "option")


if __name__ == "__main__":
    unittest.main()

"""Tests for parsers/asset_reclass.py::reclass_asset.

Asset-class overrides applied at display time, after the statement parser's
section-header classification. Two reasons the broker tag can't be trusted
verbatim:

  * Commodity tickers (GLD, IAU, ...) are tagged inconsistently across sources
    (JPM "other", Fidelity "fixed_income"); they collapse to ``gold``.
  * Fidelity's May-2026 statement format dropped its "Exchange Traded Products"
    section and listed ETFs (SPY, SGOV) under "Stocks / Common Stock", so the
    holdings parser tagged them ``equity_stock``. A ticker->class override map
    (``etf_class``) restores the right class regardless of which section a given
    month's statement happened to file the security under.

Precedence (first match wins): TLH account -> commodity ticker -> phase-0
``commodity_etf`` -> ``etf_class`` map -> the broker's tag unchanged.
"""
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from asset_reclass import reclass_asset  # noqa: E402

# A representative user override map (the real one lives in config_local).
ETF = {"SPY": "equity_etf", "JEPQ": "equity_etf", "SGOV": "fixed_income"}
TLH = "TLH-00000"


class TestReclassAsset(unittest.TestCase):
    def _r(self, account_id, symbol, asset_class, etf=ETF):
        return reclass_asset(account_id, symbol, asset_class,
                             tlh_account_id=TLH, etf_class=etf)

    # --- The regression: an ETF the broker filed under Common Stock --------
    def test_etf_misfiled_as_stock_is_reclassed_to_etf(self):
        self.assertEqual(self._r("X10-000007", "SPY", "equity_stock"),
                         "equity_etf")

    def test_bond_etf_misfiled_as_stock_becomes_fixed_income(self):
        self.assertEqual(self._r("X10-000007", "SGOV", "equity_stock"),
                         "fixed_income")

    def test_override_is_idempotent_when_statement_already_correct(self):
        # A later statement that files SPY under Equity ETPs again: no-op.
        self.assertEqual(self._r("X10-000007", "SPY", "equity_etf"),
                         "equity_etf")

    # --- Untouched cases ---------------------------------------------------
    def test_unmapped_individual_stock_passes_through(self):
        self.assertEqual(self._r("X10-000007", "CLS", "equity_stock"),
                         "equity_stock")

    def test_option_on_an_etf_underlying_is_not_reclassed(self):
        # A SPY put/call carries the underlying's symbol ("SPY") but an option
        # asset_class. The ETF override keys on symbol, so it must NOT clobber
        # the option leg into equity_etf (would corrupt the options slice and
        # the hedging tab).
        self.assertEqual(self._r("X10-000007", "SPY", "option_put"),
                         "option_put")
        self.assertEqual(self._r("X10-000007", "SPY", "option_call"),
                         "option_call")

    def test_cash_row_with_mapped_symbol_stays_cash(self):
        # Defensive: only equity_stock (the broker's mis-file) is upgraded.
        self.assertEqual(self._r("X10-000007", "SPY", "cash"), "cash")

    def test_empty_override_map_is_a_no_op(self):
        self.assertEqual(self._r("X10-000007", "SPY", "equity_stock", etf={}),
                         "equity_stock")

    # --- Precedence: existing commodity / TLH rules still win --------------
    def test_commodity_ticker_overrides_even_when_tagged_stock(self):
        # GLD (not in etf_class) -> gold regardless of the broker tag.
        self.assertEqual(self._r("X10-000007", "GLD", "equity_stock"), "gold")

    def test_phase0_commodity_etf_still_collapses_to_gold(self):
        self.assertEqual(self._r("X10-000007", "IAU", "commodity_etf"), "gold")

    def test_tlh_account_forces_tax_loss_harvesting_over_etf_map(self):
        self.assertEqual(self._r(TLH, "SPY", "equity_etf"),
                         "tax_loss_harvesting")

    # --- Hygiene: symbol normalization / non-string symbols ----------------
    def test_symbol_is_normalized_for_case_and_whitespace(self):
        self.assertEqual(self._r("X10-000007", "  spy ", "equity_stock"),
                         "equity_etf")

    def test_non_string_symbol_passes_through_without_raising(self):
        # Cash-sweep rows carry NaN symbols; must not raise.
        self.assertEqual(self._r("X10-000007", float("nan"), "cash"), "cash")

    # --- WSD-3: broker display-format option legs --------------------------
    # Interim JPM option activity arrives as e.g. "SPY DEC 26 PUT 650.00" — not
    # OCC format — so the synthesizer's OCC regex misses it and books the leg
    # asset_class 'other', which then slips into the income dividend universe.
    # reclass_asset must map these to option_put/option_call so the engine's
    # NON_INCOME_CLASSES exclusion catches them.
    def test_display_format_put_reclassed_to_option_put(self):
        self.assertEqual(
            self._r("ACC-1", "SPY DEC 26 PUT 650.00", "other"), "option_put")

    def test_display_format_call_reclassed_to_option_call(self):
        self.assertEqual(
            self._r("ACC-1", "OKTA MAY 26 CALL 115.00", "other"), "option_call")

    def test_plain_ticker_stays_other(self):
        self.assertEqual(self._r("ACC-1", "SPY", "other"), "other")

    def test_bondish_other_symbol_stays_other(self):
        self.assertEqual(
            self._r("ACC-1", "UST NOTE 4.25 2030", "other"), "other")

    def test_already_classed_option_not_clobbered(self):
        # Guarded on 'other' so a correctly-classed row is left alone.
        self.assertEqual(
            self._r("ACC-1", "SPY DEC 26 PUT 650.00", "option_put"), "option_put")


if __name__ == "__main__":
    unittest.main()

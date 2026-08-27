"""
Phase 1A audit follow-up (item #13): lock the assumption that the
Polygon price-fetch path uses adjusted=true (splits-only — NOT dividend-
adjusted).

The downstream pipeline relies on this distinction: portfolio daily
returns synthesized from daily_prices.csv are PRICE returns only, and
dividends arrive separately as DIVIDEND / INTEREST cash flows in the
transactions stream. If the fetch ever flipped to adjusted=false (or
Polygon ever changed adjusted=true semantics to also include dividends),
the daily-return synthesis would silently double-count dividends — once
via the price-side adjustment + once via the transaction-stream rows.

This test is the regression guard. It does not call Polygon; it inspects
the source of fetch_daily_history and asserts the parameter literal is
still ``"adjusted": "true"``.
"""
import inspect
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))
sys.path.insert(0, str(ROOT))

import fetch_benchmark as fb
import fetch_daily_prices as fdp
import fetch_long_history as flh
import fetch_holding_prices as fhp


class TestPolygonFetchUsesAdjustedTrue(unittest.TestCase):
    """All three price fetchers must pass adjusted=true to Polygon. The
    assertion message documents WHY changing it breaks the downstream
    pipeline so a future engineer hitting this failure understands the
    invariant rather than just disabling the test.
    """

    _WHY = (
        "Polygon's adjusted=true is splits-only (verified empirically "
        "per the fetch_daily_prices.py docstring). The downstream "
        "pipeline relies on this so price returns DON'T include "
        "dividends — those arrive separately as DIVIDEND / INTEREST "
        "transactions in the transactions stream. Changing this to "
        "false (or Polygon ever returning dividend-adjusted prices "
        "under adjusted=true) would double-count dividends in the "
        "daily-return synthesis."
    )

    def test_fetch_daily_history_uses_adjusted_true(self) -> None:
        src = inspect.getsource(fdp.fetch_daily_history)
        self.assertIn('"adjusted": "true"', src,
                      f"fetch_daily_history must pass adjusted=true. {self._WHY}")

    def test_fetch_latest_close_uses_adjusted_true(self) -> None:
        src = inspect.getsource(fhp.fetch_latest_close)
        self.assertIn('"adjusted": "true"', src,
                      f"fetch_latest_close must pass adjusted=true. {self._WHY}")

    def test_fetch_benchmark_uses_adjusted_true(self) -> None:
        # fetch_benchmark drives benchmark_spy.csv → benchmark_spy_tr.csv,
        # which is the SPY series the dashboard reindexes onto daily_prices
        # for β/α and the bench-tab spread. If this ever flips to false (or
        # Polygon redefines adjusted=true to include dividends), the TR
        # construction in build_benchmark_total_return double-counts
        # dividends: once via the price-side adjustment + once via the
        # explicit dividend-reinvest math. ~+1.5%/yr drift.
        src = inspect.getsource(fb.fetch_daily)
        self.assertIn('"adjusted": "true"', src,
                      f"fetch_benchmark.fetch_daily must pass "
                      f"adjusted=true. {self._WHY}")


class TestFetchLongHistoryHonorsTickerAlias(unittest.TestCase):
    """Phase 1A audit follow-up (item #6): fetch_long_history.py was the
    only one of the three price fetchers that did not apply the
    statement-ticker → Polygon-ticker alias lookup (BRKB → BRK.B etc.).
    Currently dormant because the macro universe (SPY, SGOV, GLD, BIL)
    needs no alias, but trips silently if a Class-B share gets added.
    """

    def test_main_loop_applies_ticker_aliases(self) -> None:
        src = inspect.getsource(flh.main)
        self.assertIn("TICKER_ALIASES.get(", src,
                      "fetch_long_history.main must look up the Polygon "
                      "symbol via TICKER_ALIASES before calling "
                      "fetch_daily_history, matching fetch_daily_prices "
                      "and fetch_holding_prices. Without it, a Class-B "
                      "share added to TICKERS would silently 404 against "
                      "Polygon (statement ticker BRKB doesn't exist; the "
                      "Polygon symbol is BRK.B).")

    def test_ticker_aliases_importable_from_long_history_module(self) -> None:
        # If the import line gets dropped during a refactor this test
        # surfaces it before the regression hits a real run.
        self.assertTrue(hasattr(flh, "TICKER_ALIASES"))


if __name__ == "__main__":
    unittest.main()

"""
Regression tests for parsers/mark_to_market.py.

The audit pass (PR #24, 2026-05-19) fixed a bug where the live-quote overlay
overwrote `price` and `market_value` on the latest snapshot but left
`unrealized_gl` at its statement-date value, so Holdings showed contradictory
$ and % G/L columns. These tests pin that fix and the surrounding invariants:

  - only the latest statement_date is touched (historicals stay pure)
  - only `ok` / `cash_fixed_1` price rows contribute
  - missing prices and NaN cost_basis are graceful no-ops
  - unrealized_gl is consistent with the overwritten market_value
"""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from mark_to_market import mark_to_market  # noqa: E402


def _positions(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["statement_date", "symbol", "quantity",
                                     "price", "market_value", "cost_basis",
                                     "unrealized_gl"])
    df = pd.DataFrame(rows)
    df["statement_date"] = pd.to_datetime(df["statement_date"])
    return df


def _prices(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestMarkToMarket(unittest.TestCase):
    def test_overwrites_latest_and_recomputes_gl(self) -> None:
        # Snapshot price 100, live price 110, 10 shares, basis $900 → live G/L = +$200.
        positions = _positions([
            {"statement_date": "2026-04-30", "symbol": "AAA",
             "quantity": 10.0, "price": 100.0, "market_value": 1000.0,
             "cost_basis": 900.0, "unrealized_gl": 100.0},
        ])
        prices = _prices([{"symbol": "AAA", "close": 110.0, "status": "ok"}])
        out = mark_to_market(positions, prices)
        self.assertAlmostEqual(out.at[0, "price"], 110.0)
        self.assertAlmostEqual(out.at[0, "market_value"], 1100.0)
        self.assertAlmostEqual(out.at[0, "unrealized_gl"], 200.0)

    def test_historical_rows_untouched(self) -> None:
        # Same symbol on two snapshot dates; only the latest must be overwritten.
        positions = _positions([
            {"statement_date": "2026-03-31", "symbol": "AAA",
             "quantity": 10.0, "price": 80.0, "market_value": 800.0,
             "cost_basis": 900.0, "unrealized_gl": -100.0},
            {"statement_date": "2026-04-30", "symbol": "AAA",
             "quantity": 10.0, "price": 100.0, "market_value": 1000.0,
             "cost_basis": 900.0, "unrealized_gl": 100.0},
        ])
        prices = _prices([{"symbol": "AAA", "close": 110.0, "status": "ok"}])
        out = mark_to_market(positions, prices)
        # March row identical to input
        mar = out[out["statement_date"] == pd.Timestamp("2026-03-31")].iloc[0]
        self.assertAlmostEqual(mar["price"], 80.0)
        self.assertAlmostEqual(mar["market_value"], 800.0)
        self.assertAlmostEqual(mar["unrealized_gl"], -100.0)
        # April row overwritten
        apr = out[out["statement_date"] == pd.Timestamp("2026-04-30")].iloc[0]
        self.assertAlmostEqual(apr["price"], 110.0)
        self.assertAlmostEqual(apr["market_value"], 1100.0)
        self.assertAlmostEqual(apr["unrealized_gl"], 200.0)

    def test_missing_price_keeps_statement_values(self) -> None:
        # A symbol not in the prices file (typical for bonds/options) must
        # pass through unchanged — no crash, no NaN injection.
        positions = _positions([
            {"statement_date": "2026-04-30", "symbol": "BOND",
             "quantity": 10.0, "price": 102.5, "market_value": 1025.0,
             "cost_basis": 1000.0, "unrealized_gl": 25.0},
        ])
        prices = _prices([{"symbol": "AAA", "close": 110.0, "status": "ok"}])
        out = mark_to_market(positions, prices)
        self.assertAlmostEqual(out.at[0, "price"], 102.5)
        self.assertAlmostEqual(out.at[0, "market_value"], 1025.0)
        self.assertAlmostEqual(out.at[0, "unrealized_gl"], 25.0)

    def test_nan_cost_basis_does_not_recompute_gl(self) -> None:
        # Positions without a cost basis (e.g. some cash sweep rows) must
        # have price + MV overwritten but unrealized_gl LEFT ALONE — otherwise
        # `new_mv - NaN` would silently NaN-pollute the column.
        positions = _positions([
            {"statement_date": "2026-04-30", "symbol": "CASH",
             "quantity": 1000.0, "price": 1.0, "market_value": 1000.0,
             "cost_basis": np.nan, "unrealized_gl": 0.0},
        ])
        prices = _prices([{"symbol": "CASH", "close": 1.0, "status": "cash_fixed_1"}])
        out = mark_to_market(positions, prices)
        self.assertAlmostEqual(out.at[0, "price"], 1.0)
        self.assertAlmostEqual(out.at[0, "market_value"], 1000.0)
        self.assertAlmostEqual(out.at[0, "unrealized_gl"], 0.0)

    def test_skips_non_ok_status_rows(self) -> None:
        # `status` filter: only ok / cash_fixed_1 contribute. A "stale" or
        # "not_found" row in prices_latest must not push a number into MV.
        positions = _positions([
            {"statement_date": "2026-04-30", "symbol": "AAA",
             "quantity": 10.0, "price": 100.0, "market_value": 1000.0,
             "cost_basis": 900.0, "unrealized_gl": 100.0},
        ])
        prices = _prices([{"symbol": "AAA", "close": 999.0, "status": "stale"}])
        out = mark_to_market(positions, prices)
        # No change — status filtered out.
        self.assertAlmostEqual(out.at[0, "price"], 100.0)
        self.assertAlmostEqual(out.at[0, "market_value"], 1000.0)
        self.assertAlmostEqual(out.at[0, "unrealized_gl"], 100.0)

    def test_empty_inputs_are_noops(self) -> None:
        empty_pos = _positions([])
        prices = _prices([{"symbol": "AAA", "close": 1.0, "status": "ok"}])
        self.assertTrue(mark_to_market(empty_pos, prices).empty)

        positions = _positions([
            {"statement_date": "2026-04-30", "symbol": "AAA",
             "quantity": 10.0, "price": 100.0, "market_value": 1000.0,
             "cost_basis": 900.0, "unrealized_gl": 100.0},
        ])
        empty_prices = _prices([])
        out = mark_to_market(positions, empty_prices)
        # Returned unchanged — same identity wouldn't be guaranteed but
        # contents must equal the input.
        self.assertAlmostEqual(out.at[0, "price"], 100.0)
        self.assertAlmostEqual(out.at[0, "market_value"], 1000.0)

    def test_skips_option_rows(self) -> None:
        """Options carry their underlying ticker in `symbol`, but their
        market value is per-contract premium × 100, not qty × spot. The
        overlay must skip option rows so the underlying close doesn't
        bleed in as a fake option MV (e.g. NVDA 135P × 11 contracts
        would otherwise display as 11 × $215 = $2,369 instead of the
        statement's $2,585).
        """
        positions = _positions([
            {"statement_date": "2026-04-30", "symbol": "NVDA",
             "asset_class": "option_put",
             "quantity": 11.0, "price": 2.35, "market_value": 2585.0,
             "cost_basis": 3286.80, "unrealized_gl": -701.80},
            {"statement_date": "2026-04-30", "symbol": "SPY",
             "asset_class": "option_put",
             "quantity": 6.0, "price": 8.20, "market_value": 4920.0,
             "cost_basis": 5501.10, "unrealized_gl": -581.10},
            # Equity row with same ticker — must still be marked.
            {"statement_date": "2026-04-30", "symbol": "NVDA",
             "asset_class": "equity_stock",
             "quantity": 100.0, "price": 200.0, "market_value": 20000.0,
             "cost_basis": 18000.0, "unrealized_gl": 2000.0},
        ])
        prices = _prices([
            {"symbol": "NVDA", "close": 215.33, "status": "ok"},
            {"symbol": "SPY",  "close": 745.64, "status": "ok"},
        ])
        out = mark_to_market(positions, prices)
        # Option rows unchanged — statement MV preserved.
        self.assertAlmostEqual(out.at[0, "market_value"], 2585.0)
        self.assertAlmostEqual(out.at[0, "price"], 2.35)
        self.assertAlmostEqual(out.at[1, "market_value"], 4920.0)
        self.assertAlmostEqual(out.at[1, "price"], 8.20)
        # Equity row with same ticker still gets the live mark.
        self.assertAlmostEqual(out.at[2, "market_value"], 21533.0)
        self.assertAlmostEqual(out.at[2, "price"], 215.33)


class TestStatementBasisStash(unittest.TestCase):
    """The pre-mark stash column (`market_value_stmt`).

    The Data Health reconciliation compares extracted vs the statement's
    reported total on the STATEMENT basis (same as the ingest gate). Both UIs
    hand it frames whose latest snapshot was already re-marked to live prices,
    so mark_to_market must stash the pre-mark market_value in
    `market_value_stmt` — otherwise a real market move since the statement
    date reads as reconciliation drift (the 2026-07-13 false-ERROR incident).
    """

    def test_stashes_pre_mark_market_value_on_all_rows(self) -> None:
        # Historical row, marked latest row, and an unmatched latest row: the
        # stash must equal the pre-mark value on every one of them.
        positions = _positions([
            {"statement_date": "2026-03-31", "symbol": "AAA",
             "quantity": 10.0, "price": 80.0, "market_value": 800.0,
             "cost_basis": 900.0, "unrealized_gl": -100.0},
            {"statement_date": "2026-04-30", "symbol": "AAA",
             "quantity": 10.0, "price": 100.0, "market_value": 1000.0,
             "cost_basis": 900.0, "unrealized_gl": 100.0},
            {"statement_date": "2026-04-30", "symbol": "BOND",
             "quantity": 10.0, "price": 102.5, "market_value": 1025.0,
             "cost_basis": 1000.0, "unrealized_gl": 25.0},
        ])
        prices = _prices([{"symbol": "AAA", "close": 110.0, "status": "ok"}])
        out = mark_to_market(positions, prices)
        self.assertIn("market_value_stmt", out.columns)
        self.assertEqual(list(out["market_value_stmt"]), [800.0, 1000.0, 1025.0])
        # ... while the marked row's live market_value did move.
        self.assertAlmostEqual(out.at[1, "market_value"], 1100.0)

    def test_remark_keeps_the_original_stash(self) -> None:
        # Loaders re-mark derived frames (e.g. positions_monthly rebuilt from
        # already-marked positions). First mark wins: a second pass must not
        # clobber the true statement values with marked ones.
        positions = _positions([
            {"statement_date": "2026-04-30", "symbol": "AAA",
             "quantity": 10.0, "price": 100.0, "market_value": 1000.0,
             "cost_basis": 900.0, "unrealized_gl": 100.0},
        ])
        once = mark_to_market(positions,
                              _prices([{"symbol": "AAA", "close": 110.0,
                                        "status": "ok"}]))
        twice = mark_to_market(once,
                               _prices([{"symbol": "AAA", "close": 120.0,
                                         "status": "ok"}]))
        self.assertAlmostEqual(twice.at[0, "market_value"], 1200.0)
        self.assertAlmostEqual(twice.at[0, "market_value_stmt"], 1000.0)

    def test_noop_paths_add_no_column(self) -> None:
        # The early returns (empty prices, no usable status rows) hand back
        # the input frame untouched — no stash column either.
        positions = _positions([
            {"statement_date": "2026-04-30", "symbol": "AAA",
             "quantity": 10.0, "price": 100.0, "market_value": 1000.0,
             "cost_basis": 900.0, "unrealized_gl": 100.0},
        ])
        out = mark_to_market(positions, _prices([]))
        self.assertNotIn("market_value_stmt", out.columns)
        out = mark_to_market(positions,
                             _prices([{"symbol": "AAA", "close": 999.0,
                                       "status": "stale"}]))
        self.assertNotIn("market_value_stmt", out.columns)


if __name__ == "__main__":
    unittest.main()

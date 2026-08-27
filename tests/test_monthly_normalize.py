"""Unit tests for parsers/monthly_normalize.py.

`monthly_normalize` collapses a positions frame to one snapshot per
(account, calendar month) and forward-fills gaps so an account that is
missing a month does not silently vanish from the NAV / Holdings views.

Two fill kinds:
  - INTERNAL gap   — a month between an account's first and last real
                     statement (e.g. a skipped Fidelity month). Pre-existing.
  - TRAILING gap   — a month AFTER an account's last real statement but at or
                     before the portfolio's global latest month, i.e. the
                     account lags the newest broker statement. NEW: without
                     this a small Fidelity account whose only statements are
                     April, while the rest reached May, drops out of the May
                     snapshot entirely.

Filled rows are tagged `_filled=True` and carry `_as_of_date` = the original
statement_date of the last real snapshot they were carried from, so the
Holdings tab can badge them "as of <month>".
"""
import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from monthly_normalize import (  # noqa: E402
    monthly_normalize,
    month_canonical_dates,
    slice_as_of_month,
)


def _positions(rows: list[dict]) -> pd.DataFrame:
    """Build a positions frame from terse row dicts.

    Each row needs at least statement_date + account_id; holdings columns
    (symbol/quantity/market_value) are carried verbatim by the fill logic.
    """
    df = pd.DataFrame(rows)
    df["statement_date"] = pd.to_datetime(df["statement_date"])
    return df


def _row(date: str, acct: str, symbol: str = "AAA",
         qty: float = 10.0, mv: float = 1000.0, broker: str = "fidelity") -> dict:
    return {"statement_date": date, "account_id": acct, "broker": broker,
            "symbol": symbol, "quantity": qty, "market_value": mv}


class TestMonthlyNormalizeExisting(unittest.TestCase):
    """Characterization tests pinning the pre-existing behavior."""

    def test_keeps_latest_statement_date_within_a_month(self) -> None:
        # Same account + month, two statement_dates → keep only the later one.
        pos = _positions([
            _row("2026-03-30", "A", mv=900.0),
            _row("2026-03-31", "A", mv=1000.0),
        ])
        out = monthly_normalize(pos)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["market_value"], 1000.0)
        self.assertEqual(out.iloc[0]["statement_date"], pd.Timestamp("2026-03-31"))

    def test_internal_gap_is_filled(self) -> None:
        # A has Jan + Mar (Feb missing) and B anchors the global range Jan..Mar.
        pos = _positions([
            _row("2026-01-31", "A", mv=100.0),
            _row("2026-03-31", "A", mv=300.0),
            _row("2026-01-31", "B"), _row("2026-02-28", "B"), _row("2026-03-31", "B"),
        ])
        out = monthly_normalize(pos)
        a_feb = out[(out["account_id"] == "A") & (out["month"] == pd.Period("2026-02", "M"))]
        self.assertEqual(len(a_feb), 1, "internal Feb gap for A should be filled")
        self.assertTrue(bool(a_feb.iloc[0]["_filled"]))
        # Carried verbatim from Jan (the prior real month).
        self.assertEqual(a_feb.iloc[0]["market_value"], 100.0)

    def test_real_rows_are_not_flagged_filled(self) -> None:
        pos = _positions([_row("2026-03-31", "A"), _row("2026-04-30", "A")])
        out = monthly_normalize(pos)
        self.assertFalse(out["_filled"].any())

    def test_empty_input_returns_empty_without_error(self) -> None:
        empty = pd.DataFrame(
            columns=["statement_date", "account_id", "broker", "symbol",
                     "quantity", "market_value"])
        empty["statement_date"] = pd.to_datetime(empty["statement_date"])
        out = monthly_normalize(empty)
        self.assertTrue(out.empty)


class TestTrailingCarryForward(unittest.TestCase):
    """The new behavior: accounts lagging the global latest month are carried
    forward instead of dropping out."""

    def _lagging_book(self) -> pd.DataFrame:
        # A reaches May; B (the laggard) only has Mar + Apr.
        return _positions([
            _row("2026-03-31", "A", mv=100.0),
            _row("2026-04-30", "A", mv=110.0),
            _row("2026-05-31", "A", mv=120.0),
            _row("2026-03-31", "B", symbol="BBB", qty=5.0, mv=500.0),
            _row("2026-04-30", "B", symbol="BBB", qty=5.0, mv=550.0),
        ])

    def test_lagging_account_is_carried_into_latest_month(self) -> None:
        out = monthly_normalize(self._lagging_book())
        b_may = out[(out["account_id"] == "B") & (out["month"] == pd.Period("2026-05", "M"))]
        self.assertEqual(len(b_may), 1,
                         "lagging account B must appear in May, not vanish")
        self.assertTrue(bool(b_may.iloc[0]["_filled"]),
                        "carried-forward row must be flagged _filled")

    def test_carried_row_preserves_last_known_holdings(self) -> None:
        out = monthly_normalize(self._lagging_book())
        b_may = out[(out["account_id"] == "B") & (out["month"] == pd.Period("2026-05", "M"))].iloc[0]
        # Carried verbatim from B's last real statement (April).
        self.assertEqual(b_may["symbol"], "BBB")
        self.assertEqual(b_may["quantity"], 5.0)
        self.assertEqual(b_may["market_value"], 550.0)

    def test_carried_row_records_as_of_date_provenance(self) -> None:
        out = monthly_normalize(self._lagging_book())
        b_may = out[(out["account_id"] == "B") & (out["month"] == pd.Period("2026-05", "M"))].iloc[0]
        # Provenance points at B's last REAL statement (April 30), so the
        # Holdings tab can badge "as of Apr 30".
        self.assertEqual(b_may["_as_of_date"], pd.Timestamp("2026-04-30"))

    def test_real_rows_as_of_date_equals_statement_date(self) -> None:
        out = monthly_normalize(self._lagging_book())
        real = out[~out["_filled"].astype(bool)]
        self.assertTrue((real["_as_of_date"] == real["statement_date"]).all())

    def test_carried_row_statement_date_matches_month_canonical_date(self) -> None:
        # A's May statement is dated the 29th (e.g. a last-business-day broker),
        # NOT the calendar month end. The carried row must adopt that SAME date
        # so it lines up with the Holdings "as of" selector (which only offers
        # real statement dates) and reaches mark-to-market (which marks
        # statement_date == max).
        pos = _positions([
            _row("2026-04-30", "A", mv=110.0),
            _row("2026-05-29", "A", mv=120.0),
            _row("2026-04-30", "B", symbol="BBB", mv=550.0),
        ])
        out = monthly_normalize(pos)
        b_may = out[(out["account_id"] == "B") & (out["month"] == pd.Period("2026-05", "M"))].iloc[0]
        self.assertEqual(b_may["statement_date"], pd.Timestamp("2026-05-29"))

    def test_multi_month_lag_carries_through_each_trailing_month(self) -> None:
        # B last reported in March; A reached May → B carried into Apr AND May,
        # both flagged, both pointing back at the March source.
        pos = _positions([
            _row("2026-03-31", "A", mv=100.0),
            _row("2026-04-30", "A", mv=110.0),
            _row("2026-05-31", "A", mv=120.0),
            _row("2026-03-31", "B", symbol="BBB", qty=5.0, mv=500.0),
        ])
        out = monthly_normalize(pos)
        b_fills = out[(out["account_id"] == "B") & out["_filled"].astype(bool)]
        self.assertEqual(set(b_fills["month"]),
                         {pd.Period("2026-04", "M"), pd.Period("2026-05", "M")})
        self.assertTrue((b_fills["_as_of_date"] == pd.Timestamp("2026-03-31")).all())
        self.assertTrue((b_fills["market_value"] == 500.0).all())

    def test_no_trailing_fill_when_accounts_are_aligned(self) -> None:
        # Everyone reaches May → no fills at all.
        pos = _positions([
            _row("2026-04-30", "A"), _row("2026-05-31", "A"),
            _row("2026-04-30", "B"), _row("2026-05-31", "B"),
        ])
        out = monthly_normalize(pos)
        self.assertFalse(out["_filled"].any())


class TestMonthCanonicalDates(unittest.TestCase):
    """Picker pool: one canonical (month-max) statement_date per month, desc."""

    def test_dual_date_month_collapses_to_month_max(self) -> None:
        pos = _positions([
            _row("2026-03-30", "B", broker="jpm"),       # JPM last-biz-day
            _row("2026-03-31", "A", broker="fidelity"),  # Fidelity month-end
            _row("2026-04-29", "B", broker="jpm"),
            _row("2026-04-30", "A", broker="fidelity"),
        ])
        out = month_canonical_dates(pos)
        self.assertEqual(
            out, [pd.Timestamp("2026-04-30"), pd.Timestamp("2026-03-31")],
            "one date per month (the month's max), newest first",
        )

    def test_single_date_months_unchanged(self) -> None:
        pos = _positions([_row("2026-01-31", "A"), _row("2026-02-28", "A")])
        self.assertEqual(
            month_canonical_dates(pos),
            [pd.Timestamp("2026-02-28"), pd.Timestamp("2026-01-31")],
        )

    def test_empty_frame_returns_empty_list(self) -> None:
        empty = pd.DataFrame({"statement_date": pd.to_datetime([])})
        self.assertEqual(month_canonical_dates(empty), [])

    def test_all_nat_frame_returns_empty(self) -> None:
        df = pd.DataFrame({"statement_date": pd.to_datetime([None, None])})
        self.assertEqual(month_canonical_dates(df), [])


class TestSliceAsOfMonth(unittest.TestCase):
    """Calendar-month slice — both brokers in a dual-date month."""

    def test_dual_date_slice_keeps_both_brokers(self) -> None:
        pos = _positions([
            _row("2026-03-30", "B", symbol="BBB", mv=2000.0, broker="jpm"),
            _row("2026-03-31", "A", symbol="AAA", mv=1000.0, broker="fidelity"),
        ])
        # Any in-month date (either statement date, or a mid-month date)
        # returns the FULL month.
        for d in ("2026-03-31", "2026-03-30", "2026-03-15"):
            out = slice_as_of_month(pos, pd.Timestamp(d))
            self.assertEqual(set(out["broker"]), {"jpm", "fidelity"},
                             f"slice for {d} dropped a broker")
            self.assertEqual(out["market_value"].sum(), 3000.0)

    def test_excludes_other_months(self) -> None:
        pos = _positions([
            _row("2026-03-31", "A", mv=1000.0),
            _row("2026-04-30", "A", mv=2000.0),
        ])
        out = slice_as_of_month(pos, pd.Timestamp("2026-03-31"))
        self.assertEqual(out["market_value"].sum(), 1000.0)

    def test_idempotent_on_monthly_normalized_frame(self) -> None:
        # positions_monthly is already one snapshot per (account, month);
        # slicing it by month returns exactly that month's accounts.
        pos = _positions([
            _row("2026-03-30", "B", mv=2000.0, broker="jpm"),
            _row("2026-03-31", "A", mv=1000.0, broker="fidelity"),
            _row("2026-04-30", "A", mv=1100.0, broker="fidelity"),
            _row("2026-04-29", "B", mv=2100.0, broker="jpm"),
        ])
        pm = monthly_normalize(pos)
        out = slice_as_of_month(pm, pd.Timestamp("2026-03-31"))
        self.assertEqual(set(out["account_id"]), {"A", "B"})
        self.assertEqual(out["market_value"].sum(), 3000.0)

    def test_empty_frame_returns_empty(self) -> None:
        empty = pd.DataFrame({"statement_date": pd.to_datetime([])})
        self.assertTrue(slice_as_of_month(empty, pd.Timestamp("2026-03-31")).empty)

    def test_none_or_nat_as_of_returns_empty(self) -> None:
        pos = _positions([_row("2026-03-31", "A")])
        self.assertTrue(slice_as_of_month(pos, None).empty)
        self.assertTrue(slice_as_of_month(pos, pd.NaT).empty)

    def test_all_nat_frame_returns_empty(self) -> None:
        df = pd.DataFrame({"statement_date": pd.to_datetime([None, None])})
        self.assertTrue(slice_as_of_month(df, pd.Timestamp("2026-03-31")).empty)


if __name__ == "__main__":
    unittest.main()

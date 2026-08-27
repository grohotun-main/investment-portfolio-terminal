"""Tests for parsers/iv_at_buy.py.

`book_iv_at_buy_rank` is the pure math for the caption's
"bought avg at rank M" segment. The UI constructs the leg list
upstream (joining reconstructed lots to current MV).
"""
import math
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from iv_at_buy import (  # noqa: E402
    BookAtBuy,
    book_iv_at_buy_rank,
    leg_iv_at_buy,
)


def _hist(values_by_underlying: dict[str, list[float]],
          as_of: date = date(2026, 5, 26)) -> pd.DataFrame:
    """Synthesize history: each underlying gets a list of daily IVs
    ending on as_of (newest last)."""
    rows = []
    for u, vals in values_by_underlying.items():
        for i, v in enumerate(reversed(vals)):
            rows.append({
                "date": pd.Timestamp(as_of - timedelta(days=i)).isoformat(),
                "underlying": u, "atm_iv": v,
            })
    return pd.DataFrame(rows)


class TestLegIvAtBuy(unittest.TestCase):
    """Per-leg rank: where was IV in its 52w window on the leg's
    open_date? The 52w window is contemporaneous (ends on open_date)."""

    def test_open_date_in_history(self):
        # 10-day history (oldest-first), open_date = 5 days before as_of.
        # values[0..9] map to days -9..-0 from as_of; values[4] = day-5.
        h = _hist({"SPY": [0.10, 0.12, 0.14, 0.16, 0.20, 0.18, 0.15, 0.13, 0.11, 0.10]})
        r = leg_iv_at_buy(h, "SPY", date(2026, 5, 21))  # day-5
        # IV at buy = 0.20 (peak of the window so far);
        # window day-9..-5 = [0.10, 0.12, 0.14, 0.16, 0.20] -> min 0.10, max 0.20
        # rank = 100 * (0.20 - 0.10) / (0.20 - 0.10) = 100
        self.assertAlmostEqual(r.iv_at_buy, 0.20)
        self.assertAlmostEqual(r.rank_at_buy, 100.0)

    def test_no_row_for_underlying_returns_none(self):
        h = _hist({"SPY": [0.10, 0.20]})
        self.assertIsNone(leg_iv_at_buy(h, "NVDA", date(2026, 5, 26)))

    def test_open_date_before_history_starts_returns_none(self):
        h = _hist({"SPY": [0.10, 0.20]})
        self.assertIsNone(leg_iv_at_buy(h, "SPY", date(2023, 1, 1)))

    def test_uses_prior_row_if_open_date_not_a_trading_day(self):
        # Skip a date — open_date falls on a "weekend".
        h = _hist({"SPY": [0.10, 0.20, 0.15, 0.30]})  # days as_of-3, -2, -1, 0
        # open_date 2026-05-25 (skipped — not in history). Should use
        # the last <= open_date row, which is 2026-05-24 -> 0.15.
        r = leg_iv_at_buy(h, "SPY", date(2026, 5, 25))
        # But 2026-05-25 is in history (as_of-1=2026-05-25 was row 0.15)
        # Adjust: pick a true gap. open_date = as_of-2 = 2026-05-24 is row 0.20.
        r = leg_iv_at_buy(h, "SPY", date(2026, 5, 24))
        self.assertAlmostEqual(r.iv_at_buy, 0.20)


class TestBookIvAtBuyRank(unittest.TestCase):
    def test_single_leg(self):
        h = _hist({"SPY": [0.10, 0.20, 0.30, 0.15]})
        legs = [{"underlying": "SPY",
                 "open_date": pd.Timestamp(date(2026, 5, 25)),  # as_of-1
                 "market_value": 10_000.0}]
        b = book_iv_at_buy_rank(h, legs, as_of=date(2026, 5, 26))
        # On as_of-1: history is [0.10, 0.20, 0.30]; iv_at_buy = 0.30
        # rank = 100 * (0.30 - 0.10) / (0.30 - 0.10) = 100
        self.assertAlmostEqual(b.rank, 100.0)
        self.assertEqual(b.covered_legs, 1)

    def test_two_legs_weighted(self):
        # Construct so per-leg ranks differ — confirms the weighted average
        # actually combines them rather than degenerating.
        # SPY 4-day history with open at day-1: window day-3..-1 = [0.10, 0.20, 0.40]
        #     iv_at_buy = 0.40 -> rank = 100
        # NVDA 4-day history with open at day-1: window = [0.50, 0.30, 0.20]
        #     iv_at_buy = 0.20 -> rank = 0 (it's the min of its window)
        h = _hist({"SPY": [0.10, 0.20, 0.40, 0.05],
                   "NVDA": [0.50, 0.30, 0.20, 0.10]})
        legs = [
            {"underlying": "SPY", "open_date": pd.Timestamp(date(2026, 5, 25)),
             "market_value": 7_500.0},
            {"underlying": "NVDA", "open_date": pd.Timestamp(date(2026, 5, 25)),
             "market_value": 2_500.0},
        ]
        # weighted: (0.75 * 100) + (0.25 * 0) = 75
        b = book_iv_at_buy_rank(h, legs, as_of=date(2026, 5, 26))
        self.assertAlmostEqual(b.rank, 75.0)
        self.assertEqual(b.covered_legs, 2)

    def test_legs_without_history_are_skipped(self):
        # SPY history days -2..-0 = [0.10, 0.20, 0.30]. Open at day-1
        # uses window [0.10, 0.20] → rank 100 (max of own window).
        h = _hist({"SPY": [0.10, 0.20, 0.30]})
        legs = [
            {"underlying": "SPY", "open_date": pd.Timestamp(date(2026, 5, 25)),
             "market_value": 5_000.0},
            {"underlying": "MSFT", "open_date": pd.Timestamp(date(2026, 5, 25)),
             "market_value": 5_000.0},
        ]
        b = book_iv_at_buy_rank(h, legs, as_of=date(2026, 5, 26))
        # MSFT skipped → book rank = SPY rank only = 100
        self.assertAlmostEqual(b.rank, 100.0)
        self.assertEqual(b.covered_legs, 1)
        self.assertEqual(b.skipped_legs, 1)

    def test_no_legs_returns_none(self):
        h = _hist({"SPY": [0.10, 0.20]})
        self.assertIsNone(book_iv_at_buy_rank(h, [], as_of=date(2026, 5, 26)))


if __name__ == "__main__":
    unittest.main()

"""Tests for parsers/iv_rank.py — the IV Rank math feeding the caption
on the Options Hedging tab.

Two pure surfaces:
  * `iv_rank` — single-underlying rank against a 52w window.
  * `book_iv_rank` — MV-weighted aggregate across the sleeve.
"""
import math
import sys
import unittest
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from iv_rank import (  # noqa: E402
    BookPercentile,
    BookRank,
    UnderlyingRank,
    book_iv_percentile,
    book_iv_percentile_series,
    book_iv_rank,
    format_iv_rank_caption,
    format_weighted_iv_caption,
    iv_percentile,
    iv_rank,
)
from iv_at_buy import BookAtBuy  # noqa: E402


def _hist(values: list[float], underlying: str = "SPY") -> pd.DataFrame:
    """Build a synthetic ATM IV history with one row per recent date."""
    d0 = date(2026, 5, 26)
    rows = [
        {"date": pd.Timestamp(d0 - timedelta(days=i)).isoformat(),
         "underlying": underlying, "atm_iv": v}
        for i, v in enumerate(reversed(values))
    ]
    return pd.DataFrame(rows)


class TestIvRank(unittest.TestCase):
    def test_midrange_rank(self):
        # history [0.10, 0.20, 0.30, 0.15] → today=0.15, min=0.10, max=0.30
        # rank = 100 * (0.15 - 0.10) / (0.30 - 0.10) = 25
        h = _hist([0.10, 0.20, 0.30, 0.15])
        r = iv_rank(h, "SPY", as_of=date(2026, 5, 26))
        self.assertAlmostEqual(r.rank, 25.0)
        self.assertAlmostEqual(r.iv_min, 0.10)
        self.assertAlmostEqual(r.iv_max, 0.30)

    def test_today_at_min_returns_zero(self):
        h = _hist([0.20, 0.30, 0.10])  # today=0.10, min
        r = iv_rank(h, "SPY", as_of=date(2026, 5, 26))
        self.assertAlmostEqual(r.rank, 0.0)

    def test_today_at_max_returns_100(self):
        h = _hist([0.10, 0.20, 0.30])  # today=0.30, max
        r = iv_rank(h, "SPY", as_of=date(2026, 5, 26))
        self.assertAlmostEqual(r.rank, 100.0)

    def test_all_equal_history_returns_nan_rank(self):
        h = _hist([0.20, 0.20, 0.20])
        r = iv_rank(h, "SPY", as_of=date(2026, 5, 26))
        self.assertTrue(math.isnan(r.rank))

    def test_empty_history_returns_none(self):
        h = pd.DataFrame(columns=["date", "underlying", "atm_iv"])
        r = iv_rank(h, "SPY", as_of=date(2026, 5, 26))
        self.assertIsNone(r)

    def test_missing_underlying_returns_none(self):
        h = _hist([0.20, 0.30], underlying="NVDA")
        r = iv_rank(h, "SPY", as_of=date(2026, 5, 26))
        self.assertIsNone(r)


class TestBookIvRank(unittest.TestCase):
    """Book rank: MV-weighted across legs by their underlying's rank."""

    def test_single_underlying_book_equals_underlying_rank(self):
        h = _hist([0.10, 0.20, 0.30, 0.15], underlying="SPY")
        positions_mv = {"SPY": 100_000.0}
        b = book_iv_rank(h, positions_mv, as_of=date(2026, 5, 26))
        single = iv_rank(h, "SPY", as_of=date(2026, 5, 26))
        self.assertAlmostEqual(b.rank, single.rank, places=4)

    def test_two_underlyings_weighted_avg(self):
        # SPY history with last=0.20 in range [0.10, 0.30] -> rank=50
        # NVDA history with last=0.50 in range [0.30, 0.70] -> rank=50
        # Different MVs but same rank — book rank = 50.
        spy = _hist([0.10, 0.30, 0.20], underlying="SPY")
        nvda = _hist([0.30, 0.70, 0.50], underlying="NVDA")
        h = pd.concat([spy, nvda], ignore_index=True)
        b = book_iv_rank(h, {"SPY": 60_000.0, "NVDA": 40_000.0},
                         as_of=date(2026, 5, 26))
        self.assertAlmostEqual(b.rank, 50.0)

    def test_skips_underlyings_with_no_history(self):
        # NVDA in book, SPY only in history. Book rank should equal
        # SPY's rank weighted only by SPY's MV — NVDA dropped.
        spy = _hist([0.10, 0.30, 0.20], underlying="SPY")
        b = book_iv_rank(spy, {"SPY": 60_000.0, "NVDA": 40_000.0},
                         as_of=date(2026, 5, 26))
        self.assertAlmostEqual(b.rank, 50.0)
        self.assertEqual(b.covered_underlyings, ["SPY"])
        self.assertEqual(b.skipped_underlyings, ["NVDA"])

    def test_empty_positions_returns_none(self):
        h = _hist([0.10, 0.30, 0.20])
        b = book_iv_rank(h, {}, as_of=date(2026, 5, 26))
        self.assertIsNone(b)


class TestFormatIvRankCaption(unittest.TestCase):
    """Pure string formatter used under the Weighted IV tile."""

    def test_both_available(self):
        book = BookRank(
            rank=35.0, iv_today_weighted=0.155,
            iv_min_weighted=0.10, iv_max_weighted=0.30,
            as_of=date(2026, 5, 26),
            covered_underlyings=["SPY", "NVDA"], skipped_underlyings=[],
        )
        at_buy = BookAtBuy(rank=60.0, covered_legs=2, skipped_legs=0)
        s = format_iv_rank_caption(book, at_buy)
        self.assertIn("rank **35**", s)
        self.assertIn("10.0%", s)
        self.assertIn("30.0%", s)
        self.assertIn("bought avg at rank **60**", s)

    def test_no_at_buy(self):
        book = BookRank(
            rank=35.0, iv_today_weighted=0.155,
            iv_min_weighted=0.10, iv_max_weighted=0.30,
            as_of=date(2026, 5, 26),
            covered_underlyings=["SPY"], skipped_underlyings=[],
        )
        s = format_iv_rank_caption(book, None)
        self.assertIn("rank **35**", s)
        self.assertNotIn("bought avg", s)

    def test_no_history_at_all(self):
        s = format_iv_rank_caption(None, None)
        self.assertIn("not available", s.lower())

    def test_skipped_underlyings_surfaced(self):
        book = BookRank(
            rank=35.0, iv_today_weighted=0.155,
            iv_min_weighted=0.10, iv_max_weighted=0.30,
            as_of=date(2026, 5, 26),
            covered_underlyings=["SPY"], skipped_underlyings=["NVDA"],
        )
        s = format_iv_rank_caption(book, None)
        self.assertIn("NVDA", s)


def _pct_hist(values, underlying="SPY", start="2025-01-02"):
    dates = pd.bdate_range(start, periods=len(values))
    return pd.DataFrame({
        "date": dates,
        "underlying": [underlying] * len(values),
        "atm_iv": values,
    })


class TestIvPercentile(unittest.TestCase):
    """Empirical-percentile (rank-order) helper for SPY ATM IV history."""

    def test_iv_percentile_basic_ascending(self):
        # 100 ascending obs; today is the last (highest) -> ~p100.
        hist = _pct_hist([v / 100 for v in range(1, 101)])
        today = hist["date"].iloc[-1].date()
        pct = iv_percentile(hist, "SPY", as_of=today)
        self.assertAlmostEqual(pct, 100.0, delta=0.5)

    def test_iv_percentile_basic_median(self):
        # 101 obs; today's IV equals the median -> p~50.
        vals = [v / 100 for v in range(1, 102)]
        hist = _pct_hist(vals)
        # Force today's IV to the median value (51st when 1..101).
        hist.loc[hist.index[-1], "atm_iv"] = 0.51
        today = hist["date"].iloc[-1].date()
        pct = iv_percentile(hist, "SPY", as_of=today)
        self.assertAlmostEqual(pct, 50.5, delta=0.01)

    def test_iv_percentile_ties_average_rank(self):
        # 5 values; today (0.5) ties with one earlier obs (also 0.5).
        # Under method="average", today's rank = (4+5)/2 = 4.5
        #   → pct = 100*(4.5-1)/(5-1) = 87.5
        # Under method="max", today's rank = 5 → pct = 100.0
        # Under method="min", today's rank = 4 → pct = 75.0
        # The tight tolerance below rules out the wrong methods.
        hist = _pct_hist([0.1, 0.2, 0.5, 0.3, 0.5])
        today = hist["date"].iloc[-1].date()
        pct = iv_percentile(hist, "SPY", as_of=today)
        self.assertAlmostEqual(pct, 87.5, delta=0.1)

    def test_iv_percentile_short_window_partial(self):
        # 30 obs (< 252) -> still returns a value, computed over what's available.
        hist = _pct_hist([v / 100 for v in range(1, 31)])
        today = hist["date"].iloc[-1].date()
        pct = iv_percentile(hist, "SPY", as_of=today)
        self.assertAlmostEqual(pct, 100.0, delta=2.0)

    def test_iv_percentile_window_is_trailing_rows_not_calendar(self):
        # The window must count trailing SESSIONS (rows), not calendar days.
        # Sparse dates with a gap make the two semantics diverge: with
        # window=5, tail(5 rows) reaches back past a spike that a 5-calendar-
        # day window would exclude.
        #   dates:  06-01 06-03 06-05 06-20 06-23 06-25(today)
        #   iv:     0.10  0.10  0.90  0.20  0.25  0.30
        # tail(5 rows) = 06-03..06-25 = {0.10,0.90,0.20,0.25,0.30};
        #   today=0.30 ranks 4th of 5 -> 100*(4-1)/(5-1) = 75.
        # 5-calendar-day window = 06-20..06-25 = {0.20,0.25,0.30};
        #   today=0.30 = max -> 100. Asserting 75 pins the row-based window.
        hist = pd.DataFrame({
            "date": pd.to_datetime([
                "2026-06-01", "2026-06-03", "2026-06-05",
                "2026-06-20", "2026-06-23", "2026-06-25",
            ]),
            "underlying": ["SPY"] * 6,
            "atm_iv": [0.10, 0.10, 0.90, 0.20, 0.25, 0.30],
        })
        pct = iv_percentile(hist, "SPY", as_of=date(2026, 6, 25),
                            window_days=5)
        self.assertAlmostEqual(pct, 75.0, delta=1e-6)

    def test_iv_percentile_ignores_nan_rows_when_taking_tail(self):
        # CM derive emits NaN rows for dead days; they must be dropped
        # BEFORE the tail so the window holds 252 real observations, not
        # 252 rows padded with NaN. Here: 2 NaN at the end + 3 real obs;
        # today (last real, 0.30) is the max of the 3 real -> 100.
        hist = _pct_hist([0.10, 0.20, 0.30, float("nan"), float("nan")])
        today = hist["date"].iloc[-1].date()
        pct = iv_percentile(hist, "SPY", as_of=today, window_days=252)
        self.assertAlmostEqual(pct, 100.0, delta=1e-6)

    def test_iv_percentile_empty_underlying(self):
        hist = _pct_hist([0.2, 0.3], underlying="NVDA")
        today = hist["date"].iloc[-1].date()
        self.assertTrue(math.isnan(iv_percentile(hist, "SPY", as_of=today)))

    def test_iv_percentile_empty_history(self):
        pct = iv_percentile(
            pd.DataFrame(), "SPY",
            as_of=pd.Timestamp("2025-06-01").date(),
        )
        self.assertTrue(math.isnan(pct))


class TestBookIVPercentile(unittest.TestCase):
    """MV-weighted empirical percentile across the sleeve — the gauge that
    replaces the degenerate min/max book rank in the live caption."""

    def _two_name_history(self):
        # SPY ascends 10..50, NVDA ascends 40..80; today (1/5) is the max of
        # each series → each underlying sits at percentile 100.
        dates = pd.to_datetime([
            "2026-01-01", "2026-01-02", "2026-01-03",
            "2026-01-04", "2026-01-05",
        ])
        return pd.DataFrame({
            "date": list(dates) * 2,
            "underlying": ["SPY"] * 5 + ["NVDA"] * 5,
            "atm_iv": [0.10, 0.20, 0.30, 0.40, 0.50,
                       0.40, 0.50, 0.60, 0.70, 0.80],
        })

    def test_book_percentile_today_at_top_is_100(self):
        book = book_iv_percentile(
            self._two_name_history(),
            {"SPY": 1000.0, "NVDA": 3000.0},
            as_of=date(2026, 1, 5),
        )
        self.assertIsNotNone(book)
        self.assertAlmostEqual(book.percentile, 100.0)
        self.assertEqual(set(book.covered_underlyings), {"SPY", "NVDA"})
        self.assertEqual(book.skipped_underlyings, [])

    def test_book_percentile_mv_weights_blend(self):
        # SPY today at top (pct 100), NVDA today at bottom (pct 0).
        hist = self._two_name_history().copy()
        hist.loc[hist["underlying"] == "NVDA", "atm_iv"] = [
            0.80, 0.70, 0.60, 0.50, 0.40,  # descending → today is min → 0
        ]
        book = book_iv_percentile(
            hist, {"SPY": 1000.0, "NVDA": 3000.0}, as_of=date(2026, 1, 5),
        )
        # weighted = (1000*100 + 3000*0) / 4000 = 25
        self.assertAlmostEqual(book.percentile, 25.0)

    def test_book_percentile_skips_underlying_missing_from_history(self):
        book = book_iv_percentile(
            self._two_name_history(),
            {"SPY": 1000.0, "QQQ": 5000.0},  # QQQ not in history
            as_of=date(2026, 1, 5),
        )
        self.assertIn("QQQ", book.skipped_underlyings)
        self.assertEqual(book.covered_underlyings, ["SPY"])
        # QQQ's weight is dropped; SPY alone → 100
        self.assertAlmostEqual(book.percentile, 100.0)

    def test_book_percentile_window_excludes_old_data(self):
        # Tail-row semantics: with >= window rows of recent data, anything
        # older than the most-recent `window_days` SESSIONS is excluded.
        # 30 recent rows ascending 0.20..0.49 + one ancient spike (0.99).
        # tail(30) keeps the recent block; the spike (31st from the end)
        # drops out, so today (max of the block) -> 100. If the spike
        # leaked in, today would rank 30/31 -> ~96.7.
        recent_dates = pd.bdate_range("2026-01-01", periods=30)
        recent = pd.DataFrame({
            "date": recent_dates,
            "underlying": ["SPY"] * 30,
            "atm_iv": [0.20 + 0.01 * i for i in range(30)],
        })
        old = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-01"]),
            "underlying": ["SPY"], "atm_iv": [0.99],
        })
        hist = pd.concat([old, recent], ignore_index=True)
        today = recent_dates[-1].date()
        book = book_iv_percentile(
            hist, {"SPY": 1000.0}, as_of=today, window_days=30,
        )
        self.assertAlmostEqual(book.percentile, 100.0)

    def test_book_percentile_empty_positions_returns_none(self):
        self.assertIsNone(
            book_iv_percentile(
                self._two_name_history(), {}, as_of=date(2026, 1, 5),
            )
        )

    def test_book_percentile_all_skipped_returns_none(self):
        self.assertIsNone(
            book_iv_percentile(
                self._two_name_history(),
                {"QQQ": 1000.0}, as_of=date(2026, 1, 5),
            )
        )


class TestBookIVPercentileQuality(unittest.TestCase):
    """The CM history carries a per-row `quality` flag (interp / approx /
    none). The gauge surfaces underlyings whose CURRENT reading is a
    one-sided `approx` (not a true 90d), so the caption can flag it."""

    def _q_hist(self, rows):
        # rows: list of (date, underlying, atm_iv, quality)
        return pd.DataFrame(
            [{"date": pd.Timestamp(d), "underlying": u,
              "atm_iv": v, "quality": q} for d, u, v, q in rows]
        )

    def test_approx_today_is_flagged(self):
        hist = self._q_hist([
            ("2026-01-01", "SPY", 0.10, "interp"),
            ("2026-01-02", "SPY", 0.20, "interp"),
            ("2026-01-03", "SPY", 0.30, "approx"),  # today: one-sided
        ])
        book = book_iv_percentile(hist, {"SPY": 1000.0},
                                  as_of=date(2026, 1, 3))
        self.assertEqual(book.approx_underlyings, ["SPY"])

    def test_interp_today_not_flagged_even_if_past_was_approx(self):
        # Only TODAY's reading quality matters for the gauge point.
        hist = self._q_hist([
            ("2026-01-01", "SPY", 0.10, "approx"),
            ("2026-01-02", "SPY", 0.20, "interp"),
            ("2026-01-03", "SPY", 0.30, "interp"),  # today: clean
        ])
        book = book_iv_percentile(hist, {"SPY": 1000.0},
                                  as_of=date(2026, 1, 3))
        self.assertEqual(book.approx_underlyings, [])

    def test_missing_quality_column_means_no_flags(self):
        # Back-compat: pre-CM history (no quality column) → never flagged.
        hist = pd.DataFrame({
            "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "underlying": ["SPY", "SPY"], "atm_iv": [0.20, 0.30],
        })
        book = book_iv_percentile(hist, {"SPY": 1000.0},
                                  as_of=date(2026, 1, 2))
        self.assertEqual(book.approx_underlyings, [])


class TestFormatWeightedIVCaption(unittest.TestCase):
    def _book(self, pct, covered, skipped, approx=None):
        return BookPercentile(
            percentile=pct, covered_underlyings=list(covered),
            skipped_underlyings=list(skipped), as_of=date(2026, 1, 5),
            approx_underlyings=list(approx or []),
        )

    def test_caption_includes_percentile_and_window(self):
        cap = format_weighted_iv_caption(
            0.194, self._book(42.0, ["SPY", "NVDA"], []), window_days=252,
        )
        self.assertIn("42", cap)
        self.assertIn("252", cap)

    def test_caption_states_constant_maturity(self):
        # The whole point of this work: the caption must say what IV it is.
        cap = format_weighted_iv_caption(
            0.194, self._book(42.0, ["SPY"], []), window_days=252,
        )
        self.assertIn("90-day constant-maturity", cap)

    def test_caption_notes_skipped_underlyings(self):
        cap = format_weighted_iv_caption(
            0.194, self._book(42.0, ["SPY"], ["NVDA"]), window_days=252,
        )
        self.assertIn("NVDA", cap)

    def test_caption_flags_approx_underlyings(self):
        cap = format_weighted_iv_caption(
            0.194, self._book(42.0, ["SPY", "NVDA"], [], approx=["NVDA"]),
            window_days=252,
        )
        self.assertIn("NVDA", cap)
        self.assertIn("approx", cap.lower())

    def test_caption_when_percentile_unavailable(self):
        cap = format_weighted_iv_caption(0.194, None, window_days=252)
        # Graceful — no crash, signals the gauge isn't populated
        self.assertIsInstance(cap, str)
        self.assertTrue(len(cap) > 0)


class TestBookIVPercentileSeries(unittest.TestCase):
    """Rolling trajectory of the MV-weighted book percentile — the data
    behind the sparkline. Each row is the book percentile AS OF that day,
    ranked against its own trailing `window_days` window."""

    def test_returns_date_percentile_frame(self):
        # window 5, span 3, enough history to fill both.
        hist = _pct_hist([v / 100 for v in range(1, 13)])  # 12 sessions
        as_of = hist["date"].iloc[-1].date()
        out = book_iv_percentile_series(
            hist, {"SPY": 1000.0}, as_of=as_of,
            window_days=5, span_days=3,
        )
        self.assertIsInstance(out, pd.DataFrame)
        self.assertEqual(list(out.columns), ["date", "percentile"])
        self.assertEqual(len(out), 3)

    def test_last_row_equals_headline_gauge(self):
        # THE invariant: the sparkline's final point must equal today's
        # single-number gauge, so the chart is literally "the headline over
        # time". Two-name MV-weighted book, same fixture shape as the gauge.
        spy = _pct_hist([v / 100 for v in range(1, 31)], underlying="SPY")
        nvda = _pct_hist([v / 50 for v in range(1, 31)], underlying="NVDA")
        hist = pd.concat([spy, nvda], ignore_index=True)
        as_of = spy["date"].iloc[-1].date()
        positions = {"SPY": 60_000.0, "NVDA": 40_000.0}

        series = book_iv_percentile_series(
            hist, positions, as_of=as_of, window_days=10, span_days=5,
        )
        headline = book_iv_percentile(
            hist, positions, as_of=as_of, window_days=10,
        )
        self.assertAlmostEqual(
            float(series["percentile"].iloc[-1]), headline.percentile,
            places=6,
        )

    def test_rising_iv_gives_rising_trajectory(self):
        # Monotonically rising IV → today sits ever-higher in its trailing
        # window → percentile trajectory is non-decreasing.
        hist = _pct_hist([v / 100 for v in range(1, 41)])  # strictly rising
        as_of = hist["date"].iloc[-1].date()
        out = book_iv_percentile_series(
            hist, {"SPY": 1000.0}, as_of=as_of,
            window_days=10, span_days=10,
        )
        pcts = out["percentile"].tolist()
        # Each day is at/near the top of its own rising window → all ~100,
        # and never decreasing.
        for a, b in zip(pcts, pcts[1:]):
            self.assertGreaterEqual(b + 1e-9, a)

    def test_short_history_yields_shorter_series_no_crash(self):
        # Only 4 sessions but span 10 requested → return what exists, capped
        # at available days, no padding, no error.
        hist = _pct_hist([0.10, 0.20, 0.15, 0.30])
        as_of = hist["date"].iloc[-1].date()
        out = book_iv_percentile_series(
            hist, {"SPY": 1000.0}, as_of=as_of,
            window_days=5, span_days=10,
        )
        self.assertLessEqual(len(out), 4)
        # last point still equals the headline for this short history
        headline = book_iv_percentile(
            hist, {"SPY": 1000.0}, as_of=as_of, window_days=5,
        )
        self.assertAlmostEqual(
            float(out["percentile"].iloc[-1]), headline.percentile, places=6,
        )

    def test_empty_positions_returns_empty_frame(self):
        hist = _pct_hist([0.10, 0.20, 0.30])
        out = book_iv_percentile_series(
            hist, {}, as_of=hist["date"].iloc[-1].date(),
            window_days=5, span_days=3,
        )
        self.assertTrue(out.empty)
        self.assertEqual(list(out.columns), ["date", "percentile"])

    def test_no_usable_history_returns_empty_frame(self):
        out = book_iv_percentile_series(
            pd.DataFrame(columns=["date", "underlying", "atm_iv"]),
            {"SPY": 1000.0}, as_of=date(2026, 6, 1),
            window_days=5, span_days=3,
        )
        self.assertTrue(out.empty)
        self.assertEqual(list(out.columns), ["date", "percentile"])

    def test_dates_are_ascending(self):
        hist = _pct_hist([v / 100 for v in range(1, 21)])
        as_of = hist["date"].iloc[-1].date()
        out = book_iv_percentile_series(
            hist, {"SPY": 1000.0}, as_of=as_of,
            window_days=5, span_days=8,
        )
        dates = list(out["date"])
        self.assertEqual(dates, sorted(dates))


if __name__ == "__main__":
    unittest.main()

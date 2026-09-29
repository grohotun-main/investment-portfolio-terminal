"""Unit tests for parsers/incremental_fetch.py (fake fetch_fn, temp CSV; no network)."""
import sys
import threading
import unittest
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from incremental_fetch import (  # noqa: E402
    incremental_refresh,
    refresh_csv,
    assert_not_regressive,
    RegressionError,
)

LB = date(2016, 1, 1)      # lookback_start
TODAY = date(2026, 6, 5)


def _df(rows, cols=("symbol", "date", "close")):
    return pd.DataFrame(rows, columns=list(cols))


def _slice_fetch(full_by_ticker, cols=("date", "close")):
    """fetch_fn that returns the [start,end] slice of each ticker's full series."""
    def fetch_fn(ticker, start, end):
        df = full_by_ticker.get(ticker)
        if df is None or df.empty:
            return pd.DataFrame(columns=list(cols))
        d = pd.to_datetime(df["date"]).dt.date
        return df[(d >= start) & (d <= end)].copy()
    return fetch_fn


class TestIncrementalRefresh(unittest.TestCase):
    def test_new_ticker_full_pull(self):
        existing = _df([])  # empty
        full = {"AAA": _df([("AAA", "2026-06-03", 10.0), ("AAA", "2026-06-04", 11.0)])}
        merged, summary = incremental_refresh(
            existing, ["AAA"], _slice_fetch(full), lookback_start=LB, today=TODAY, settle_buffer_days=0)
        self.assertEqual(summary["full"], 1)
        self.assertEqual(len(merged), 2)

    def test_append_no_split(self):
        existing = _df([("AAA", "2026-06-03", 10.0), ("AAA", "2026-06-04", 11.0)])
        # full series adds 06-05; overlap bar 06-04 close unchanged -> append
        full = {"AAA": _df([("AAA", "2026-06-03", 10.0), ("AAA", "2026-06-04", 11.0),
                            ("AAA", "2026-06-05", 12.0)])}
        merged, summary = incremental_refresh(
            existing, ["AAA"], _slice_fetch(full), lookback_start=LB, today=TODAY, settle_buffer_days=0)
        self.assertEqual(summary["append"], 1)
        self.assertEqual(list(merged["date"]), ["2026-06-03", "2026-06-04", "2026-06-05"])
        self.assertEqual(merged.loc[merged["date"] == "2026-06-05", "close"].iloc[0], 12.0)

    def test_current_no_new_bars(self):
        existing = _df([("AAA", "2026-06-04", 11.0)])
        full = {"AAA": _df([("AAA", "2026-06-04", 11.0)])}  # nothing newer than last
        merged, summary = incremental_refresh(
            existing, ["AAA"], _slice_fetch(full), lookback_start=LB, today=TODAY, settle_buffer_days=0)
        self.assertEqual(summary["current"], 1)
        self.assertEqual(len(merged), 1)

    def test_split_detected_triggers_full_repull(self):
        # On disk pre-split (close 100 @ 06-04). After a 2:1 split, Polygon
        # re-adjusts EVERY bar: the overlap bar 06-04 now reads 50 -> mismatch
        # -> re-pull the full (re-adjusted) history, not an append.
        existing = _df([("AAA", "2026-06-03", 200.0), ("AAA", "2026-06-04", 100.0)])
        full = {"AAA": _df([("AAA", "2026-06-03", 100.0), ("AAA", "2026-06-04", 50.0),
                            ("AAA", "2026-06-05", 55.0)])}
        merged, summary = incremental_refresh(
            existing, ["AAA"], _slice_fetch(full), lookback_start=LB, today=TODAY, settle_buffer_days=0)
        self.assertEqual(summary["resplit"], 1)
        # whole series replaced with the re-adjusted values (06-04 == 50, not 100)
        self.assertEqual(merged.loc[merged["date"] == "2026-06-04", "close"].iloc[0], 50.0)
        self.assertEqual(len(merged), 3)

    def test_full_flag_forces_full(self):
        existing = _df([("AAA", "2026-06-04", 11.0)])
        full = {"AAA": _df([("AAA", "2026-06-03", 9.0), ("AAA", "2026-06-04", 11.0)])}
        merged, summary = incremental_refresh(
            existing, ["AAA"], _slice_fetch(full), lookback_start=LB, today=TODAY, full=True)
        self.assertEqual(summary["full"], 1)
        self.assertEqual(len(merged), 2)

    def test_multi_symbol_isolation(self):
        # AAA appends; BBB splits. No cross-contamination.
        existing = _df([("AAA", "2026-06-04", 11.0),
                        ("BBB", "2026-06-03", 200.0), ("BBB", "2026-06-04", 100.0)])
        full = {
            "AAA": _df([("AAA", "2026-06-04", 11.0), ("AAA", "2026-06-05", 12.0)]),
            "BBB": _df([("BBB", "2026-06-03", 100.0), ("BBB", "2026-06-04", 50.0)]),
        }
        merged, summary = incremental_refresh(
            existing, ["AAA", "BBB"], _slice_fetch(full), lookback_start=LB, today=TODAY, settle_buffer_days=0)
        self.assertEqual(summary["append"], 1)
        self.assertEqual(summary["resplit"], 1)
        self.assertEqual(merged.loc[merged["date"] == "2026-06-05", "close"].iloc[0], 12.0)
        self.assertEqual(merged.loc[(merged["symbol"] == "BBB") &
                                    (merged["date"] == "2026-06-04"), "close"].iloc[0], 50.0)

    def test_single_series_group_none_preserves_extra_columns(self):
        # benchmark shape: one series, OHLCV columns, no symbol col.
        cols = ("date", "open", "close")
        existing = _df([("2026-06-04", 9.5, 10.0)], cols=cols)
        full = {"SPY": _df([("2026-06-04", 9.5, 10.0), ("2026-06-05", 10.2, 11.0)], cols=cols)}
        merged, summary = incremental_refresh(
            existing, ["SPY"], _slice_fetch(full, cols=cols), lookback_start=LB,
            today=TODAY, group_col=None, settle_buffer_days=0)
        self.assertEqual(summary["append"], 1)
        self.assertIn("open", merged.columns)
        self.assertNotIn("symbol", merged.columns)
        self.assertEqual(list(merged["date"]), ["2026-06-04", "2026-06-05"])

    def test_empty_fetch_keeps_disk(self):
        existing = _df([("AAA", "2026-06-04", 11.0)])
        merged, summary = incremental_refresh(
            existing, ["AAA"], _slice_fetch({}), lookback_start=LB, today=TODAY, settle_buffer_days=0)
        self.assertEqual(summary["kept"], 1)
        self.assertEqual(len(merged), 1)  # disk untouched

    def test_carries_forward_entity_not_in_universe(self):
        # CCC is on disk but not requested -> kept verbatim, never dropped.
        existing = _df([("AAA", "2026-06-04", 11.0), ("CCC", "2020-01-02", 5.0)])
        full = {"AAA": _df([("AAA", "2026-06-04", 11.0), ("AAA", "2026-06-05", 12.0)])}
        merged, _ = incremental_refresh(
            existing, ["AAA"], _slice_fetch(full), lookback_start=LB, today=TODAY, settle_buffer_days=0)
        self.assertIn("CCC", set(merged["symbol"]))

    def test_rel_tol_small_wiggle_is_not_a_split(self):
        # A 0.005% difference at the overlap is float noise, not a split -> append.
        existing = _df([("AAA", "2026-06-04", 100.0)])
        full = {"AAA": _df([("AAA", "2026-06-04", 100.0005), ("AAA", "2026-06-05", 101.0)])}
        merged, summary = incremental_refresh(
            existing, ["AAA"], _slice_fetch(full), lookback_start=LB, today=TODAY, settle_buffer_days=0)
        self.assertEqual(summary["append"], 1)

    def test_recent_bar_refreshed_not_resplit(self):
        # The last stored bar (06-05 == today) is an intraday close (10.0); on
        # re-pull it has moved to 10.5 — an intraday update, NOT a split. With a
        # settle buffer the split-check anchors on an older settled bar (06-02),
        # so today's bar is refreshed (10.0 -> 10.5), not falsely re-split.
        existing = _df([("AAA", "2026-06-01", 9.0), ("AAA", "2026-06-02", 9.5),
                        ("AAA", "2026-06-03", 9.8), ("AAA", "2026-06-05", 10.0)])
        full = {"AAA": _df([("AAA", "2026-06-01", 9.0), ("AAA", "2026-06-02", 9.5),
                            ("AAA", "2026-06-03", 9.8), ("AAA", "2026-06-05", 10.5)])}
        merged, summary = incremental_refresh(
            existing, ["AAA"], _slice_fetch(full), lookback_start=LB, today=TODAY,
            settle_buffer_days=3)
        self.assertEqual(summary["resplit"], 0)
        self.assertEqual(summary["current"], 1)
        self.assertEqual(merged.loc[merged["date"] == "2026-06-05", "close"].iloc[0], 10.5)
        self.assertEqual(len(merged), 4)

    def test_history_younger_than_buffer_full_pulls(self):
        # All stored bars are within settle_buffer_days of today -> no settled
        # anchor -> full re-pull (no bar is old enough to trust as a comparison).
        existing = _df([("AAA", "2026-06-04", 11.0)])
        full = {"AAA": _df([("AAA", "2026-06-04", 11.0), ("AAA", "2026-06-05", 12.0)])}
        merged, summary = incremental_refresh(
            existing, ["AAA"], _slice_fetch(full), lookback_start=LB, today=TODAY,
            settle_buffer_days=7)
        self.assertEqual(summary["full"], 1)


class TestNonRegressGuard(unittest.TestCase):
    def test_shrink_raises(self):
        existing = _df([("AAA", "2026-06-03", 10.0), ("AAA", "2026-06-04", 11.0)])
        merged = _df([("AAA", "2026-06-04", 11.0)])  # one row fewer for AAA
        with self.assertRaises(RegressionError):
            assert_not_regressive(existing, merged, "symbol")

    def test_drop_entity_raises(self):
        existing = _df([("AAA", "2026-06-04", 11.0), ("BBB", "2026-06-04", 5.0)])
        merged = _df([("AAA", "2026-06-04", 11.0)])
        with self.assertRaises(RegressionError):
            assert_not_regressive(existing, merged, "symbol")

    def test_growth_ok(self):
        existing = _df([("AAA", "2026-06-04", 11.0)])
        merged = _df([("AAA", "2026-06-04", 11.0), ("AAA", "2026-06-05", 12.0)])
        assert_not_regressive(existing, merged, "symbol")  # no raise

    def test_empty_existing_passes(self):
        assert_not_regressive(_df([]), _df([("AAA", "2026-06-04", 11.0)]), "symbol")


class TestRefreshCsv(unittest.TestCase):
    def test_reads_writes_and_appends(self):
        with TemporaryDirectory() as td:
            out = Path(td) / "daily_prices.csv"
            _df([("AAA", "2026-06-04", 11.0)]).to_csv(out, index=False)
            full = {"AAA": _df([("AAA", "2026-06-04", 11.0), ("AAA", "2026-06-05", 12.0)])}
            summary = refresh_csv(
                out, ["AAA"], _slice_fetch(full), lookback_start=LB, today=TODAY,
                empty_columns=["symbol", "date", "close"], group_col="symbol",
                settle_buffer_days=0)
            self.assertEqual(summary["append"], 1)
            back = pd.read_csv(out)
            self.assertEqual(list(back["date"].astype(str)), ["2026-06-04", "2026-06-05"])

    def test_absent_csv_is_full_pull(self):
        with TemporaryDirectory() as td:
            out = Path(td) / "daily_prices.csv"  # does not exist
            full = {"AAA": _df([("AAA", "2026-06-04", 11.0)])}
            summary = refresh_csv(
                out, ["AAA"], _slice_fetch(full), lookback_start=LB, today=TODAY,
                empty_columns=["symbol", "date", "close"], group_col="symbol",
                settle_buffer_days=0)
            self.assertEqual(summary["full"], 1)
            self.assertTrue(out.exists())

    def test_accepts_and_forwards_max_workers(self):
        # refresh_csv must accept max_workers and produce the same result it
        # would sequentially (two entities so the pool path is exercised).
        with TemporaryDirectory() as td:
            out = Path(td) / "daily_prices.csv"
            _df([("AAA", "2026-06-04", 11.0), ("BBB", "2026-06-04", 5.0)]).to_csv(out, index=False)
            full = {
                "AAA": _df([("AAA", "2026-06-04", 11.0), ("AAA", "2026-06-05", 12.0)]),
                "BBB": _df([("BBB", "2026-06-04", 5.0), ("BBB", "2026-06-05", 6.0)]),
            }
            summary = refresh_csv(
                out, ["AAA", "BBB"], _slice_fetch(full), lookback_start=LB, today=TODAY,
                empty_columns=["symbol", "date", "close"], group_col="symbol",
                settle_buffer_days=0, max_workers=4)
            self.assertEqual(summary["append"], 2)
            back = pd.read_csv(out).sort_values(["symbol", "date"]).reset_index(drop=True)
            self.assertEqual(len(back), 4)


class TestConcurrency(unittest.TestCase):
    def test_concurrent_matches_sequential(self):
        # A multi-entity fixture exercising every mode at once:
        #   AAA appends, BBB resplits, CCC kept (empty fetch), DDD new full.
        # The concurrent (max_workers=8) result MUST equal the sequential
        # (max_workers=1) result — concurrency may not change the answer.
        existing = _df([
            ("AAA", "2026-06-01", 9.0), ("AAA", "2026-06-02", 9.5),
            ("BBB", "2026-06-01", 200.0), ("BBB", "2026-06-02", 100.0),
            ("CCC", "2026-06-01", 5.0), ("CCC", "2026-06-02", 5.5),
        ])
        full = {
            "AAA": _df([("AAA", "2026-06-01", 9.0), ("AAA", "2026-06-02", 9.5),
                        ("AAA", "2026-06-05", 10.0)]),
            "BBB": _df([("BBB", "2026-06-01", 100.0), ("BBB", "2026-06-02", 50.0),
                        ("BBB", "2026-06-05", 55.0)]),
            # CCC intentionally absent -> empty fetch -> kept.
            "DDD": _df([("DDD", "2026-06-04", 7.0), ("DDD", "2026-06-05", 7.5)]),
        }
        ents = ["AAA", "BBB", "CCC", "DDD"]
        seq_m, seq_s = incremental_refresh(
            existing, ents, _slice_fetch(full), lookback_start=LB, today=TODAY,
            settle_buffer_days=0, max_workers=1)
        con_m, con_s = incremental_refresh(
            existing, ents, _slice_fetch(full), lookback_start=LB, today=TODAY,
            settle_buffer_days=0, max_workers=8)
        pd.testing.assert_frame_equal(seq_m, con_m)
        self.assertEqual(seq_s, con_s)
        # sanity: the fixture really did exercise four distinct modes
        self.assertEqual(seq_s["append"], 1)
        self.assertEqual(seq_s["resplit"], 1)
        self.assertEqual(seq_s["kept"], 1)
        self.assertEqual(seq_s["full"], 1)

    def test_concurrency_uses_multiple_threads(self):
        # A Barrier(n) only releases once all n fetch_fn calls are in flight at
        # once — so it proves genuine concurrency, and would time out (raise) if
        # the engine ran sequentially.
        n = 4
        barrier = threading.Barrier(n, timeout=5)
        seen = set()
        lock = threading.Lock()

        def fetch_fn(ticker, start, end):
            barrier.wait()  # blocks until all n workers arrive
            with lock:
                seen.add(threading.get_ident())
            return pd.DataFrame([("2026-06-04", 10.0)], columns=["date", "close"])

        merged, summary = incremental_refresh(
            _df([]), ["AAA", "BBB", "CCC", "DDD"], fetch_fn,
            lookback_start=LB, today=TODAY, settle_buffer_days=0, max_workers=4)
        self.assertEqual(len(seen), n)        # n distinct threads ran at once
        self.assertEqual(summary["full"], n)  # all four new-ticker full pulls

    def test_worker_exception_propagates(self):
        # A raising fetch_fn must abort the whole run (no partial write), not be
        # swallowed. (Real fetch_fn is contracted to return empty on error; this
        # guards an *unexpected* raise.)
        def fetch_fn(ticker, start, end):
            if ticker == "BBB":
                raise RuntimeError("boom")
            return pd.DataFrame([("2026-06-04", 10.0)], columns=["date", "close"])

        with self.assertRaises(RuntimeError):
            incremental_refresh(
                _df([]), ["AAA", "BBB", "CCC"], fetch_fn,
                lookback_start=LB, today=TODAY, settle_buffer_days=0, max_workers=4)

    def test_single_entity_same_result_any_workers(self):
        # One entity must bypass the pool and return an identical result for any
        # worker count (the benchmark shape).
        existing = _df([("AAA", "2026-06-04", 11.0)])
        full = {"AAA": _df([("AAA", "2026-06-04", 11.0), ("AAA", "2026-06-05", 12.0)])}
        m1, s1 = incremental_refresh(
            existing, ["AAA"], _slice_fetch(full), lookback_start=LB, today=TODAY,
            settle_buffer_days=0, max_workers=1)
        m8, s8 = incremental_refresh(
            existing, ["AAA"], _slice_fetch(full), lookback_start=LB, today=TODAY,
            settle_buffer_days=0, max_workers=8)
        pd.testing.assert_frame_equal(m1, m8)
        self.assertEqual(s1, s8)


if __name__ == "__main__":
    unittest.main()

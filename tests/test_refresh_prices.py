"""
Tests for parsers/refresh_prices.py.

Covers:

  - `_max_date` (sidebar / orchestrator helper). The status filter must
    exclude cash sweep rows (their as_of_date is always today's calendar
    date, which would mask a stale equity fetch). Missing files and empty
    frames return None rather than raising.

  - Orchestration invariants. The Phase 1B audit found that
    fetch_benchmark.py / fetch_dividends.py / build_benchmark_total_return.py
    were absent from the refresh pipeline, so SPY TR silently drifted
    behind the other Polygon files and the dashboard's reindex+ffill
    biased β/α. Tests below pin the bench pipeline membership so a future
    refactor can't drop a step and reintroduce the gap.

  - `bench_tr_staleness_days`. Drives the sidebar warning that fires when
    benchmark_spy_tr.csv is behind daily_prices.csv. Counts in trading
    days (not calendar days), tolerates missing inputs, and returns 0
    when the TR file is current.
"""
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "parsers"))

from refresh_prices import (  # noqa: E402
    _max_date,
    bench_tr_staleness_days,
    classify_exit,
    FETCHERS,
    EXTRA_FETCHERS,
    POST_STEPS,
    _extra_flags,
)


class TestMaxDate(unittest.TestCase):
    def test_returns_max_when_no_status_filter(self) -> None:
        with TemporaryDirectory() as td:
            csv = Path(td) / "p.csv"
            pd.DataFrame([
                {"date": "2026-05-15"},
                {"date": "2026-05-18"},
                {"date": "2026-05-12"},
            ]).to_csv(csv, index=False)
            self.assertEqual(_max_date(csv, "date", set()), "2026-05-18")

    def test_excludes_cash_rows_so_equity_date_wins(self) -> None:
        # Cash rows stamped today (2026-05-19) must be filtered; equity row
        # at 2026-05-15 then becomes the reported max.
        with TemporaryDirectory() as td:
            csv = Path(td) / "p.csv"
            pd.DataFrame([
                {"as_of_date": "2026-05-19", "status": "cash_fixed_1"},
                {"as_of_date": "2026-05-15", "status": "ok"},
                {"as_of_date": "2026-05-15", "status": "ok"},
            ]).to_csv(csv, index=False)
            self.assertEqual(
                _max_date(csv, "as_of_date", {"cash_fixed_1"}),
                "2026-05-15",
            )

    def test_missing_file_returns_none(self) -> None:
        self.assertIsNone(_max_date(Path("/no/such/file.csv"), "date", set()))

    def test_empty_after_filter_returns_none(self) -> None:
        with TemporaryDirectory() as td:
            csv = Path(td) / "p.csv"
            pd.DataFrame([
                {"as_of_date": "2026-05-19", "status": "cash_fixed_1"},
            ]).to_csv(csv, index=False)
            self.assertIsNone(
                _max_date(csv, "as_of_date", {"cash_fixed_1"})
            )

    def test_status_filter_ignored_when_column_absent(self) -> None:
        # daily_prices / long_history don't have a status column. The exclude
        # set must be a noop in that case (rather than KeyError-ing).
        with TemporaryDirectory() as td:
            csv = Path(td) / "p.csv"
            pd.DataFrame([{"date": "2026-05-18"}]).to_csv(csv, index=False)
            self.assertEqual(
                _max_date(csv, "date", {"cash_fixed_1"}),
                "2026-05-18",
            )


class TestPipelineMembership(unittest.TestCase):
    """Lock the bench scripts into the orchestration so a future refactor
    can't silently reintroduce the staleness gap. See module docstring."""

    @staticmethod
    def _scripts(group: list) -> set[str]:
        return {script for (script, *_rest) in group}

    @staticmethod
    def _csvs(group: list) -> set[str]:
        return {out.name for (_s, out, *_rest) in group}

    def test_fetch_benchmark_is_in_strict_fetchers(self) -> None:
        # benchmark_spy.csv aligns with daily_prices on the same trading
        # day, so it must be in the strict date-match group.
        self.assertIn("parsers/fetch_benchmark.py", self._scripts(FETCHERS))
        self.assertIn("benchmark_spy.csv", self._csvs(FETCHERS))

    def test_fetch_dividends_is_in_extra_fetchers(self) -> None:
        # Dividends are sparse (~quarterly for SPY), so the last ex-date
        # in the file is not "today" — must be excluded from the strict
        # date-match check.
        self.assertIn("parsers/fetch_dividends.py",
                       self._scripts(EXTRA_FETCHERS))
        self.assertIn("dividends_spy.csv", self._csvs(EXTRA_FETCHERS))

    def test_fetch_ff_factors_is_in_extra_fetchers(self) -> None:
        # French publishes monthly with a multi-week lag, so the file's last
        # month never matches the Polygon trading day — non-strict group.
        self.assertIn("parsers/fetch_ff_factors.py",
                      self._scripts(EXTRA_FETCHERS))
        self.assertIn("ff_factors_monthly.csv", self._csvs(EXTRA_FETCHERS))

    def test_build_benchmark_tr_is_in_post_steps(self) -> None:
        # build_tr is a derived transform, not a fetcher — must run after
        # both its inputs (benchmark_spy.csv + dividends_spy.csv) land.
        self.assertIn("parsers/build_benchmark_total_return.py",
                       self._scripts(POST_STEPS))
        self.assertIn("benchmark_spy_tr.csv", self._csvs(POST_STEPS))

    def test_no_bench_script_landed_in_wrong_group(self) -> None:
        # Defensive: each bench artifact appears in exactly one group.
        all_csvs = (
            list(self._csvs(FETCHERS))
            + list(self._csvs(EXTRA_FETCHERS))
            + list(self._csvs(POST_STEPS))
        )
        for name in ("benchmark_spy.csv", "dividends_spy.csv",
                     "benchmark_spy_tr.csv"):
            self.assertEqual(all_csvs.count(name), 1,
                              f"{name} appears in multiple pipeline groups")

    def test_agg_leg_is_orchestrated(self) -> None:
        # DA-B-2: the 60/40 blend's AGG leg was orchestration-orphaned —
        # nothing refreshed it, so it went stale and silently truncated
        # every JPM-scoped daily benchmark join. All three AGG artifacts
        # must ride the same pipeline as their SPY twins, with the
        # explicit ticker in the entry's extra argv.
        def entry(group, csv_name):
            hits = [e for e in group if e[1].name == csv_name]
            self.assertEqual(len(hits), 1, csv_name)
            return hits[0]

        self.assertEqual(entry(FETCHERS, "benchmark_agg.csv")[4], ["AGG"])
        self.assertEqual(entry(EXTRA_FETCHERS, "dividends_agg.csv")[4],
                         ["AGG"])
        self.assertEqual(entry(POST_STEPS, "benchmark_agg_tr.csv")[4],
                         ["AGG"])

    def test_agg_dividends_entry_is_not_a_second_sweep(self) -> None:
        # The universe --holdings sweep belongs to the SPY-default entry
        # only; the AGG entry is a single-ticker pull.
        from refresh_prices import _extra_flags
        self.assertIn("--holdings",
                      _extra_flags("parsers/fetch_dividends.py", False, []))
        self.assertNotIn("--holdings",
                         _extra_flags("parsers/fetch_dividends.py", False,
                                      ["AGG"]))


class TestBenchTrStaleness(unittest.TestCase):
    """Lock the sidebar staleness check."""

    def _make_dir(self, td: str, bench_dates: list[str] | None,
                  daily_dates: list[str] | None) -> Path:
        data = Path(td)
        if bench_dates is not None:
            pd.DataFrame({"date": bench_dates}).to_csv(
                data / "benchmark_spy_tr.csv", index=False)
        if daily_dates is not None:
            pd.DataFrame({"date": daily_dates, "symbol": ["SPY"] * len(daily_dates),
                          "close": [400.0] * len(daily_dates)}).to_csv(
                data / "daily_prices.csv", index=False)
        return data

    def test_returns_zero_when_tr_matches_daily(self) -> None:
        with TemporaryDirectory() as td:
            data = self._make_dir(td,
                                  bench_dates=["2026-05-18", "2026-05-19", "2026-05-22"],
                                  daily_dates=["2026-05-18", "2026-05-19", "2026-05-22"])
            self.assertEqual(bench_tr_staleness_days(data), 0)

    def test_counts_trading_days_not_calendar_days(self) -> None:
        # TR ends Friday 05-15; daily runs through Friday 05-22. Calendar gap
        # is 7 days but only 5 trading days are between them in daily_prices.
        with TemporaryDirectory() as td:
            data = self._make_dir(
                td,
                bench_dates=["2026-05-13", "2026-05-14", "2026-05-15"],
                daily_dates=["2026-05-13", "2026-05-14", "2026-05-15",
                             "2026-05-18", "2026-05-19", "2026-05-20",
                             "2026-05-21", "2026-05-22"],
            )
            self.assertEqual(bench_tr_staleness_days(data), 5)

    def test_returns_zero_when_tr_is_ahead(self) -> None:
        # Shouldn't happen in practice (TR is derived from prices) but the
        # helper must not return a negative count.
        with TemporaryDirectory() as td:
            data = self._make_dir(td,
                                  bench_dates=["2026-05-22", "2026-05-23"],
                                  daily_dates=["2026-05-22"])
            self.assertEqual(bench_tr_staleness_days(data), 0)

    def test_returns_none_when_either_file_missing(self) -> None:
        with TemporaryDirectory() as td:
            # neither file
            self.assertIsNone(bench_tr_staleness_days(Path(td)))
            # only bench
            self._make_dir(td, bench_dates=["2026-05-15"], daily_dates=None)
            self.assertIsNone(bench_tr_staleness_days(Path(td)))
        with TemporaryDirectory() as td:
            # only daily
            self._make_dir(td, bench_dates=None, daily_dates=["2026-05-22"])
            self.assertIsNone(bench_tr_staleness_days(Path(td)))

    def test_ticker_parameter_reads_that_legs_file(self) -> None:
        # DA-B-2: the guard was hard-coded to SPY, so the AGG leg could go
        # (and did go) stale with no warning anywhere. ticker="AGG" reads
        # benchmark_agg_tr.csv; an absent AGG file stays None (fixture
        # dirs without the blend must not warn).
        with TemporaryDirectory() as td:
            data = self._make_dir(
                td,
                bench_dates=["2026-05-18", "2026-05-19", "2026-05-22"],
                daily_dates=["2026-05-18", "2026-05-19", "2026-05-22"])
            pd.DataFrame({"date": ["2026-05-18"]}).to_csv(
                data / "benchmark_agg_tr.csv", index=False)
            self.assertEqual(bench_tr_staleness_days(data, "AGG"), 2)
            self.assertEqual(bench_tr_staleness_days(data), 0)  # SPY intact
        with TemporaryDirectory() as td:
            data = self._make_dir(td, bench_dates=["2026-05-22"],
                                  daily_dates=["2026-05-22"])
            self.assertIsNone(bench_tr_staleness_days(data, "AGG"))


class TestClassifyExit(unittest.TestCase):
    """The exit-code policy: a flaky non-strict free source (FRED / VIX /
    dividends) must WARN, not fail the whole refresh. Strict Polygon fetchers
    and a cross-source date mismatch stay fatal."""

    def test_clean_run_is_zero(self) -> None:
        self.assertEqual(classify_exit([], [], False), 0)

    def test_soft_only_failure_is_nonfatal(self) -> None:
        # A FRED ReadTimeout (the real-world case) must not sink the refresh.
        self.assertEqual(
            classify_exit([], ["parsers/fetch_risk_free_rate.py"], False), 0)

    def test_strict_failure_is_fatal(self) -> None:
        self.assertEqual(
            classify_exit(["parsers/fetch_daily_prices.py"], [], False), 1)

    def test_date_mismatch_dominates(self) -> None:
        self.assertEqual(classify_exit([], [], True), 2)
        self.assertEqual(classify_exit(["x"], ["y"], True), 2)


class TestExtraFlags(unittest.TestCase):
    """--full only reaches the three incremental fetchers, not the others."""

    def test_full_targets_incremental_fetchers(self):
        self.assertEqual(_extra_flags("parsers/fetch_daily_prices.py", True), ["--full"])
        self.assertEqual(_extra_flags("parsers/fetch_long_history.py", True), ["--full"])
        self.assertEqual(_extra_flags("parsers/fetch_benchmark.py", True), ["--full"])

    def test_full_skips_non_incremental(self):
        self.assertEqual(_extra_flags("parsers/fetch_holding_prices.py", True), [])
        self.assertEqual(_extra_flags("parsers/fetch_risk_free_rate.py", True), [])

    def test_no_full_means_no_extra(self):
        self.assertEqual(_extra_flags("parsers/fetch_daily_prices.py", False), [])

    def test_dividends_always_cover_the_holdings_universe(self):
        # Total-return basis: every refresh covers the whole close-matrix
        # universe + splits (not SPY alone) — a market-wide delta by default,
        # the full per-ticker sweep under --full (Distributions S3).
        self.assertEqual(_extra_flags("parsers/fetch_dividends.py", False), ["--holdings"])
        self.assertEqual(_extra_flags("parsers/fetch_dividends.py", True), ["--full", "--holdings"])


if __name__ == "__main__":
    unittest.main()
